import numpy as np
import torch
from liso.mmdet3d.models.backbones.second import SECOND
from liso.mmdet3d.models.necks.second_fpn import SECONDFPN
from liso.networks.pcl_to_feature_grid.pcl_to_feature_grid import (
    PointsPillarFeatureNetWrapper,
)
from liso.networks.simple_net.simple_net_utils import get_num_dims_per_box_attr
from liso.networks.transfusion.transfusion_bbox_coder import TransFusionBBoxCoder
from liso.networks.transfusion.transfusion_head import TransFusionHead


class TransfusionStyleNet(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        assert cfg.box_prediction.rotation_representation.method == "vector"
        for act in (
            "pos",
            "rot",
        ):  # "probs"):  # "dims", "rot", "probs"):
            assert cfg.box_prediction.activations[act] == "none", (
                act,
                cfg.box_prediction.activations[act],
            )
        self.cfg = cfg
        self.pfn = PointsPillarFeatureNetWrapper(cfg)
        self.pts_backbone = SECOND(
            in_channels=64,
            out_channels=[64, 128, 256],
            layer_nums=[3, 5, 5],
            layer_strides=[2, 2, 2],
            norm_cfg=dict(type="BN", eps=0.001, momentum=0.01),
            conv_cfg=dict(type="Conv2d", bias=False),
        )
        self.pts_neck = SECONDFPN(
            in_channels=[64, 128, 256],
            out_channels=[128, 128, 128],
            upsample_strides=[0.5, 1, 2],
            norm_cfg=dict(type="BN", eps=0.001, momentum=0.01),
            upsample_cfg=dict(type="deconv", bias=False),
            use_conv_for_no_stride=True,
        )
        self.pts_bbox_head = TransFusionHead(
            num_proposals=self.cfg.network.transfusion.num_pred_boxes,
            auxiliary=True,
            in_channels=128 * 3,
            hidden_channel=128,
            num_classes=1,  # len(class_names),
            num_decoder_layers=1,
            num_heads=8,
            learnable_query_pos=False,
            initialize_by_heatmap=True,
            nms_kernel_size=3,
            ffn_channel=256,
            dropout=0.1,
            bn_momentum=0.1,
            activation="relu",
            common_heads=dict(
                center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
            ),
            test_cfg={
                "grid_size": np.array(self.cfg.data.img_grid_size),
                "out_size_factor": self.cfg.network.transfusion.out_size_factor,
                "dataset": "nuScenes",
            },
            loss_cls=dict(
                use_sigmoid=True,
            ),
        )
        self.num_dims_per_attr = get_num_dims_per_box_attr(cfg)
        self.bbox_coder = TransFusionBBoxCoder(
            pc_range=-np.array(self.cfg.data.bev_range_m)
            / 2.0,  # point_cloud_range[:2],
            voxel_size=np.array(self.cfg.data.bev_range_m)
            / np.array(self.cfg.data.img_grid_size),
            out_size_factor=self.cfg.network.transfusion.out_size_factor,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10,
        )

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, "pts_neck") and self.pts_neck is not None

    def forward(self, img_t0, pcls):
        # in [256, 256]

        bev_enc, bev_occupancy_map = self.pfn(pcl_t0=pcls, img_t0=img_t0)
        aux_outputs = {"bev_net_input_dbg": bev_occupancy_map}
        x = self.pts_backbone(bev_enc)
        if self.with_pts_neck:
            x = self.pts_neck(x)

        pred_dict = self.pts_bbox_head(feats=x, img_feats=None, img_metas=None)  # [0]
        assert len(pred_dict) == 1 and len(pred_dict[0]) == 1, pred_dict
        pred_dict = pred_dict[0][0]
        aux_outputs["probs"] = pred_dict["dense_heatmap"].clone().permute((0, 2, 3, 1))
        _, channels_last_shapes = self.bbox_coder.decode(
            heatmap=pred_dict["heatmap"],
            rot=pred_dict["rot"],
            dim=pred_dict["dim"],
            center=pred_dict["center"],
            height=pred_dict["height"],
            vel=pred_dict["vel"],
        )
        for k in channels_last_shapes:
            channels_last_shapes[k] = channels_last_shapes[k][
                :, :, : self.num_dims_per_attr[k]
            ]
        # out: 64, 64
        return channels_last_shapes, aux_outputs
