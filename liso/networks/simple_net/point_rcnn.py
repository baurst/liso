# see https://github.com/open-mmlab/mmdetection3d/blob/main/configs/_base_/models/point_rcnn.py

from copy import deepcopy

import torch
from liso.kabsch.shape_utils import Shape
from liso.utils.torch_transformation import torch_decompose_matrix

# need to trigger the whole register module stuff, so import it here
from mmcv.ops.roipoint_pool3d import RoIPointPool3d  # noqa: F401
from mmdet3d.core.bbox.coders.point_xyzwhlr_bbox_coder import PointXYZWHLRBBoxCoder
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmdet3d.models.backbones.pointnet2_sa_msg import PointNet2SAMSG
from mmdet3d.models.dense_heads.point_rpn_head import PointRPNHead
from mmdet3d.models.detectors.point_rcnn import PointRCNN
from mmdet3d.models.necks.pointnet2_fp_neck import PointNetFPNeck
from mmdet3d.models.roi_heads.bbox_heads.point_rcnn_bbox_head import PointRCNNBboxHead
from mmdet3d.models.roi_heads.point_rcnn_roi_head import PointRCNNRoIHead

# need to trigger the whole register module stuff, so import it here
from mmdet3d.models.roi_heads.roi_extractors.single_roipoint_extractor import (  # noqa: F401
    Single3DRoIPointExtractor,
)
from mmdet.models.losses import FocalLoss, SmoothL1Loss
from omegaconf import OmegaConf


class PointRCNNWrapper(torch.nn.Module):
    def __init__(self, cfg):
        super(PointRCNNWrapper, self).__init__()
        self.cfg = cfg
        backbone = PointNet2SAMSG(
            in_channels=4,
            num_points=(4096, 1024, 256, 64),
            radii=((0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0)),
            num_samples=((16, 32), (16, 32), (16, 32), (16, 32)),
            sa_channels=(
                ((16, 16, 32), (32, 32, 64)),
                ((64, 64, 128), (64, 96, 128)),
                ((128, 196, 256), (128, 196, 256)),
                ((256, 256, 512), (256, 384, 512)),
            ),
            fps_mods=(("D-FPS"), ("D-FPS"), ("D-FPS"), ("D-FPS")),
            fps_sample_range_lists=((-1), (-1), (-1), (-1)),
            aggregation_channels=(None, None, None, None),
            dilated_group=(False, False, False, False),
            out_indices=(0, 1, 2, 3),
            norm_cfg=dict(type="BN2d", eps=1e-3, momentum=0.1),
            sa_cfg=dict(
                type="PointSAModuleMSG",
                pool_mod="max",
                use_xyz=True,
                normalize_xyz=False,
            ),
        )
        neck = PointNetFPNeck(
            fp_channels=(
                (1536, 512, 512),
                (768, 512, 512),
                (608, 256, 256),
                (257, 128, 128),
            )
        )

        train_cfg = OmegaConf.create(
            {
                "nms_cfg": {
                    "multi_classes_nms": False,
                    # "nms_type": "nms_gpu",
                    "nms_type": "iou_3d",
                    # "nms_thresh": 0.8,
                    "iou_thr": 0.8,
                    # "nms_pre_maxsize": 4096,
                    "nms_pre": 4096,
                    # "nms_post_maxsize": 500,
                    "nms_post": 500,
                    "use_rotate_nms": True,
                },
                "assigner": {
                    "type": "MaxIoUAssigner",
                    "iou_calculator": dict(type="BboxOverlaps3D", coordinate="lidar"),
                    "pos_iou_thr": 0.55,
                    "neg_iou_thr": 0.55,
                    "min_pos_iou": 0.55,
                    "ignore_iof_thr": -1,
                    "gt_max_assign_all": False,
                },
                "sampler": dict(
                    type="IoUNegPiecewiseSampler",
                    num=128,  # = roi_per_image
                    pos_fraction=0.5,  # = fg_ratio
                    neg_piece_fractions=[
                        0.8,
                        0.2,
                    ],  # = [hard_bg_ratio, 1 - hard_bg_ratio]
                    neg_iou_piece_thrs=[
                        0.45,
                        0.1,
                    ],  # = [cls_bg_thresh, cls_bg_thresh_lo]
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False,
                    return_iou=True,
                ),
                # # "box_coder": ResidualCoder,
                # "type": "IoUNegPiecewiseSampler",
                # "roi_per_image": 128,
                # "fg_ratio": 0.5,
                # "sample_roi_by_each_class": True,
                # "cls_score_type": "cls",
                "cls_pos_thr": 0.6,  # modest differs from the default value for pointrcnn 0.8
                "cls_neg_thr": 0.45,  # modest differs from the default value for pointrcnn 0.25
                # "cls_bg_thresh_lo": 0.1,
                # "hard_bg_ratio": 0.8,
                # "reg_fg_thresh": 0.55,
                "score_thr": None,
                "nms_thr": None,
                "use_rotate_nms": True,
            }
        )

        test_cfg = deepcopy(train_cfg)
        test_cfg.use_rotate_nms = True
        test_cfg.score_thr = 0.1
        test_cfg.nms_thr = 0.1

        rpn_head = PointRPNHead(
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            num_classes=1,
            enlarge_width=0.1,
            pred_layer_cfg=OmegaConf.create(
                {
                    "in_channels": 128,
                    "cls_linear_channels": (256, 256),
                    "reg_linear_channels": (256, 256),
                }
            ),
            cls_loss=FocalLoss(
                use_sigmoid=True,
                reduction="sum",
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            ),
            bbox_loss=SmoothL1Loss(beta=1.0 / 9.0, reduction="sum", loss_weight=1.0),
            bbox_coder=PointXYZWHLRBBoxCoder(
                code_size=8,
                # code_size: (center residual (3), size regression (3),
                #             torch.cos(yaw) (1), torch.sin(yaw) (1)
                use_mean_size=True,
                mean_size=[[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]],
            ),
        )
        roi_head = PointRCNNRoIHead(
            train_cfg=train_cfg,
            # test_cfg=test_cfg,
            test_cfg=test_cfg,
            point_roi_extractor=dict(
                type="Single3DRoIPointExtractor",
                roi_layer=dict(type="RoIPointPool3d", num_sampled_points=512),
            ),
            bbox_head=PointRCNNBboxHead(
                num_classes=1,
                pred_layer_cfg=OmegaConf.create(
                    dict(
                        in_channels=512,
                        cls_conv_channels=(256, 256),
                        reg_conv_channels=(256, 256),
                        bias=True,
                    )
                ),
                in_channels=5,
                # 5 = 3 (xyz) + scores + depth
                mlp_channels=[128, 128],
                num_points=(128, 32, -1),
                radius=(0.2, 0.4, 100),
                num_samples=(16, 16, 16),
                sa_channels=((128, 128, 128), (128, 128, 256), (256, 256, 512)),
                with_corner_loss=True,
            ),
            depth_normalizer=70.0,
        )
        self.model = PointRCNN(
            backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=None,
            test_cfg=None,
        )

    def forward(self, pcls, gt_boxes, train=True):
        pcls_padded = torch.nn.utils.rnn.pad_sequence(
            pcls, batch_first=True, padding_value=0.0
        )
        pcls = [pcl for pcl in pcls_padded]
        input_metas = [{"box_type_3d": LiDARInstance3DBoxes} for _ in pcls]
        if train:
            gt_boxes_list = gt_boxes.into_list_of_shapes()
            gt_bboxes_mmdet = []
            gt_labels_mmdet = []
            for boxes in gt_boxes_list:
                box_poses_gt = boxes.get_poses()
                box_pos, box_rot = torch_decompose_matrix(box_poses_gt)
                # box_pos_orig = box_pos.clone()

                # the code assumes box center at the bottom of the box
                box_pos[:, 2] -= 0.5 * boxes.dims[:, 2]

                gt_boxes_tensor = torch.cat(
                    [
                        box_pos,
                        boxes.dims,
                        box_rot,
                    ],
                    dim=-1,
                )
                mmdet_gt_bbox = LiDARInstance3DBoxes(
                    # boxes3d,
                    gt_boxes_tensor,
                    box_dim=7,
                )

                # call tensor on gt box
                # transform the bbox coordinate to the point cloud coordinate
                # START point_rpn_head test
                # point_rpn_head_gt_bboxes_3d_tensor = mmdet_gt_bbox.tensor.clone()
                # point_rpn_head_gt_bboxes_3d_tensor[..., 2] += point_rpn_head_gt_bboxes_3d_tensor[..., 5] / 2
                # assert torch.allclose(point_rpn_head_gt_bboxes_3d_tensor[..., 2], box_pos_orig[:, 2].to(point_rpn_head_gt_bboxes_3d_tensor.dtype))
                # END point_rpn_head test

                gt_bboxes_mmdet.append(mmdet_gt_bbox)
                # HERE WE MAKE ALL MOVABLE OBJECTS HAVE THE SAME CLASS
                gt_labels_mmdet.append(
                    torch.squeeze(0 * boxes.class_id, dim=-1).to(torch.long)
                )

            loss_dict = self.model(
                points=pcls,
                img_metas=input_metas,
                gt_bboxes_3d=gt_bboxes_mmdet,
                gt_labels_3d=gt_labels_mmdet,
            )
            # some losses are per box and thus not 0-D -> take their mean
            for loss_name, loss_val in loss_dict.items():
                if torch.numel(loss_val) > 1:
                    loss_dict[loss_name] = torch.mean(loss_val)
            return loss_dict
        else:
            bbox_results = [
                # for some reason simple_test assumes single batch size??
                self.model.simple_test(
                    points=[
                        pcl,
                    ],
                    img_metas=[
                        input_meta,
                    ],
                )[0]
                for pcl, input_meta in zip(pcls, input_metas)
            ]
            boxes_ours = []
            for box_result in bbox_results:
                box_tensor = box_result["boxes_3d"].tensor

                box_ours = Shape(
                    pos=box_tensor[:, 0:3],
                    dims=box_tensor[:, 3:6],
                    rot=box_tensor[:, [6]],
                    probs=box_result["scores_3d"][..., None],
                )
                # predicts box center at the bottom of the box, but we want it at the center
                box_ours.pos[:, 2] += 0.5 * box_ours.dims[:, 2]
                boxes_ours.append(box_ours)
            boxes_ours = Shape.from_list_of_shapes(boxes_ours)
            return boxes_ours
