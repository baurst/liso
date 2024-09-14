from copy import deepcopy

import torch
from liso.kabsch.shape_utils import Shape
from liso.networks.pcl_to_feature_grid.pcl_to_feature_grid import (
    PointsPillarFeatureNetWrapper,
)
from liso.utils.torch_transformation import torch_decompose_matrix

# from mmdet3d.models.dense_heads.anchor3d_head import
from mmdet3d.core.anchor.anchor_3d_generator import (  # trigger REGISTER_MODULES
    AlignedAnchor3DRangeGenerator,
)
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmdet3d.models.detectors.mvx_faster_rcnn import MVXFasterRCNN
from omegaconf import OmegaConf


class PointPillarsWrapper(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        # voxel_size = [0.25, 0.25, 8]
        voxel_size = [0.25, 0.25, 2 * cfg.data.z_pillar_cutoff_value]

        # self.pfn = PointsPillarFeatureNetWrapper(cfg)

        point_cloud_range = [
            -cfg.data.bev_range_m[0] / 2,
            -cfg.data.bev_range_m[1] / 2,
            -cfg.data.z_pillar_cutoff_value,
            cfg.data.bev_range_m[0] / 2,
            cfg.data.bev_range_m[1] / 2,
            cfg.data.z_pillar_cutoff_value,
        ]
        anchor_range = deepcopy(point_cloud_range)
        anchor_range[2] = -1.8
        anchor_range[-1] = -1.8

        self.model = MVXFasterRCNN(
            pts_voxel_layer=dict(
                max_num_points=32,
                # point_cloud_range=[-50, -50, -5, 50, 50, 3],
                deterministic=False,
                point_cloud_range=point_cloud_range,
                voxel_size=voxel_size,
                max_voxels=(20000, 30000),
            ),
            pts_voxel_encoder=dict(
                type="HardVFE",
                in_channels=4,
                feat_channels=[64, 64],
                with_distance=False,
                voxel_size=voxel_size,
                with_cluster_center=True,
                with_voxel_center=True,
                point_cloud_range=point_cloud_range,
                # point_cloud_range=[-50, -50, -5, 50, 50, 3],
                norm_cfg=dict(type="naiveSyncBN1d", eps=1e-3, momentum=0.01),
            ),
            pts_middle_encoder=dict(
                type="PointPillarsScatter", in_channels=64, output_shape=[400, 400]
            ),
            pts_backbone=dict(
                type="SECOND",
                in_channels=64,
                norm_cfg=dict(type="naiveSyncBN2d", eps=1e-3, momentum=0.01),
                layer_nums=[3, 5, 5],
                layer_strides=[2, 2, 2],
                out_channels=[64, 128, 256],
            ),
            pts_neck=dict(
                type="FPN",
                norm_cfg=dict(type="naiveSyncBN2d", eps=1e-3, momentum=0.01),
                act_cfg=dict(type="ReLU"),
                in_channels=[64, 128, 256],
                out_channels=256,
                start_level=0,
                num_outs=3,
            ),
            pts_bbox_head=dict(
                type="Anchor3DHead",
                num_classes=1,  # baurst: was 10
                in_channels=256,
                feat_channels=256,
                use_direction_classifier=True,
                anchor_generator=dict(
                    type="AlignedAnchor3DRangeGenerator",
                    ranges=[anchor_range],
                    scales=[1, 2, 4],
                    sizes=[
                        [2.5981, 0.8660, 1.0],  # 1.5 / sqrt(3)
                        [1.7321, 0.5774, 1.0],  # 1 / sqrt(3)
                        [1.0, 1.0, 1.0],
                        [0.4, 0.4, 1],
                    ],
                    # custom_values=[0, 0],
                    rotations=[0, 1.57],
                    reshape_out=True,
                ),
                assigner_per_size=False,
                diff_rad_by_sin=True,
                # dir_offset=-0.7854,  # -pi / 4
                bbox_coder=dict(type="DeltaXYZWLHRBBoxCoder", code_size=7),
                loss_cls=dict(
                    type="FocalLoss",
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0,
                ),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0),
                loss_dir=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.2
                ),
            ),
            # model training and testing settings
            train_cfg=OmegaConf.create(
                dict(
                    pts=dict(
                        assigner=dict(
                            type="MaxIoUAssigner",
                            iou_calculator=dict(type="BboxOverlapsNearest3D"),
                            pos_iou_thr=0.6,
                            neg_iou_thr=0.3,
                            min_pos_iou=0.3,
                            ignore_iof_thr=-1,
                        ),
                        allowed_border=0,
                        code_weight=[
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ],
                        pos_weight=-1,
                        debug=False,
                    )
                )
            ),
            test_cfg=OmegaConf.create(
                dict(
                    pts=dict(
                        use_rotate_nms=True,
                        nms_across_levels=False,
                        nms_pre=1000,
                        nms_thr=0.2,
                        score_thr=0.05,
                        min_bbox_size=0,
                        max_num=500,
                    )
                )
            ),
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
            # losses are per level? -> sum them
            loss_dict = {k: sum(v) for k, v in loss_dict.items()}
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
                box_tensor = box_result["pts_bbox"]["boxes_3d"].tensor

                box_ours = Shape(
                    pos=box_tensor[:, 0:3],
                    dims=box_tensor[:, 3:6],
                    rot=box_tensor[:, [6]],
                    probs=box_result["pts_bbox"]["scores_3d"][..., None],
                )
                # predicts box center at the bottom of the box, but we want it at the center
                box_ours.pos[:, 2] += 0.5 * box_ours.dims[:, 2]
                boxes_ours.append(box_ours)
            boxes_ours = Shape.from_list_of_shapes(boxes_ours)
            boxes_ours = boxes_ours.to(
                pcls[0].device
            )  # for some reason bbox3d2result dumps the boxes to cpu
            return boxes_ours
