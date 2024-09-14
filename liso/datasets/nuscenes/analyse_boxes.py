#!/usr/bin/env python3
import numpy as np
import torch


def voxelize_pcl(pcl_np, grid_range_m_np, grid_size):
    pointwise_voxel_coors_all_pts = (pcl_np[:, :3] + 0.5 * grid_range_m_np) / (
        grid_range_m_np
    )
    if torch.is_tensor(pcl_np):
        pointwise_voxel_coors_all_pts = (pointwise_voxel_coors_all_pts * grid_size).to(
            torch.int32
        )
    else:
        pointwise_voxel_coors_all_pts = (
            pointwise_voxel_coors_all_pts * grid_size
        ).astype(np.int32)
    point_is_in_voxel = (
        (0 <= pointwise_voxel_coors_all_pts[:, 0])
        & (0 <= pointwise_voxel_coors_all_pts[:, 1])
        & (0 <= pointwise_voxel_coors_all_pts[:, 2])
        & (pointwise_voxel_coors_all_pts[:, 0] < grid_size[0])
        & (pointwise_voxel_coors_all_pts[:, 1] < grid_size[1])
        & (pointwise_voxel_coors_all_pts[:, 2] < grid_size[2])
    )
    return pointwise_voxel_coors_all_pts, point_is_in_voxel
