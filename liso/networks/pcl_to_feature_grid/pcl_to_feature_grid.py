import numpy as np
import torch
from mmcv.ops import Voxelization
from mmcv.runner import force_fp32
from mmdet3d.models.middle_encoders.pillar_scatter import PointPillarsScatter
from mmdet3d.models.voxel_encoders.pillar_encoder import PillarFeatureNet
from torch.nn import functional as F


class PointsPillarFeatureNetWrapper(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        z_pillar_cutoff_value = cfg.data.setdefault("z_pillar_cutoff_value", 5.0)
        assert z_pillar_cutoff_value > 0.0, z_pillar_cutoff_value
        pc_range_half = np.append(
            np.array(self.cfg.data.bev_range_m) / 2.0, z_pillar_cutoff_value
        )
        pc_range = np.concatenate([-pc_range_half, pc_range_half], axis=0)
        voxel_size = np.array(self.cfg.data.bev_range_m) / np.array(
            self.cfg.data.img_grid_size
        )
        voxel_size = np.append(voxel_size, 2 * z_pillar_cutoff_value)
        self.pts_voxel_layer = Voxelization(
            max_num_points=20,
            voxel_size=voxel_size,
            max_voxels=(40000, 40000),
            point_cloud_range=pc_range,
            deterministic=False,
        )
        if "use_lidar_intensity" in self.cfg.data:
            num_input_channels = [3, 4][self.cfg.data.use_lidar_intensity]
        else:
            print("Warning - legacy mode: Using no lidar intensity!")
            num_input_channels = 3

        channel_reduction_factor = cfg.network.centerpoint.setdefault(
            "channel_reduction_factor", 1
        )
        self.pts_voxel_encoder = PillarFeatureNet(
            in_channels=num_input_channels,  # 5 with intensity + layer index,
            feat_channels=[64 // channel_reduction_factor],
            with_distance=False,
            voxel_size=voxel_size,
            norm_cfg={"type": "BN1d", "eps": 0.001, "momentum": 0.01},
            point_cloud_range=pc_range,
        )
        self.pts_middle_encoder = PointPillarsScatter(
            in_channels=64 // channel_reduction_factor,
            output_shape=self.cfg.data.img_grid_size,
        )
        self.debug_occupancy_pts_middle_encoder = PointPillarsScatter(
            in_channels=1, output_shape=self.cfg.data.img_grid_size
        )

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        # copied from TransFusion/mmdet3d/models/detectors/transfusion.py
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            # BAURST TODO: reverse row_col axis to match our convention
            res_coors = res_coors[:, [0, 2, 1]]
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode="constant", value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def extract_pts_feat(self, pts):
        # copied from TransFusion/mmdet3d/models/detectors/transfusion.py
        """Extract features of points."""
        assert isinstance(pts, (list, tuple)), type(pts)
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(
            voxels,
            num_points,
            coors,
        )
        batch_size = len(pts)
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)

        occupancy_grid = self.debug_occupancy_pts_middle_encoder(
            torch.ones_like(voxel_features[:, [0]]), coors, batch_size
        )
        return x, occupancy_grid

    def forward(self, pcl_t0, img_t0=None):
        bev_features = self.extract_pts_feat(pcl_t0)

        return bev_features
