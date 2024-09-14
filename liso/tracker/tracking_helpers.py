from typing import List

import torch
from liso.utils.torch_transformation import homogenize_pcl


def accumulate_pcl(
    pcls: List[torch.FloatTensor],
    sensor_odometry_ti_tii: List[torch.FloatTensor],
    concat_at_the_end: bool = True,
) -> List[torch.FloatTensor]:
    assert len(pcls) - len(sensor_odometry_ti_tii) <= 1, "size mismatch"
    w_Ts_sti = aggregate_odometry_to_world_poses(sensor_odometry_ti_tii)
    accumulated_pcl = []
    for pcl, w_T_sti in zip(pcls, w_Ts_sti):
        assert len(pcl.shape) == 2, "batching not supported"
        pcl_homog = homogenize_pcl(pcl[:, :3])
        assert torch.all(pcl_homog[:, -1] == 1.0), "not homog"
        pcl_world_homog = torch.einsum(
            "ij,nj->ni", w_T_sti, pcl_homog.to(torch.double)
        ).to(torch.float32)
        pcl_world = torch.concat([pcl_world_homog[:, :3], pcl[:, [3]]], dim=-1)
        accumulated_pcl.append(pcl_world)
    if concat_at_the_end:
        return torch.concat(accumulated_pcl, dim=0)
    else:
        return accumulated_pcl


def aggregate_odometry_to_world_poses(sensor_odometry_ti_tii, w_T_st0_start_pose=None):
    if w_T_st0_start_pose is None:
        w_T_st0 = torch.eye(
            4, dtype=torch.float64, device=sensor_odometry_ti_tii[0].device
        )
    else:
        w_T_st0 = w_T_st0_start_pose
    w_Ts_sti = [
        w_T_st0,
    ]
    for sti_T_stii in sensor_odometry_ti_tii:
        assert w_Ts_sti[-1].dtype == torch.float64
        assert sti_T_stii.dtype == torch.float64
        w_Ts_sti.append(w_Ts_sti[-1] @ sti_T_stii)
    return torch.stack(w_Ts_sti, dim=0)
