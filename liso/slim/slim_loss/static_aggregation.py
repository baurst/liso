#!/usr/bin/env python3


import torch
from liso.slim.slim_loss.weighted_pc_alignment import weighted_pc_alignment


def batched_grid_data_to_pointwise_data(
    grid_data,
    pointwise_voxel_coordinates_fs,
    pointwise_valid_mask,
    default_value,
):
    assert len(grid_data.shape) == 4, grid_data.shape
    # pointwise_data = default_value * torch.ones(
    #     pointwise_voxel_coordinates_fs.shape[1],
    #     grid_data.shape[-1],
    #     device=grid_data.device,
    #     dtype=grid_data.dtype,
    # )
    pointwise_voxel_coordinates_fs[~pointwise_valid_mask] = 0
    pointwise_batch_coors = torch.arange(
        pointwise_valid_mask.shape[0], device=grid_data.device
    )[:, None, None].repeat((1, pointwise_valid_mask.shape[1], 1))
    pointwise_data = grid_data[
        pointwise_batch_coors[..., 0].long(),
        pointwise_voxel_coordinates_fs[..., 0].long(),
        pointwise_voxel_coordinates_fs[..., 1].long(),
    ]
    pointwise_data[~pointwise_valid_mask] = default_value
    return pointwise_data


def compute_batched_bev_static_aggregated_flow(
    pc: torch.FloatTensor,
    pointwise_voxel_coordinates_fs: torch.IntTensor,
    pointwise_valid_mask: torch.ByteTensor,
    static_flow_bev: torch.FloatTensor,
    staticness_weights: torch.FloatTensor,
    voxel_center_metric_coordinates_bev: torch.FloatTensor,
    use_eps_for_weighted_pc_alignment: bool = False,
):
    bs = staticness_weights.shape[0]
    aggflows = []
    Ts = []
    neps = []
    assert len(static_flow_bev.shape) == 4
    assert static_flow_bev.shape[-1] == 2
    static_3d_flow_grid = torch.cat(
        [static_flow_bev, torch.zeros_like(static_flow_bev[..., :1])],
        dim=-1,
    )
    assert torch.all(
        pointwise_voxel_coordinates_fs[pointwise_valid_mask] >= 0
    ), "negative pixel coordinates found"
    for i in range(1, 3):
        assert torch.all(
            pointwise_voxel_coordinates_fs < static_3d_flow_grid.shape[i]
        ), "too large pixel coordinates found"

    pointwise_flow = batched_grid_data_to_pointwise_data(
        static_3d_flow_grid,
        pointwise_voxel_coordinates_fs,
        pointwise_valid_mask,
        default_value=0.0,
    )

    pointwise_staticness = batched_grid_data_to_pointwise_data(
        staticness_weights[..., None],
        pointwise_voxel_coordinates_fs,
        pointwise_valid_mask,
        default_value=0.0,
    )[..., 0]

    pc0_grid = torch.cat(
        [
            voxel_center_metric_coordinates_bev,
            torch.zeros_like(voxel_center_metric_coordinates_bev[..., :1]),
        ],
        dim=-1,
    )
    assert pc0_grid.shape == static_3d_flow_grid.shape[1:]
    for b in range(bs):
        T, not_enough_points = weighted_pc_alignment(
            pc[b][pointwise_valid_mask[b]][..., :3],
            (pc[b][..., :3] + pointwise_flow[b])[pointwise_valid_mask[b]],
            pointwise_staticness[b][pointwise_valid_mask[b]],
            use_epsilon_on_weights=use_eps_for_weighted_pc_alignment,
        )

        static_aggr_flow = torch.einsum(
            "ij,hwj->hwi",
            T - torch.eye(4, dtype=torch.float64, device=T.device),
            torch.cat(
                [
                    pc0_grid,
                    torch.ones_like(pc0_grid[..., 0][..., None]),
                ],
                axis=-1,
            ),
        )[..., 0:2].float()

        aggflows.append(static_aggr_flow)
        Ts.append(T)
        neps.append(not_enough_points)
    return (
        torch.stack(aggflows, dim=0),
        torch.stack(Ts, dim=0),
        torch.stack(neps, dim=0),
    )
