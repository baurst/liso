import torch


def scatter_pointwise2bev(values, pointwise_voxel_coors, pointwise_mask, grid_size):
    assert pointwise_voxel_coors.dtype in (
        torch.int,
        torch.int32,
        torch.int64,
        torch.long,
    ), pointwise_voxel_coors.dtype
    assert pointwise_mask.dtype == torch.bool
    assert len(pointwise_voxel_coors.size()) == 3
    assert len(pointwise_mask.size()) == 2
    assert (
        values.size()[:2]
        == pointwise_voxel_coors.size()[:2]
        == pointwise_mask.size()[:2]
    )
    assert pointwise_voxel_coors.shape[2] == 2

    values_shape = values.size()
    bs, max_num_points = values_shape[:2]
    for s in values_shape[2:]:
        assert isinstance(s, int)

    assert (
        torch.amin(pointwise_voxel_coors) >= 0
    ), "negative voxel coors dont make sense"

    assert torch.all(
        torch.amax(pointwise_voxel_coors, axis=[0, 1])
        < torch.tensor(grid_size, device=pointwise_voxel_coors.device)
    ), "too large voxel coors for grid size %s" % str(grid_size)
    assert torch.all(pointwise_voxel_coors[~pointwise_mask] == 0), (
        "not all masked voxel coors were set too 0",
        (
            pointwise_voxel_coors[~pointwise_mask].size(),
            pointwise_voxel_coors[~pointwise_mask].size(),
        ),
    )

    scatter_nd_inds = torch.cat(
        [
            torch.broadcast_to(
                torch.arange(bs, device=pointwise_voxel_coors.device)[:, None],
                [bs, max_num_points],
            )[..., None],
            pointwise_voxel_coors,
        ],
        axis=-1,
    )
    target_shape = [bs, *grid_size] + list(values_shape[2:])
    result = torch.zeros(target_shape, dtype=values.dtype, device=values.device)

    result[
        scatter_nd_inds[..., 0], scatter_nd_inds[..., 1], scatter_nd_inds[..., 2]
    ] = torch.where(pointwise_mask[..., None], values, torch.zeros_like(values))
    bev_valid_mask = torch.zeros(
        size=[bs, *grid_size], dtype=torch.bool, device=pointwise_mask.device
    )
    bev_valid_mask[
        scatter_nd_inds[..., 0], scatter_nd_inds[..., 1], scatter_nd_inds[..., 2]
    ] = pointwise_mask

    return result, bev_valid_mask
