from typing import Tuple

import numpy as np
import torch


def pillarize_pointcloud(
    pcl_np: np.ndarray, bev_range_m: np.ndarray, pillar_bev_resolution: np.ndarray
):
    assert len(pcl_np.shape) == 2, pcl_np.shape
    pointwise_pillars_coords = (pcl_np[:, :2] + 0.5 * bev_range_m) / bev_range_m
    pointwise_pillars_coords = (
        pointwise_pillars_coords * pillar_bev_resolution.astype(np.float32)
    ).astype(np.int32)
    pointwise_pillars_coords = np.minimum(
        pointwise_pillars_coords, pillar_bev_resolution - 1
    )
    pointwise_pillars_coords = np.maximum(pointwise_pillars_coords, 0)
    return pointwise_pillars_coords


def torch_batched_pillarize_pointcloud(
    pcl_torch: torch.Tensor,
    bev_range_m: torch.Tensor,
    pillar_bev_resolution: torch.Tensor,
):
    pointwise_pillars_coords = (
        pcl_torch[:, :, :2] + 0.5 * bev_range_m[None, None, ...]
    ) / bev_range_m[None, None, ...]
    pointwise_pillars_coords = (
        pointwise_pillars_coords * pillar_bev_resolution.to(torch.float32)
    ).to(torch.int32)
    pointwise_pillars_coords = torch.minimum(
        pointwise_pillars_coords, pillar_bev_resolution - 1
    )
    pointwise_pillars_coords = torch.maximum(
        pointwise_pillars_coords, torch.zeros_like(pillar_bev_resolution)
    )
    bs, num_pts, _ = pcl_torch.shape
    batch_coors = torch.arange(bs, device=pcl_torch.device)[:, None, None].repeat(
        (1, num_pts, 1)
    )
    return batch_coors.to(torch.long), pointwise_pillars_coords.to(torch.long)


def create_occupancy_pcl_image(
    pcl,
    bev_range_m: np.ndarray,
    img_shape: np.ndarray,
):
    assert len(pcl.shape) == 2, pcl.shape
    img = np.zeros(tuple(img_shape) + (1,), dtype=np.float32)
    pillar_coords = pillarize_pointcloud(pcl, bev_range_m, img_shape)

    img[pillar_coords[..., 0], pillar_coords[..., 1]] = 1.0
    return img


def create_topdown_f32_pcl_image_variable_extent(
    pcl: torch.FloatTensor,
    intensity: torch.FloatTensor,
    coords_min: torch.FloatTensor,
    coords_max: torch.FloatTensor,
    img_grid_size: torch.IntTensor,
) -> Tuple[torch.IntTensor, torch.Tensor]:
    assert len(pcl.shape) == 2, pcl.shape
    assert pcl.shape[-1] == 4, pcl.shape
    assert len(intensity.shape) == 1, intensity.shape
    assert intensity.shape[0] == pcl.shape[0], (intensity.shape, pcl.shape)
    if intensity.min() < 0.0 or intensity.max() > 1.0:
        intensity = intensity.clone()
        intensity -= intensity.min()
        intensity /= intensity.max()
    assert coords_min.numel() == 2, coords_min
    assert coords_min.numel() == 2, coords_min

    is_inside_bev_extent, linear_image_indices = get_linear_bev_idx(
        pcl, coords_min, coords_max, img_grid_size
    )

    intensity = intensity[is_inside_bev_extent]
    linear_image_indices = linear_image_indices[is_inside_bev_extent]

    intensity_img_1d = torch.zeros(
        torch.prod(img_grid_size), dtype=intensity.dtype, device=intensity.device
    )
    intensity_img_1d = intensity_img_1d.scatter_(
        dim=0,
        index=linear_image_indices,
        src=intensity,
    )
    # pytorch 1.12
    # intensity_img_1d = intensity_img_1d.scatter_reduce(
    #     dim=0,
    #     src=intensity,
    #     index=linear_image_indices,
    #     reduce="amax",
    #     include_self=False,
    # )
    occup_mask_1d = torch.zeros(
        torch.prod(img_grid_size), dtype=torch.bool, device=intensity.device
    )
    occup_mask_1d = occup_mask_1d.scatter_(
        dim=0,
        index=linear_image_indices,
        src=torch.ones_like(pcl[:, 0], dtype=torch.bool),
    )

    intensity_img = intensity_img_1d.reshape(tuple(img_grid_size))
    pixel_is_occupied = occup_mask_1d.reshape(tuple(img_grid_size))
    return intensity_img, pixel_is_occupied


def get_linear_bev_idx(
    pcl: torch.FloatTensor,
    coords_min: torch.FloatTensor,
    coords_max: torch.FloatTensor,
    img_grid_size: torch.IntTensor,
):
    assert coords_min.shape[0] == 2, coords_min.shape
    assert coords_max.shape[0] == 2, coords_max.shape
    assert img_grid_size.numel() == 2, img_grid_size
    epsilon_m = 0.01  # 1cm
    pcl_2d = pcl[:, :2]
    is_inside_bev_extent = torch.all(
        (coords_min + epsilon_m) < pcl_2d, axis=-1
    ) & torch.all(pcl_2d < (coords_max - epsilon_m), axis=-1)

    rowcol = project_2d_pcl_to_rowcol_nonsquare_bev_range(
        pcl_2d,
        coords_min,
        coords_max,
        img_grid_size,
    )

    linear_image_indices = rowcol.long()[:, 0] * img_grid_size[1] + rowcol.long()[:, 1]
    return is_inside_bev_extent, linear_image_indices


def project_2d_pcl_to_rowcol_nonsquare_bev_range(
    pcl_2d: torch.FloatTensor,
    coords_min: torch.FloatTensor,
    coords_max: torch.FloatTensor,
    img_grid_size: torch.IntTensor,
):
    assert coords_min.shape[0] == 2, coords_min.shape
    assert coords_max.shape[0] == 2, coords_max.shape
    assert img_grid_size.numel() == 2, img_grid_size
    pcl_2d = pcl_2d - coords_min

    map_size = coords_max - coords_min
    coord_to_im_factors = img_grid_size / map_size
    # coord_to_im_factors =
    min_factor = coord_to_im_factors.min()
    coord_to_im_factors = torch.stack((min_factor, min_factor))
    pcl_2d *= coord_to_im_factors
    return pcl_2d
