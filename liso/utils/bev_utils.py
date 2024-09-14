import numpy as np
import torch


def get_metric_voxel_center_coords(bev_range_x, bev_range_y, dataset_img_shape):
    bev_extent_m = 0.5 * np.array(
        [-bev_range_x, -bev_range_y, bev_range_x, bev_range_y]
    )
    voxel_center_metric_coordinates = get_voxel_center_coords_m(
        bev_extent_m=bev_extent_m, net_output_shape_pix=dataset_img_shape
    )
    homog_metric_voxel_center_coords = np.concatenate(
        [
            voxel_center_metric_coordinates,
            np.zeros_like(voxel_center_metric_coordinates[..., :1]),
            np.ones_like(voxel_center_metric_coordinates[..., :1]),
        ],
        axis=-1,
    )
    return homog_metric_voxel_center_coords


def get_voxel_center_coords_m(bev_extent_m, net_output_shape_pix):
    voxel_center_metric_coordinates = (
        np.stack(
            np.meshgrid(
                np.arange(net_output_shape_pix[0]),
                np.arange(net_output_shape_pix[1]),
                indexing="ij",
            ),
            axis=-1,
        )
        + 0.5
    )
    voxel_center_metric_coordinates /= net_output_shape_pix
    voxel_center_metric_coordinates *= bev_extent_m[2:] - bev_extent_m[:2]
    voxel_center_metric_coordinates += bev_extent_m[:2]
    return voxel_center_metric_coordinates


def get_bev_setup_params(cfg):
    bev_range_m_np = np.array(cfg.data.bev_range_m, np.float32)
    img_grid_size_np = np.array(cfg.data.img_grid_size).astype(np.int32)
    bev_pixel_per_meter_res_np = (img_grid_size_np / bev_range_m_np).astype(np.float32)
    pcl_bev_center_coords_homog_np = get_metric_voxel_center_coords(
        bev_range_x=bev_range_m_np[0],
        bev_range_y=bev_range_m_np[1],
        dataset_img_shape=img_grid_size_np,
    ).astype(np.float32)

    torch_params = {
        "bev_range_m": torch.from_numpy(bev_range_m_np),
        "bev_pixel_per_meter_resolution": torch.from_numpy(bev_pixel_per_meter_res_np),
        "img_grid_size": torch.from_numpy(img_grid_size_np),
        "pcl_bev_center_coords_homog": torch.from_numpy(pcl_bev_center_coords_homog_np),
    }

    return (
        bev_range_m_np,
        img_grid_size_np,
        bev_pixel_per_meter_res_np,
        pcl_bev_center_coords_homog_np,
        torch_params,
    )
