from math import isqrt
from pathlib import Path
from typing import List

import numpy as np
import torch
from liso.datasets.torch_dataset_commons import load_sanitize_box_augmentation_database
from liso.kabsch.shape_utils import Shape
from liso.visu.pcl_image import create_topdown_f32_pcl_image_variable_extent
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import make_grid


def visualize_augm_boxes_with_points_inside_them(
    path_to_augm_box_db: Path,
    num_boxes_to_visualize: int,
    writer: SummaryWriter,
    global_step: int,
    writer_prefix="",
):
    augm_db = load_sanitize_box_augmentation_database(
        path_to_augm_box_db, confidence_threshold_mined_boxes=0.0
    )
    box_idxs = torch.from_numpy(
        np.random.choice(
            np.arange(augm_db["boxes"].shape[0]),
            size=min(num_boxes_to_visualize, augm_db["boxes"].shape[0]),
            replace=False,
        )
    )

    # boxes = augm_db["boxes"][box_idxs]
    all_boxes = augm_db["boxes"]
    all_pcls_points_in_box = augm_db["pcl_in_box_cosy"]
    num_boxes_per_image = 8
    create_log_grids_of_single_box_images(
        all_boxes,
        all_pcls_points_in_box,
        writer,
        writer_prefix,
        num_boxes_per_image,
        global_step,
        box_idxs=box_idxs,
    )


def create_log_grids_of_single_box_images(
    all_boxes: Shape,
    all_pcls_points_in_box: List[np.ndarray],
    writer: SummaryWriter,
    writer_prefix: str,
    num_boxes_per_image: int,
    global_step: int,
    box_idxs=None,
):
    box_imgs = create_single_box_images_with_points(
        all_boxes, all_pcls_points_in_box, sequence_of_box_idxs=box_idxs
    )
    for i in range(0, len(box_imgs), num_boxes_per_image):
        img_chunk = box_imgs[i : i + num_boxes_per_image]
        log_grid_of_box_fits(
            writer,
            global_step + i,
            writer_prefix,
            num_boxes_per_image,
            img_chunk,
        )


def create_single_box_images_with_points(
    all_boxes: Shape,
    all_pcls_points_in_box: List[np.ndarray],
    sequence_of_box_idxs=None,
    img_size_per_box=(256, 128),
    box_bloat_factor=0.6,
) -> List[torch.FloatTensor]:
    if sequence_of_box_idxs is None:
        sequence_of_box_idxs = np.arange(len(all_boxes))
    img_size_per_box = torch.tensor(img_size_per_box)
    box_imgs = []
    for box_idx in sequence_of_box_idxs:
        points_in_box = torch.from_numpy(all_pcls_points_in_box[box_idx])
        box = all_boxes[box_idx]

        _, box_bev_img = create_topdown_f32_pcl_image_variable_extent(
            pcl=points_in_box,
            intensity=points_in_box[:, -1],
            coords_min=-box_bloat_factor * box.dims[:2],
            coords_max=box_bloat_factor * box.dims[:2],
            img_grid_size=img_size_per_box,
        )

        box_border_points = generate_point_lines_on_box_borders(img_size_per_box, box)
        _, box_border_bev_img = create_topdown_f32_pcl_image_variable_extent(
            pcl=box_border_points,
            intensity=box_border_points[:, -1],
            coords_min=-box_bloat_factor * box.dims[:2],
            coords_max=box_bloat_factor * box.dims[:2],
            img_grid_size=img_size_per_box,
        )

        box_img_border_red = (
            torch.tensor([1.0, 0.0, 0.0])[..., None, None]
            * box_border_bev_img[None, ...]
        )
        box_img_points_white = torch.ones(3)[..., None, None] * box_bev_img[None, ...]

        box_img = torch.maximum(box_img_border_red, box_img_points_white)
        box_imgs.append(box_img)
    return box_imgs


def generate_point_lines_on_box_borders(
    img_size_per_box: torch.IntTensor, box: Shape
) -> torch.IntTensor:
    num_pts_per_line = torch.max(img_size_per_box)
    coords_along_x = torch.linspace(-box.dims[0] / 2, box.dims[0] / 2, num_pts_per_line)
    coords_along_y = torch.linspace(-box.dims[1] / 2, box.dims[1] / 2, num_pts_per_line)
    left_line_xy = torch.cat(
        [
            coords_along_x[..., None],
            -box.dims[1] / 2 * torch.ones_like(coords_along_x)[..., None],
        ],
        dim=-1,
    )
    right_line_xy = torch.cat(
        [
            coords_along_x[..., None],
            box.dims[1] / 2 * torch.ones_like(coords_along_x)[..., None],
        ],
        dim=-1,
    )
    top_line_xy = torch.cat(
        [
            -box.dims[0] / 2 * torch.ones_like(coords_along_x)[..., None],
            coords_along_y[..., None],
        ],
        dim=-1,
    )
    bottom_line_xy = torch.cat(
        [
            box.dims[0] / 2 * torch.ones_like(coords_along_x)[..., None],
            coords_along_y[..., None],
        ],
        dim=-1,
    )
    center_line_front = torch.cat(
        [
            coords_along_x[num_pts_per_line // 2 :, None],
            torch.zeros_like(coords_along_x[num_pts_per_line // 2 :])[..., None],
        ],
        dim=-1,
    )

    box_lines = torch.cat(
        [left_line_xy, right_line_xy, top_line_xy, bottom_line_xy, center_line_front],
        dim=0,
    )
    box_lines = torch.cat([box_lines, torch.ones_like(box_lines)], dim=-1)
    return box_lines


def log_grid_of_box_fits(
    writer: SummaryWriter,
    global_step: int,
    writer_prefix: str,
    num_boxes_per_image: int,
    box_imgs: List[torch.FloatTensor],
):
    if num_boxes_per_image == isqrt(num_boxes_per_image) ** 2:
        # make square grid
        num_rows = isqrt(num_boxes_per_image)
    else:
        num_rows = max(num_boxes_per_image, len(box_imgs)) // 2

    grid_of_imgs = make_grid(
        box_imgs,
        nrow=num_rows,
        padding=4,
        pad_value=1.0,
    )
    writer.add_image(
        f"{writer_prefix.rstrip('/')}/box_fit_imgs",
        grid_of_imgs,
        global_step=global_step,
    )
