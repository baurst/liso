from typing import List

import matplotlib
import numpy as np
import torch
import torchvision
from liso.kabsch.kabsch_mask import torch_compose_matrix
from liso.kabsch.shape_utils import Shape
from liso.utils.nms_iou import perform_nms_on_shapes
from liso.visu.pcl_image import (
    project_2d_pcl_to_rowcol_nonsquare_bev_range,
    torch_batched_pillarize_pointcloud,
)
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont
from skimage.draw import line_aa


def range_project_points(
    pcl,
    elevation_rad_max,
    elevation_rad_min,
    range_image_height,
    range_image_width,
):
    d_xy = torch.linalg.norm(pcl[..., :2], dim=-1)
    elevation_angle_rad = torch.atan2(pcl[..., 2], d_xy)
    elevation_angle_rad_normed = 1.0 - (
        torch.clip(
            elevation_angle_rad,
            min=elevation_rad_min,
            max=elevation_rad_max,
        )
        - elevation_rad_min
    ) / (elevation_rad_max - elevation_rad_min)

    row_idx = (
        torch.clip(
            elevation_angle_rad_normed * range_image_height,
            min=0,
            max=range_image_height - 1,
        )
        .cpu()
        .numpy()
    ).astype(np.uint32)
    azimuth_angle_rad = torch.atan2(pcl[..., 1], pcl[..., 0])
    azimuth_angle_rad_normed = (azimuth_angle_rad + np.pi) / (2 * np.pi)

    col_idx = (
        torch.clip(
            azimuth_angle_rad_normed * range_image_width,
            min=0,
            max=range_image_width - 1,
        )
        .cpu()
        .numpy()
    ).astype(np.uint32)

    return row_idx, col_idx


def render_pcl_range_image(
    pcls: List[torch.Tensor],
    elevation_rad_max: float,
    elevation_rad_min: float,
    range_image_height: int,
    range_image_width: int,
):
    assert elevation_rad_max > elevation_rad_min, (elevation_rad_max, elevation_rad_min)
    batch_size = len(pcls)
    img_canvas = np.zeros(
        (batch_size, range_image_height, range_image_width, 3),
        dtype=np.float32,
    )
    for batch_idx in range(batch_size):
        pcl = pcls[batch_idx]
        if pcl.shape[-1] != 4:
            range_m = torch.linalg.norm(pcl[:, :3], dim=-1)
            range_normed = (range_m - range_m.min()) / (range_m.max() - range_m.min())
            point_color = cm.gist_rainbow(range_normed.cpu().numpy())[..., :3]
        else:
            point_color = cm.gist_rainbow(pcl[:, 3].cpu().numpy())[..., :3]

        row_idx, col_idx = range_project_points(
            pcl,
            elevation_rad_max,
            elevation_rad_min,
            range_image_height,
            range_image_width,
        )
        img_canvas[batch_idx, row_idx, col_idx] = point_color
    return img_canvas


@torch.no_grad()
def draw_boxes_on_2d_projection(
    img_canvas: np.ndarray,
    gt_boxes: Shape,
    elevation_rad_max: float,
    elevation_rad_min: float,
    box_color: np.ndarray,
):
    assert img_canvas.shape[-1] == box_color.shape[-1]
    batch_size, range_image_height, range_image_width, _ = img_canvas.shape
    gt_boxes_no_nan_padding = gt_boxes.clone()
    gt_boxes_no_nan_padding.set_padding_val_to(0.0)
    gt_boxes_no_nan_padding.dims = torch.clip(
        gt_boxes_no_nan_padding.dims, min=0.3, max=15.0
    )
    box_corners, line_plot_sequence = gt_boxes_no_nan_padding.get_box_corners()
    box_row_idx, box_col_idx = range_project_points(
        box_corners,
        elevation_rad_max,
        elevation_rad_min,
        range_image_height,
        range_image_width,
    )
    box_points_in_image = np.stack([box_row_idx, box_col_idx], axis=-1)
    for batch_idx in range(batch_size):
        for box_idx in range(box_corners.shape[1]):
            box_valid_mask = gt_boxes[batch_idx].valid
            if box_valid_mask[box_idx]:
                if box_color.shape == gt_boxes.pos.shape:
                    line_color = box_color[batch_idx, box_idx]
                else:
                    line_color = box_color
                for (
                    line_start_idx,
                    line_end_idx,
                ) in line_plot_sequence:
                    line_start_point = box_points_in_image[
                        batch_idx, box_idx, line_start_idx
                    ]
                    line_end_point = box_points_in_image[
                        batch_idx, box_idx, line_end_idx
                    ]
                    line_column_diff = abs(
                        line_start_point[1].astype(np.int64)
                        - line_end_point[1].astype(np.int64)
                    )

                    if line_column_diff > range_image_width / 2:
                        # probably wraparound!
                        continue

                    draw_line_on_image(
                        img_canvas[batch_idx],
                        line_start_point,
                        line_end_point,
                        line_color,
                    )


def draw_connected_lines_on_image(batched_line_segments, img, line_colors):
    for object_corners, line_color in zip(batched_line_segments, line_colors):
        for idx, _ in enumerate(object_corners):
            line_start = object_corners[idx]
            line_end = object_corners[(idx + 1) % len(object_corners)]
            draw_line_on_image(img, line_start, line_end, line_color)
    return img


def draw_line_on_image(
    img: np.ndarray,
    line_start,
    line_end,
    line_color: np.ndarray,
):
    rows, cols, weights = line_aa(
        line_start[0], line_start[1], line_end[0], line_end[1]
    )  # antialias line

    w = weights.reshape([-1, 1])  # reshape anti-alias weights
    rows = np.clip(rows, 0, img.shape[0] - 1)
    cols = np.clip(cols, 0, img.shape[1] - 1)
    img[rows, cols, ...] = (
        np.multiply(
            (1 - w) * np.ones([1, line_color.shape[-1]]),
            img[rows, cols, ...],
        )
        + w * line_color
    )


@torch.no_grad()
def draw_box_onto_image(
    draw_shape: Shape,
    img_array: np.ndarray,
    bev_range_m: tuple,
    color: np.ndarray,
):
    assert img_array.shape[-1] in (3, 4), img_array.shape
    assert len(img_array.shape) == 4, img_array.shape
    assert len(draw_shape.pos.shape) == 3, draw_shape.pos.shape
    box_corners_sensor = get_box_corners_in_sensor_coordinates(draw_shape)
    grid_size = torch.tensor(img_array.shape[1:3], device=draw_shape.pos.device)
    if len(bev_range_m) == 2:
        bev_range_m = torch.tensor(bev_range_m, device=draw_shape.pos.device)
        box_corner_row_col_indices = (
            (box_corners_sensor[..., 0:2] + 0.5 * bev_range_m) * grid_size / bev_range_m
        )
        box_corner_row_col_indices = torch.round(box_corner_row_col_indices).long()
        box_corner_row_col_indices = (
            # torch.clamp(box_corner_row_col_indices, min=0, max=grid_size[0] - 1)
            box_corner_row_col_indices.detach()
            .cpu()
            .numpy()
        )
    elif torch.is_tensor(bev_range_m) and bev_range_m.numel() == 4:
        batch_size, num_boxes, num_corners, _ = box_corners_sensor.shape
        box_corners_2d = box_corners_sensor[..., :2].reshape((-1, 2))
        box_corner_row_col_indices = project_2d_pcl_to_rowcol_nonsquare_bev_range(
            pcl_2d=box_corners_2d,
            coords_min=bev_range_m[:2],
            coords_max=bev_range_m[2:],
            img_grid_size=grid_size,
        )
        box_corner_row_col_indices = (
            box_corner_row_col_indices.reshape((batch_size, num_boxes, num_corners, 2))
            .detach()
            .cpu()
            .numpy()
        ).astype(np.int64)
    else:
        raise NotImplementedError()
    # box_corner_row_col_indices = np.clip(box_corner_row_col_indices, -1e6, 1e6).astype(
    #     np.int64
    # )
    draw_shape = draw_shape.numpy()
    for batch_idx, img in enumerate(img_array):
        valid_objs = draw_shape.valid[batch_idx, :]
        if np.count_nonzero(valid_objs) > 0:
            valid_confidence = np.squeeze(draw_shape.probs[batch_idx, :], axis=-1)[
                valid_objs
            ]
            if isinstance(color, np.ndarray):
                assert color.dtype == img_array.dtype, (color.dtype, img_array.dtype)
                if len(color.shape) == len(draw_shape.valid.shape) + 1:
                    per_line_colors = color[batch_idx][valid_objs]
                elif len(color.shape) == 1:
                    per_line_colors = (
                        np.ones_like(valid_confidence)[..., None] * color[None, ...]
                    )
            elif color == "confidence":
                if valid_confidence.min() != valid_confidence.max():
                    confidence_normed = (valid_confidence - valid_confidence.min()) / (
                        valid_confidence.max() - valid_confidence.min()
                    )
                else:
                    confidence_normed = np.ones_like(valid_confidence)

                per_line_colors = cm.summer(confidence_normed)[..., :3]
            img_array[batch_idx] = draw_connected_lines_on_image(
                box_corner_row_col_indices[batch_idx][valid_objs],
                img,
                per_line_colors,
            )
    return img_array


def get_box_corners_in_sensor_coordinates(draw_shape: Shape):
    if draw_shape.dims.shape[-1] in (2, 3):
        zeros = torch.zeros_like(draw_shape.dims[..., 0])
        ones = torch.ones_like(zeros)
        s_T_box = torch_compose_matrix(
            t_x=draw_shape.pos[..., 0],
            t_y=draw_shape.pos[..., 1],
            theta_z=draw_shape.rot[..., 0],
        )
        pos = 0.5
        neg = -0.5
        box_corners = torch.stack(
            [
                torch.stack(
                    [
                        pos * draw_shape.dims[..., 0],
                        pos * draw_shape.dims[..., 1],
                        zeros,
                        ones,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        neg * draw_shape.dims[..., 0],
                        pos * draw_shape.dims[..., 1],
                        zeros,
                        ones,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        neg * draw_shape.dims[..., 0],
                        neg * draw_shape.dims[..., 1],
                        zeros,
                        ones,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        pos * draw_shape.dims[..., 0],
                        neg * draw_shape.dims[..., 1],
                        zeros,
                        ones,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        # top hat of box
                        pos * draw_shape.dims[..., 0] + 0.5 * draw_shape.dims[..., 1],
                        zeros,
                        zeros,
                        ones,
                    ],
                    dim=-1,
                ),
            ],
            dim=2,
        )
    elif draw_shape.dims.shape[-1] == 1:
        # assume circle
        theta = (
            2
            * np.pi
            * torch.linspace(start=0, steps=20, end=1, device=draw_shape.dims.device)
        )
        x = (
            0.5
            * draw_shape.dims[:, :, None, [0]]
            * torch.cos(theta)[None, None, :, None]
        )
        y = (
            0.5
            * draw_shape.dims[:, :, None, [0]]
            * torch.sin(theta)[None, None, :, None]
        )
        box_corners = torch.cat([x, y, torch.zeros_like(x), torch.ones_like(x)], dim=-1)
        s_T_box = torch_compose_matrix(
            t_x=draw_shape.pos[..., 0],
            t_y=draw_shape.pos[..., 1],
            theta_z=torch.zeros_like(draw_shape.pos[..., 0]),
        )
    else:
        raise NotImplementedError(draw_shape.dims.shape)
    box_corners_sensor = torch.einsum(
        "bsij, bsnj -> bsni", s_T_box, box_corners.to(s_T_box.dtype)
    ).to(torch.float32)
    return box_corners_sensor


def batched_np_img_to_torch_img_grid(img):
    return torchvision.utils.make_grid(
        torch.from_numpy(img).permute(0, -1, 1, 2), pad_value=0.5
    )


def create_range_image_w_boxes(
    pcls: torch.FloatTensor,
    boxes: Shape = None,
    fitted_boxes: Shape = None,
    range_image_height: int = 128,
    range_image_width: int = 1024,
    elevation_deg_max: float = 10.0,
    elevation_deg_min: float = -30.0,
    vertical_resulution_upsampling_factor: int = 2,
):
    elevation_rad_max = np.deg2rad(elevation_deg_max)
    elevation_rad_min = np.deg2rad(elevation_deg_min)
    img_canvas = render_pcl_range_image(
        pcls,
        elevation_rad_max,
        elevation_rad_min,
        range_image_height,
        range_image_width,
    )
    if boxes is not None:
        draw_boxes_on_2d_projection(
            img_canvas,
            boxes,
            elevation_rad_max,
            elevation_rad_min,
            box_color=np.array([1.0, 0.0, 0.0]),
        )
    if fitted_boxes is not None:
        draw_boxes_on_2d_projection(
            img_canvas,
            fitted_boxes,
            elevation_rad_max,
            elevation_rad_min,
            box_color=np.array([0.0, 1.0, 1.0]),
        )

    img_canvas = np.repeat(
        img_canvas,
        repeats=vertical_resulution_upsampling_factor,
        axis=1,  # increase vertical resolution
    )

    return img_canvas


def scalar_colored_box_img_w_text(
    cfg,
    bev_range_m_torch: torch.FloatTensor,
    canvas_np_gray_channel_last: np.ndarray,
    gt_boxes: Shape,
    color_these_boxes: Shape,
    per_obj_scalar: torch.FloatTensor,
):
    rotation_error_box_img = attribute_colored_box_image(
        cfg=cfg,
        canvas_np_gray_channel_last=canvas_np_gray_channel_last,
        gt_boxes=gt_boxes,
        pred_boxes=color_these_boxes,
        per_pred_box_scalar=per_obj_scalar,
    )
    canvii = plot_per_box_text_on_canvas(
        bev_range_m=bev_range_m_torch,
        max_num_images=cfg.logging.max_log_img_batches,
        pred_boxes=color_these_boxes,
        per_obj_scalar=per_obj_scalar,
        target_canvas_np=rotation_error_box_img,
    )

    return canvii


def attribute_colored_box_image(
    cfg, canvas_np_gray_channel_last, gt_boxes, pred_boxes, per_pred_box_scalar
):
    rgb_canvas = canvas_np_gray_channel_last * np.ones((1, 3))
    if gt_boxes is not None:
        rgb_canvas = draw_box_onto_image(
            gt_boxes,
            canvas_np_gray_channel_last * np.ones((1, 3)),
            cfg.data.bev_range_m,
            color=np.array([1.0, 0.0, 0.0]),
        )
    per_pred_box_scalar_npy = per_pred_box_scalar.detach().cpu().numpy()
    valid_boxes_mask = pred_boxes.valid.detach().cpu().numpy()
    if np.count_nonzero(valid_boxes_mask) > 0:
        max_scalar = per_pred_box_scalar_npy[valid_boxes_mask].max()
        min_scalar = per_pred_box_scalar_npy[valid_boxes_mask].min()
        per_pred_box_scalar_npy -= min_scalar
        per_pred_box_scalar_npy /= np.maximum(
            max_scalar - min_scalar,
            1e-6,
        )
        error_color = matplotlib.cm.summer(per_pred_box_scalar_npy)[..., 0:3]
    else:
        error_color = matplotlib.cm.summer(0.5 * np.ones_like(per_pred_box_scalar_npy))[
            ..., 0:3
        ]

    rgb_canvas = draw_box_onto_image(
        pred_boxes,
        rgb_canvas,
        cfg.data.bev_range_m,
        color=error_color,
    )

    return rgb_canvas


def plot_per_box_text_on_canvas(
    *,
    bev_range_m: torch.FloatTensor,
    pred_boxes: Shape,
    per_obj_scalar: torch.FloatTensor,
    target_canvas_np: np.ndarray,
    max_num_images: int,
):
    assert len(target_canvas_np.shape) == 4, target_canvas_np.shape
    assert target_canvas_np.shape[-1] == 3, target_canvas_np.shape
    canvii = []
    _, obj_pillar_coors = torch_batched_pillarize_pointcloud(
        pcl_torch=pred_boxes.pos,
        bev_range_m=bev_range_m,
        pillar_bev_resolution=torch.tensor(
            target_canvas_np.shape[1:3], device=pred_boxes.pos.device
        ),
    )
    per_obj_loss = per_obj_scalar.clone().detach().cpu().numpy()
    text_pos = obj_pillar_coors.detach().cpu().numpy()
    for batch_idx in range(min(max_num_images, per_obj_scalar.shape[0])):
        canvas = Image.fromarray((255.0 * target_canvas_np[batch_idx]).astype(np.uint8))
        # draw the text onto the canvas
        draw = ImageDraw.Draw(canvas)
        for box_idx in range(pred_boxes.pos.shape[1]):
            if pred_boxes.valid[batch_idx, box_idx]:
                text_row = text_pos[batch_idx, box_idx, 0]
                text_col = text_pos[batch_idx, box_idx, 1]

                offset = (text_col, text_row)
                draw.text(
                    offset,
                    np.format_float_scientific(
                        per_obj_loss[batch_idx, box_idx], precision=1
                    ),
                    font=ImageFont.load_default(),
                    fill=(255, 1, 154),
                )
        canvii.append(canvas)

    canvii = np.stack(canvii)
    return canvii


def plot_text_on_canvas_at_position(
    text_pos: np.ndarray,
    target_canvas_channels_last_uint8: np.ndarray,
    texts: List[str],
):
    assert (
        target_canvas_channels_last_uint8.dtype == np.uint8
    ), target_canvas_channels_last_uint8.dtype
    assert target_canvas_channels_last_uint8.shape[-1] == 3
    assert text_pos.shape[0] == len(texts), (text_pos.shape, len(texts))

    canvas = Image.fromarray(target_canvas_channels_last_uint8)
    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    for idx, text in enumerate(texts):
        text_row = text_pos[idx, 0]
        text_col = text_pos[idx, 1]

        offset = (text_col, text_row)
        draw.text(
            offset,
            text,
            font=ImageFont.load_default(),
            fill=(255, 1, 154),
        )
    return np.array(canvas)


@torch.no_grad()
def draw_box_image(
    *,
    cfg,
    pred_boxes,
    gt_boxes,
    canvas_f32,
    gt_background_boxes,
    gt_boxes_prev=None,
    perform_nms=False,
    max_num_boxes=40,
):
    bs_input = canvas_f32.shape[0]
    visu_slice = torch.arange(min(bs_input, cfg.logging.max_log_img_batches))
    visu_boxes = []
    if gt_boxes is not None:
        reduced_gt_boxes = gt_boxes.clone()[visu_slice]
        visu_boxes.append(
            (
                reduced_gt_boxes,
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
            )
        )
    if pred_boxes is not None and torch.count_nonzero(pred_boxes.valid) > 0:
        reduced_pred_boxes = pred_boxes.clone()[visu_slice]
        reduced_pred_boxes.dims = torch.clamp(
            reduced_pred_boxes.dims, min=0.3, max=15.0
        )
        if perform_nms:
            nms_pred_boxes = perform_nms_on_shapes(
                reduced_pred_boxes,
                max_num_boxes=max_num_boxes,
                pre_nms_max_num_boxes=1000,
                overlap_threshold=cfg.nms_iou_threshold,
            )
        else:
            nms_pred_boxes = reduced_pred_boxes
        visu_boxes.append((nms_pred_boxes, "confidence"))
    reduced_occupancy_f32 = canvas_f32[visu_slice]
    if gt_background_boxes is not None:
        reduced_gt_bg_boxes = gt_background_boxes.clone()[visu_slice]
        visu_boxes.append(
            (reduced_gt_bg_boxes, np.array([0.3, 0.3, 0.3], dtype=np.float32))
        )
    if gt_boxes_prev is not None:
        reduced_gt_t1_boxes = gt_boxes_prev.clone()[visu_slice]
        visu_boxes.append(
            (reduced_gt_t1_boxes, np.array([0.13, 0.82, 0.96], dtype=np.float32))
        )

    reconstruction_target_box_img = (
        (
            reduced_occupancy_f32.permute((0, 2, 3, 1))
            * torch.ones((1, 3), device=reduced_occupancy_f32.device)
        )
        .cpu()
        .numpy()
    )
    for boxes, color in visu_boxes:
        if boxes.valid.sum() > 0:
            reconstruction_target_box_img = draw_box_onto_image(
                boxes,
                reconstruction_target_box_img,
                cfg.data.bev_range_m,
                color,
            )

    return reconstruction_target_box_img


@torch.no_grad()
def log_box_movement(
    *,
    cfg,
    writer,
    global_step,
    sample_data_a,
    sample_data_b,
    pred_boxes,
    gt_background_boxes=None,
    writer_prefix="",
):
    image_canvas_t0 = sample_data_a["occupancy_f32_ta"]

    reconstruction_target_box_img_t0_before_nms_1k_boxes_max = draw_box_image(
        cfg=cfg,
        pred_boxes=pred_boxes,
        gt_boxes=sample_data_a["gt"].get("boxes", None),
        canvas_f32=image_canvas_t0,
        gt_background_boxes=gt_background_boxes,
        perform_nms=True,
        max_num_boxes=100,
    )
    reconstruction_target_box_img_t0 = draw_box_image(
        cfg=cfg,
        pred_boxes=pred_boxes,
        gt_boxes=sample_data_a["gt"].get("boxes"),
        canvas_f32=image_canvas_t0,
        gt_background_boxes=gt_background_boxes,
        perform_nms=True,
        max_num_boxes=40,
    )
    image_canvas_t1 = sample_data_b.get("occupancy_f32_ta", None)
    images = [
        reconstruction_target_box_img_t0_before_nms_1k_boxes_max,
        reconstruction_target_box_img_t0,
    ]
    if image_canvas_t1 is not None:
        reconstruction_target_box_img_t1 = draw_box_image(
            cfg=cfg,
            pred_boxes=None,
            gt_boxes=sample_data_b["gt"].get("boxes"),
            canvas_f32=image_canvas_t1,
            gt_background_boxes=gt_background_boxes,
            gt_boxes_prev=sample_data_a["gt"].get("boxes"),
        )
        images.append(reconstruction_target_box_img_t1)
    writer.add_images(
        writer_prefix + "RECONSTRUCTION_TARGETS_t0_t1",
        np.concatenate(
            images,
            axis=1,
        ),
        global_step=global_step,
        dataformats="NHWC",
    )


def render_gt_boxes_with_predicted_logits(
    cfg, mask_gt_renderer, sample_data_t0, gt_boxes, pred_boxes_maps
):
    pred_logits_map = pred_boxes_maps["probs"]

    obj_batch_coors, obj_pillar_coors = torch_batched_pillarize_pointcloud(
        pcl_torch=gt_boxes.pos.cpu(),
        bev_range_m=torch.tensor(cfg.data.bev_range_m),
        pillar_bev_resolution=torch.tensor(pred_logits_map.shape[1:3]),
    )

    pred_logits_at_gt_locations = pred_logits_map[
        obj_batch_coors[..., 0],
        obj_pillar_coors[..., 0],
        obj_pillar_coors[..., 1],
        ...,
    ]

    gt_boxes_with_pred_confidence = gt_boxes.clone()
    assert (
        gt_boxes_with_pred_confidence.probs.shape == pred_logits_at_gt_locations.shape
    ), (
        gt_boxes_with_pred_confidence.probs.shape,
        pred_logits_at_gt_locations.shape,
    )
    gt_boxes_with_pred_confidence.probs = pred_logits_at_gt_locations

    gt_img_with_pred_logis = scalar_colored_box_img_w_text(
        cfg=cfg,
        bev_range_m_torch=torch.tensor(
            cfg.data.bev_range_m, device=gt_boxes_with_pred_confidence.probs.device
        ),
        canvas_np_gray_channel_last=sample_data_t0["occupancy_f32_ta"]
        .clone()
        .detach()
        .permute((0, 2, 3, 1))
        .cpu()
        .numpy(),
        gt_boxes=None,
        color_these_boxes=gt_boxes_with_pred_confidence,
        per_obj_scalar=torch.squeeze(gt_boxes_with_pred_confidence.probs, axis=-1),
    )

    return gt_img_with_pred_logis
