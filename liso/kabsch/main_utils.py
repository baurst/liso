from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import numpy as np
import torch
import torchvision
from config_helper.config import get_config_hash
from liso.datasets.argoverse2.av2_torch_dataset import (
    AV2Dataset,
    get_av2_train_dataset,
    get_av2_val_dataset,
)
from liso.datasets.kitti_raw_torch_dataset import get_kitti_train_dataset
from liso.datasets.kitti_tracking_torch_dataset import (
    KittiTrackingDataset,
    get_kitti_val_dataset,
)
from liso.datasets.nuscenes_torch_dataset import (
    NuscenesDataset,
    get_nuscenes_train_dataset,
    get_nuscenes_val_dataset,
)
from liso.datasets.torch_dataset_commons import lidar_dataset_collate_fn, worker_init_fn
from liso.datasets.waymo_torch_dataset import (
    WaymoDataset,
    get_waymo_train_dataset,
    get_waymo_val_dataset,
)
from liso.eval.flow_metrics import FlowMetrics
from liso.kabsch.shape_utils import Shape
from liso.losses.hungarian_matching_loss import hungarian_matching_loss
from liso.visu.bbox_image import batched_np_img_to_torch_img_grid, draw_box_image
from liso.visu.pcl_image import torch_batched_pillarize_pointcloud
from liso.visu.utils import apply_cmap, limit_visu_image_batches
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

matplotlib.use("agg")


def normalize_img_tensor(img_tensor):
    batch_size, height, width, c = img_tensor.shape
    img_tensor = img_tensor.view(img_tensor.size(0), -1)
    img_tensor -= img_tensor.min(1, keepdim=True)[0]
    img_tensor /= img_tensor.max(1, keepdim=True)[0]
    img_tensor = img_tensor.view(batch_size, height, width, c)
    return img_tensor


def rotation_vec_on_unit_circle(raw_activated_box_pred):
    assert raw_activated_box_pred["rot"].shape[-1] == 2
    vector_len = torch.norm(raw_activated_box_pred["rot"], dim=-1)
    regularization_loss = torch.nn.functional.mse_loss(
        input=vector_len,
        target=torch.ones_like(vector_len),
    )
    return regularization_loss


def limit_rotation_to_plusminus_half_pi(rot_angle):
    assert rot_angle.shape[-1] == 1
    out_of_bounds = torch.abs(rot_angle) >= 0.5 * np.pi
    regu_loss = torch.where(
        out_of_bounds,
        torch.nn.functional.mse_loss(
            input=rot_angle,
            target=torch.zeros_like(rot_angle),
            reduction="none",
        ),
        torch.zeros_like(rot_angle),
    )

    return regu_loss.mean()


def get_box_dbs_path(cfg):
    key_remote_or_local = "local"
    box_db_path = Path(cfg.data.paths[cfg.data.source]["box_dbs"][key_remote_or_local])
    cfg_hash = get_config_hash(cfg.data.tracking_cfg)[:5]
    bev_range_str = "bev_range_m" + "_".join(
        str(int(range_dist)) for range_dist in cfg.data.bev_range_m
    )

    db_path_addon = f"{cfg.data.source}_flow_{cfg.data.flow_source}_odom_{cfg.data.odom_source}_{bev_range_str}_{cfg_hash}"
    return box_db_path / db_path_addon


def sanity_check_cfg(cfg):
    # CHECK CONFIG
    for attr_key, modif_desc in cfg.box_prediction.output_modification.items():
        assert attr_key in ("pos", "dims", "rot", "probs"), attr_key
        assert modif_desc in ("pred", "gt", "gt_fixed"), modif_desc
    assert cfg.optimization.learning_rate <= 0.01
    if cfg.box_prediction.activations.pos != "none":
        assert cfg.box_prediction.position_representation.method in (
            "global_relative",
            "local_relative_offset",
        ), f"with activation {cfg.box_prediction.activations.pos} we need to predict the relative position"
    assert cfg.box_prediction.dimensions_representation.method in (
        "predict_aspect_ratio",
        "predict_abs_size",
        "predict_log_size",
    ), cfg.box_prediction.dimensions_representation.method
    assert (
        cfg.loss.supervised.hungarian.active
        or cfg.loss.supervised.centermaps.active
        or cfg.loss.supervised.supervised_on_clusters.active
        or cfg.data.augmentation.boxes.active
        or cfg.loss.pointrcnn_loss.active
        or cfg.loss.pointpillars_loss.active
    ), cfg.loss.supervised

    assert (
        cfg.optimization.abs_num_warmup_steps <= cfg.optimization.num_training_steps
    ), (cfg.optimization.abs_num_warmup_steps, cfg.optimization.num_training_steps)


def apply_rotation_regularization_loss(
    cfg, raw_activated_box_attrs_a, supervised_loss_boxes, losses_dict
):
    if cfg.box_prediction.rotation_representation.method == "vector":
        assert (
            cfg.box_prediction.rotation_representation.regularization
            == "rot_vec_on_unit_circle"
        ), cfg.box_prediction.rotation_representation.regularization
        rot_repr_regularization_loss = (
            rotation_vec_on_unit_circle(
                raw_activated_box_pred=raw_activated_box_attrs_a
            )
            * cfg.box_prediction.rotation_representation.regul_weight
        )
        losses_dict["loss/rotation_regularization"] = rot_repr_regularization_loss
    elif (
        cfg.box_prediction.rotation_representation.method == "direct"
        and cfg.box_prediction.rotation_representation.regularize_limit_angle_to_pi
    ):
        assert (
            cfg.box_prediction.rotation_representation.regul_weight != 0.0
        ), cfg.box_prediction.rotation_representation.regul_weight
        rot_repr_regularization_loss = (
            limit_rotation_to_plusminus_half_pi(supervised_loss_boxes.rot)
            * cfg.box_prediction.rotation_representation.regul_weight
        )
        losses_dict[
            "loss/limit_rotation_to_plusminus_half_pi_loss"
        ] = rot_repr_regularization_loss


def sv_hungarian_loss(
    *,
    cfg,
    writer,
    gt_boxes_a: Shape,
    supervised_loss_boxes: Shape,
    raw_activated_box_attrs_a: Dict[str, torch.FloatTensor],
    losses_dict,
    global_step: int,
    hungarian_loss_weight=None,
    train_these_box_attrs: Tuple[str] = None,
    loss_extra_descr="supervised",
    ignore_region_is_true_mask: torch.BoolTensor = None,
):
    if train_these_box_attrs is None:
        train_these_box_attrs = {"probs", "rot", "dims", "pos"}

    if hungarian_loss_weight is None:
        hungarian_loss_weight = cfg.loss.supervised.hungarian.weight

    if cfg.box_prediction.rotation_representation.method == "vector":
        # TODO: this is a dirty solution, but it should break if misused
        gt_boxes_for_hungarian = gt_boxes_a.clone()
        gt_boxes_for_hungarian.rot = torch.cat(
            [
                torch.sin(gt_boxes_for_hungarian.rot),
                torch.cos(gt_boxes_for_hungarian.rot),
            ],
            dim=-1,
        )
        pred_boxes_for_hungarian = supervised_loss_boxes.clone()
        assert (
            pred_boxes_for_hungarian.shape
            == raw_activated_box_attrs_a["rot"].shape[:-1]
        ), (pred_boxes_for_hungarian.shape, raw_activated_box_attrs_a["rot"].shape)
        pred_boxes_for_hungarian.rot = raw_activated_box_attrs_a["rot"].clone()
    else:
        gt_boxes_for_hungarian = gt_boxes_a
        pred_boxes_for_hungarian = supervised_loss_boxes

    assert (
        cfg.box_prediction.activations.probs == "none"
    ), cfg.box_prediction.activations.probs

    if ignore_region_is_true_mask is not None and ignore_region_is_true_mask.any():
        pred_boxes_for_hungarian = pred_boxes_for_hungarian.clone()
        pred_obj_must_be_ignored = get_ignored_objects_from_ignore_mask(
            cfg, pred_boxes_for_hungarian, ignore_region_is_true_mask
        )
        pred_boxes_for_hungarian.valid = (
            pred_boxes_for_hungarian.valid & ~pred_obj_must_be_ignored
        )

        gt_boxes_for_hungarian = gt_boxes_for_hungarian.clone()
        gt_obj_must_be_ignored = get_ignored_objects_from_ignore_mask(
            cfg, gt_boxes_for_hungarian, ignore_region_is_true_mask
        )
        gt_boxes_for_hungarian.valid = (
            gt_boxes_for_hungarian.valid & ~gt_obj_must_be_ignored
        )

    hungarian_match_loss_dict = hungarian_matching_loss(
        predicted_bboxes=pred_boxes_for_hungarian,
        groundtruth_bboxes=gt_boxes_for_hungarian,
        writer=writer,
        global_step=global_step,
        train_box_attrs=train_these_box_attrs,
        prob_loss=cfg.loss.supervised.hungarian.prob_loss,
        loss_extra_descr=loss_extra_descr,
    )
    for loss_tag, loss_val in hungarian_match_loss_dict.items():
        losses_dict[loss_tag] = loss_val * hungarian_loss_weight
    return losses_dict


def get_ignored_objects_from_ignore_mask(
    cfg, pred_boxes_for_hungarian: Shape, ignore_region_is_true_mask: torch.BoolTensor
) -> torch.BoolTensor:
    (
        obj_batch_coors,
        obj_pillar_coors,
    ) = torch_batched_pillarize_pointcloud(
        pcl_torch=pred_boxes_for_hungarian.pos,
        bev_range_m=torch.tensor(
            cfg.data.bev_range_m, device=pred_boxes_for_hungarian.pos.device
        ),
        pillar_bev_resolution=torch.tensor(
            ignore_region_is_true_mask.shape[1:3],
            device=pred_boxes_for_hungarian.pos.device,
        ),
    )
    obj_must_be_ignored = ignore_region_is_true_mask[
        obj_batch_coors[..., 0], obj_pillar_coors[..., 0], obj_pillar_coors[..., 1]
    ]
    return obj_must_be_ignored


def get_network_input_pcls(
    cfg: Dict[str, Any],
    sample_data_t0: Dict[str, torch.FloatTensor],
    time_key: str,
    to_device=None,
) -> List[torch.FloatTensor]:
    net_inputs_pcl_key = (
        f"pcl_full_w_ground_{time_key}"
        if cfg.data.use_ground_for_network
        else f"pcl_full_no_ground_{time_key}"
    )
    if to_device:
        return [el.to(to_device) for el in sample_data_t0[net_inputs_pcl_key]]
    else:
        return sample_data_t0[net_inputs_pcl_key]


def log_gt_bev_maps(
    writer: SummaryWriter,
    writer_prefix: str,
    global_step: int,
    gt_regression_maps: Dict[str, torch.FloatTensor],
    gt_center_mask: torch.BoolTensor,
    ignore_region_is_true_mask: torch.BoolTensor = None,
):
    prefix = writer_prefix.rstrip("/")
    writer.add_images(
        f"{prefix}/CENTER_MASK",
        normalize_img_tensor(gt_center_mask.clone().float()[..., None]),
        global_step=global_step,
        dataformats="NHWC",
    )
    if ignore_region_is_true_mask is not None:
        writer.add_images(
            f"{prefix}/IGNORE_WHERE_TRUE_MASK",
            normalize_img_tensor(ignore_region_is_true_mask.clone().float()[..., None]),
            global_step=global_step,
            dataformats="NHWC",
        )
        writer.add_images(
            f"{prefix}/IGNORE_OR_CENTER_MASK",
            normalize_img_tensor(
                (ignore_region_is_true_mask | gt_center_mask).clone().float()[..., None]
            ),
            global_step=global_step,
            dataformats="NHWC",
        )
    for map_name in gt_regression_maps:
        writer.add_images(
            f"{prefix}/{map_name}",
            gt_regression_maps[map_name].mean(keepdims=True, dim=-1),
            global_step=global_step,
            dataformats="NHWC",
        )


def log_bce_loss_img(cfg, desc, writer, global_step, loss_img):
    writer.add_image(
        "loss_introspection/" + desc + "_hotness_normalized",
        torchvision.utils.make_grid(
            torch.from_numpy(
                apply_cmap(
                    limit_visu_image_batches(
                        loss_img,
                        max_batches=cfg.logging.max_log_img_batches,
                    ).permute(0, 2, 3, 1),
                    input_has_channel_dim=True,
                ),
            ).permute(0, 3, 1, 2),
            padding=2,
            pad_value=128,
        ),
        global_step=global_step,
    )


def get_datasets(
    cfg,
    fast_test,
    target="object",
    path_to_augmentation_db: str = None,
    path_to_mined_boxes_db: str = None,
    sv_finetuning_cfg: Dict = None,
    shuffle_validation=False,
    need_flow_during_training: bool = True,
):
    prefetch_args = {}
    num_workers = cfg.data.num_workers
    if sv_finetuning_cfg is not None:
        num_train_samples = sv_finetuning_cfg.num_samples
    else:
        num_train_samples = None
    if cfg.data.source == "nuscenes":
        train_loader, train_dataset = get_nuscenes_train_dataset(
            cfg,
            use_geom_augmentation=cfg.data.augmentation.active,
            use_skip_frames=cfg.data.use_skip_frames,
            path_to_augmentation_db=path_to_augmentation_db,
            path_to_mined_boxes_db=path_to_mined_boxes_db,
            size=num_train_samples,
            need_flow_during_training=need_flow_during_training,
        )
        val_on_train_dataset = NuscenesDataset(
            shuffle=False,
            use_geom_augmentation=False,
            use_skip_frames="never",
            mode="train",
            cfg=cfg,
        )
        val_on_train_loader = torch.utils.data.DataLoader(
            val_on_train_dataset,
            pin_memory=True,
            batch_size=cfg.data.batch_size,
            num_workers=num_workers,
            collate_fn=lidar_dataset_collate_fn,
            shuffle=True,
            worker_init_fn=worker_init_fn,
            **prefetch_args,
        )
        val_loader = get_nuscenes_val_dataset(
            cfg, use_skip_frames="never", size=None, shuffle=shuffle_validation
        )
    elif cfg.data.source == "av2":
        train_loader, train_dataset = get_av2_train_dataset(
            cfg,
            use_geom_augmentation=cfg.data.augmentation.active,
            use_skip_frames=cfg.data.use_skip_frames,
            path_to_augmentation_db=path_to_augmentation_db,
            path_to_mined_boxes_db=path_to_mined_boxes_db,
            size=num_train_samples,
            need_flow_during_training=need_flow_during_training,
        )
        val_on_train_dataset = AV2Dataset(
            shuffle=False,
            use_geom_augmentation=False,
            use_skip_frames="never",
            mode="train",
            cfg=cfg,
        )
        val_on_train_loader = torch.utils.data.DataLoader(
            val_on_train_dataset,
            pin_memory=True,
            batch_size=cfg.data.batch_size,
            num_workers=num_workers,
            collate_fn=lidar_dataset_collate_fn,
            shuffle=True,
            worker_init_fn=worker_init_fn,
            **prefetch_args,
        )
        val_loader = get_av2_val_dataset(
            cfg, use_skip_frames="never", size=None, shuffle=shuffle_validation
        )
    elif cfg.data.source == "waymo":
        train_loader, train_dataset = get_waymo_train_dataset(
            cfg,
            use_geom_augmentation=cfg.data.augmentation.active,
            use_skip_frames=cfg.data.use_skip_frames,
            path_to_augmentation_db=path_to_augmentation_db,
            path_to_mined_boxes_db=path_to_mined_boxes_db,
            size=num_train_samples,
            need_flow_during_training=need_flow_during_training,
        )
        val_on_train_dataset = WaymoDataset(
            shuffle=False,
            use_geom_augmentation=False,
            use_skip_frames="never",
            mode="train",
            cfg=cfg,
        )
        val_on_train_loader = torch.utils.data.DataLoader(
            val_on_train_dataset,
            pin_memory=True,
            batch_size=cfg.data.batch_size,
            num_workers=num_workers,
            collate_fn=lidar_dataset_collate_fn,
            shuffle=True,
            worker_init_fn=worker_init_fn,
            **prefetch_args,
        )
        val_loader = get_waymo_val_dataset(
            cfg, use_skip_frames="never", size=None, shuffle=shuffle_validation
        )
    elif cfg.data.source == "kitti":
        train_loader, train_dataset = get_kitti_train_dataset(
            cfg,
            use_geom_augmentation=cfg.data.augmentation.active,
            use_skip_frames=cfg.data.use_skip_frames,
            path_to_augmentation_db=path_to_augmentation_db,
            path_to_mined_boxes_db=path_to_mined_boxes_db,
            target=target,
            size=num_train_samples,
            need_flow_during_training=need_flow_during_training,
        )
        val_on_train_dataset = KittiTrackingDataset(
            shuffle=False,
            use_geom_augmentation=False,
            use_skip_frames="never",
            mode="val",
            cfg=cfg,
        )
        val_on_train_loader = torch.utils.data.DataLoader(
            val_on_train_dataset,
            pin_memory=True,
            batch_size=cfg.data.batch_size,
            num_workers=num_workers,
            collate_fn=lidar_dataset_collate_fn,
            shuffle=False,
            worker_init_fn=worker_init_fn,
            **prefetch_args,
        )
        val_loader, _ = get_kitti_val_dataset(
            cfg, size=None, target=target, shuffle=shuffle_validation
        )

    else:
        raise NotImplementedError(cfg.data.source)
    return train_loader, train_dataset, val_loader, val_on_train_loader


def log_box_image(
    *,
    cfg,
    pred_boxes,
    gt_boxes,
    occupancy_f32,
    writer,
    writer_prefix,
    step,
    max_num_batches=8,
    gt_background_boxes=None,
    perform_nms=True,
):
    if pred_boxes is not None:
        visu_pred_boxes = pred_boxes.clone()
        if cfg.box_prediction.activations.probs == "none":
            visu_pred_boxes.probs = torch.sigmoid(visu_pred_boxes.probs)
    else:
        visu_pred_boxes = None
    reconstruction_target_box_img = draw_box_image(
        cfg=cfg,
        pred_boxes=visu_pred_boxes,
        gt_boxes=gt_boxes,
        canvas_f32=occupancy_f32,
        gt_background_boxes=gt_background_boxes,
        perform_nms=perform_nms,
    )
    writer.add_image(
        writer_prefix + "_post_nms_" + "SHAPES_COMPARISON_PRED_GT",
        batched_np_img_to_torch_img_grid(
            limit_visu_image_batches(
                reconstruction_target_box_img,
                max_batches=max_num_batches,
            ),
        ),
        global_step=step,
        dataformats="CHW",
    )


def sanity_check_flow(
    cfg, data_loader, writer, global_step, writer_prefix="", max_num_steps=1000
):
    flow_metrics = {"fwd": FlowMetrics(), "rev": FlowMetrics()}
    for idx, data_el in enumerate(tqdm(data_loader, disable=False)):
        if idx >= max_num_steps:
            break
        (dataset_element_t0, dataset_element_t1, _, _) = data_el
        data_els = {"fwd": dataset_element_t0, "rev": dataset_element_t1}
        for flow_dir in {"fwd", "rev"}:
            data_element = data_els[flow_dir]
            points = data_element["pcl_ta"]["pcl"].cpu().numpy()
            if (
                "flow_ta_tb" not in data_element["gt"]
                or cfg.data.flow_source not in data_element
            ):
                print(
                    "Skipping flow sanity check, as gt or source flow is not available!"
                )
                continue
            source_point_flow = (
                data_element[cfg.data.flow_source]["flow_ta_tb"].cpu().numpy()
            )
            gt_point_flow = data_element["gt"]["flow_ta_tb"].cpu().numpy()
            moving_mask = (
                data_element["gt"]["moving_mask"].cpu().numpy()
                & data_element["pcl_ta"]["pcl_is_valid"].cpu().numpy()
                & data_element["gt"]["point_has_valid_flow_label"].cpu().numpy()
            )
            valid_mask = data_element["pcl_ta"]["pcl_is_valid"].cpu().numpy()

            for batch_idx in range(data_element["pcl_ta"]["pcl"].shape[0]):
                flow_metrics[flow_dir].update(
                    points=points[batch_idx],
                    flow_gt=gt_point_flow[batch_idx],
                    flow_pred=source_point_flow[batch_idx],
                    is_moving=moving_mask[batch_idx],
                    mask=valid_mask[batch_idx],
                )

    for flow_dir in {"fwd", "rev"}:
        flow_metrics[flow_dir].log_metrics_curves(
            global_step=global_step,
            summary_writer=writer,
            writer_prefix=str(Path(writer_prefix).joinpath(flow_dir)) + "/",
        )
