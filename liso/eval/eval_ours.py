import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib
import numpy as np
import torch
from config_helper.config import dumb_load_yaml_to_omegaconf, parse_config, save_config
from liso.datasets.argoverse2.av2_torch_dataset import AV2Dataset, get_av2_val_dataset
from liso.datasets.kitti_object_torch_dataset import KittiObjectDataset
from liso.datasets.kitti_tracking_torch_dataset import (
    KittiTrackingDataset,
    get_kitti_val_dataset,
)
from liso.datasets.nuscenes.trafo_conversion import kitti_lidar_T_nusc_vehicle
from liso.datasets.nuscenes_torch_dataset import (
    NuscenesDataset,
    get_nuscenes_val_dataset,
)
from liso.datasets.waymo_torch_dataset import (
    WaymoDataset,
    get_waymo_val_dataset,
    vehicle_Twaymo_lidar,
)
from liso.eval.flow_metrics import FlowMetrics
from liso.eval.nuscenes_metrics_wrapper import NuscenesObjectDetectionMetrics
from liso.eval.od_metrics import ObjectDetectionMetrics, WaymoObjectDetectionMetrics
from liso.kabsch.box_groundtruth_matching import (
    slow_greedy_match_boxes_by_desending_confidence_by_dist,
)
from liso.kabsch.main_utils import get_network_input_pcls
from liso.kabsch.mask_dataset import RecursiveDeviceMover
from liso.kabsch.shape_utils import Shape
from liso.networks.flow_cluster_detector.flow_cluster_detector import (
    FlowClusterDetector,
    fit_bev_box_z_and_height_using_points_in_box,
)
from liso.networks.simple_net.point_rcnn import PointRCNNWrapper
from liso.networks.simple_net.pointpillars import PointPillarsWrapper
from liso.networks.simple_net.simple_net import BoxLearner, select_network
from liso.networks.simple_net.simple_net_utils import load_checkpoint_check_sanity
from liso.utils.config_helper_helper import parse_cli_args, pretty_json
from liso.utils.nms_iou import iou_based_nms
from liso.utils.torch_transformation import torch_decompose_matrix
from liso.visu.bbox_image import (
    create_range_image_w_boxes,
    log_box_movement,
    render_gt_boxes_with_predicted_logits,
)
from liso.visu.flow_image import log_flow_image
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

matplotlib.use("agg")


@torch.no_grad()
def convert_box_ours_to_mmdet3d(
    non_batched_boxes: Shape, idx_to_class_name_mapping: Dict[int, str]
):
    pred_boxes = non_batched_boxes.clone()
    pred_boxes = pred_boxes.drop_padding_boxes()
    box_poses_gt = pred_boxes.get_poses()
    box_pos, box_rot = torch_decompose_matrix(box_poses_gt)
    box_pos[:, 2] -= 0.5 * pred_boxes.dims[:, 2]

    gt_boxes_tensor = torch.cat(
        [
            box_pos,
            pred_boxes.dims,
            box_rot,
        ],
        dim=-1,
    )
    mmdet_gt_bbox = LiDARInstance3DBoxes(
        gt_boxes_tensor,
        box_dim=7,
    )
    labels_3d = np.array(
        [
            idx_to_class_name_mapping[class_id]
            for class_id in pred_boxes.class_id.cpu().numpy().squeeze()
        ]
    )
    box_dict = {
        "boxes_3d": mmdet_gt_bbox.tensor.numpy(),
        "scores_3d": pred_boxes.probs.cpu().numpy(),
        "labels_3d": labels_3d,
    }
    return box_dict


@torch.no_grad()
def count_box_points_in_kitti_annotated_fov(pred_boxes: Shape, pcl: torch.FloatTensor):
    kitti_cam_min_opening_angle__deg = -41.95
    kitti_cam_max_opening_angle__deg = 40.16

    pc_x, pc_y = (
        pcl[:, 0],
        pcl[:, 1],
    )
    angles = torch.atan2(pc_y, pc_x)
    min_angle = kitti_cam_min_opening_angle__deg / 180.0 * np.pi
    max_angle = kitti_cam_max_opening_angle__deg / 180.0 * np.pi
    filter_mask = (angles >= min_angle) & (angles <= max_angle)
    pcl_in_cam_fov = pcl[filter_mask]

    (num_pts_in_box, _, _) = fit_bev_box_z_and_height_using_points_in_box(
        pcl_in_cam_fov[:, :3],
        pred_boxes,
        box_height=1000.0,
    )
    return num_pts_in_box


# @torch.no_grad()
def run_val(
    cfg,
    val_loader,
    box_predictor: BoxLearner,
    mask_gt_renderer,
    writer_prefix: str,
    writer,
    global_step: int,
    max_num_steps=None,
    logit_threshold=-1.0e32,
    incremental_log_every_n_hours=6,
    img_log_interval=None,
    device="cuda:0",
    export_predictions_for_visu=None,
    export_predictions_mmdet3d: Path = None,
):
    if export_predictions_for_visu:
        export_predictions_for_visu = Path(export_predictions_for_visu)
        export_predictions_for_visu.mkdir(parents=True, exist_ok=True)
    if export_predictions_mmdet3d:
        export_predictions_mmdet3d = Path(export_predictions_mmdet3d)
        export_predictions_mmdet3d.mkdir(parents=True, exist_ok=True)
    if isinstance(device, str):
        device = torch.device(device)

    moving_velocity_thresh = (
        0.1
        if not hasattr(cfg.validation, "obj_is_moving_velocity_thresh")
        else cfg.validation.obj_is_moving_velocity_thresh
    )
    num_img_log_samples = 10
    num_extra_tbs = 0
    if max_num_steps is None:
        max_num_steps = len(val_loader) + 1
    if img_log_interval is None:
        img_log_interval = max(max_num_steps // num_img_log_samples, 1)
    eval_start_time = datetime.now()
    print(
        f"{eval_start_time} start eval {writer_prefix} at step {global_step} for {max_num_steps} steps on {val_loader.dataset.__class__.__name__}."
    )

    time_last_logged = time.time()
    range_bins = [(0.0, 1000.0), (0.0, 20.0), (20.0, 40.0), (40.0, 60.0)]

    range_based_od_metrics = {
        k: {
            f"{int(min_range)}_{int(max_range)}m": {
                matching_criterion: ObjectDetectionMetrics(
                    moving_velocity_thresh=moving_velocity_thresh,
                    use_slow_nuscenes_matching=True,
                    min_recall=0.0,
                    min_precision=(
                        0.0 if isinstance(val_loader.dataset, AV2Dataset) else 0.1
                    ),
                    box_matching_criterion=matching_criterion,
                    min_eval_range_m=min_range,
                    max_eval_range_m=max_range,
                )
                for matching_criterion in (
                    "iou_3d",
                    "iou_bev",
                    # "dist", we have NUSC metrics for that now
                )
            }
            for (min_range, max_range) in range_bins
        }
        for k in ("visible", "benchmark")
    }

    range_based_od_metrics["waymo_cropped"] = {
        f"{int(min_range)}_{int(max_range)}m": {
            matching_criterion: ObjectDetectionMetrics(
                moving_velocity_thresh=moving_velocity_thresh,
                use_slow_nuscenes_matching=True,
                min_recall=0.0,
                min_precision=0.0,
                box_matching_criterion=matching_criterion,
                iou_matching_thresholds=(0.3, 0.4, 0.5, 0.7),
                filter_detections_by_bev_area_min_max_m=[-50.0, -20.0, 50.0, 20.0],
                min_eval_range_m=min_range,
                max_eval_range_m=max_range,
            )
            for matching_criterion in (
                "iou_3d",
                "iou_bev",
            )
        }
        for (min_range, max_range) in range_bins
    }
    nusc_metrics = NuscenesObjectDetectionMetrics(eval_movable_classes_as_one=True)
    if isinstance(val_loader.dataset, NuscenesDataset):
        class_based_metrics = NuscenesObjectDetectionMetrics(
            eval_movable_classes_as_one=False
        )
        metric_description = "NUSC_OFFICIAL/per_class"
    elif isinstance(val_loader.dataset, (KittiObjectDataset, KittiTrackingDataset)):
        metric_description = "KITTI/per_class"
        class_based_metrics = ObjectDetectionMetrics(
            moving_velocity_thresh=moving_velocity_thresh,
            use_slow_nuscenes_matching=True,
            class_names=val_loader.dataset.movable_class_names,
            class_idxs=tuple(
                val_loader.dataset.class_name_to_idx_mapping[class_name]
                for class_name in val_loader.dataset.movable_class_names
            ),
            min_recall=0.0,
            box_matching_criterion="iou_bev",
            eval_movable_classes_as_one=False,
        )
    elif isinstance(val_loader.dataset, (AV2Dataset)):
        metric_description = "AV2/per_class"
        class_based_metrics = ObjectDetectionMetrics(
            moving_velocity_thresh=moving_velocity_thresh,
            use_slow_nuscenes_matching=True,
            min_recall=0.0,
            min_precision=0.0,
            box_matching_criterion="iou_bev",
            eval_movable_classes_as_one=False,
        )
    elif isinstance(val_loader.dataset, WaymoDataset):
        metric_description = "WAYMO/per_class"
        class_based_metrics = WaymoObjectDetectionMetrics(
            eval_movable_classes_as_one=False,
            class_names=val_loader.dataset.movable_class_names,
            class_idxs=tuple(
                val_loader.dataset.class_name_to_idx_mapping[class_name]
                for class_name in val_loader.dataset.movable_class_names
            ),
        )
    waymo_metrics = {
        f"{int(min_range)}_{int(max_range)}m": WaymoObjectDetectionMetrics(
            min_eval_range_m=min_range,
            max_eval_range_m=max_range,
        )
        for (min_range, max_range) in range_bins
    }
    flow_metrics = FlowMetrics(range_bins=(0.0, 25.0, 50.0, 75.0, 100.0))
    for val_step, train_data in enumerate(
        tqdm(val_loader, total=max_num_steps, disable=False)
    ):
        if max_num_steps is not None and val_step > max_num_steps:
            break
        trigger_img_logging = val_step % img_log_interval == 0
        trigger_export_for_visu = trigger_img_logging

        sample_data_t0, sample_data_t1, _, meta_data = mask_gt_renderer(
            train_data,
            need_sample_data_t1=(trigger_img_logging),
            need_augm_sample_data_t0=False,
        )

        centermaps_gt = None

        gt_boxes = sample_data_t0["gt"]["boxes"].to(device)
        benchmark_gt_boxes = sample_data_t0["gt"]["boxes_nusc"].to(device)
        sample_name = meta_data["sample_id"]
        writer.add_text(
            writer_prefix.rstrip("/") + "/sample_name",
            ",".join(sample_name),
            global_step=global_step + val_step,
        )
        if isinstance(box_predictor, Dict):
            with torch.no_grad():
                pred_boxes = []
                for sn in sample_name:
                    if sn in box_predictor:
                        pred_boxes.append(
                            Shape(**box_predictor[sn]["raw_box"]).to_tensor()
                        )
                    else:
                        pred_boxes.append(Shape.createEmpty().to_tensor())
                pred_boxes = Shape.from_list_of_shapes(pred_boxes).to(device)
                pred_boxes.class_id = pred_boxes.class_id.to(gt_boxes.class_id.dtype)
                pred_boxes_maps = None
                if (
                    cfg.data.flow_source in sample_data_t0
                    and "flow_ta_tb" in sample_data_t0[cfg.data.flow_source]
                ):
                    pred_flow = (
                        sample_data_t0[cfg.data.flow_source]["flow_ta_tb"].cpu().numpy()
                    )
                else:
                    pred_flow = None

        else:
            with torch.no_grad():
                if isinstance(box_predictor, (FlowClusterDetector,)):
                    pred_boxes = box_predictor(
                        sample_data_t0,
                        writer=writer,
                        writer_prefix=writer_prefix + "/flow_cluster_detector/",
                        global_step=val_step % img_log_interval,
                        is_batched=True,
                    )
                    pred_boxes_maps = None
                else:
                    pred_boxes, _, pred_boxes_maps, _ = box_predictor(
                        None,
                        get_network_input_pcls(
                            cfg, sample_data_t0, "ta", to_device=device
                        ),
                        gt_boxes,
                        centermaps_gt,
                        train=False,
                    )

                if (
                    cfg.data.flow_source in sample_data_t0
                    and "flow_ta_tb" in sample_data_t0[cfg.data.flow_source]
                ):
                    pred_flow = (
                        sample_data_t0[cfg.data.flow_source]["flow_ta_tb"].cpu().numpy()
                    )
                else:
                    pred_flow = None

        pred_boxes = pred_boxes.to(device)
        assert pred_boxes.rot.shape[-1] == 1, pred_boxes.rot.shape
        if (
            cfg.data.flow_source != "gt" and cfg.data.flow_source in sample_data_t0
        ) and "flow_ta_tb" in sample_data_t0["gt"]:
            points = sample_data_t0["pcl_ta"]["pcl"].cpu().numpy()

            gt_point_flow = sample_data_t0["gt"]["flow_ta_tb"].cpu().numpy()
            moving_mask = (
                sample_data_t0["gt"]["moving_mask"].cpu().numpy()
                & sample_data_t0["pcl_ta"]["pcl_is_valid"].cpu().numpy()
                & sample_data_t0["gt"]["point_has_valid_flow_label"].cpu().numpy()
            )
            valid_mask = sample_data_t0["pcl_ta"]["pcl_is_valid"].cpu().numpy()

            for batch_idx in range(sample_data_t0["pcl_ta"]["pcl"].shape[0]):
                flow_metrics.update(
                    points=points[batch_idx],
                    flow_gt=gt_point_flow[batch_idx],
                    flow_pred=pred_flow[batch_idx],
                    is_moving=moving_mask[batch_idx],
                    mask=valid_mask[batch_idx],
                )

        with torch.no_grad():
            pred_boxes_after_nms_threshold_number_limit = []
            for batch_idx in range(gt_boxes.pos.shape[0]):
                non_batched_pred_boxes = pred_boxes[batch_idx].drop_padding_boxes()

                box_confidence_too_low = torch.squeeze(
                    non_batched_pred_boxes.probs < logit_threshold, dim=-1
                )
                non_batched_pred_boxes = non_batched_pred_boxes[~box_confidence_too_low]
                pred_boxes_for_nms_w_valid_probability = non_batched_pred_boxes.clone()
                if cfg.box_prediction.activations.probs == "none" and not (
                    isinstance(box_predictor, BoxLearner)
                    and isinstance(
                        box_predictor.model, (PointPillarsWrapper, PointRCNNWrapper)
                    )
                ):
                    pred_boxes_for_nms_w_valid_probability.probs = torch.sigmoid(
                        pred_boxes_for_nms_w_valid_probability.probs
                    )

                nms_pred_box_idxs = iou_based_nms(
                    pred_boxes_for_nms_w_valid_probability,
                    overlap_threshold=cfg.nms_iou_threshold,
                    post_nms_max_boxes=500,
                )
                non_batched_pred_boxes = non_batched_pred_boxes[nms_pred_box_idxs]
                if isinstance(
                    val_loader.dataset, (KittiTrackingDataset, KittiObjectDataset)
                ):
                    # filter detections that fell into areas that have no labels
                    MIN_NUM_PTS_PER_BOX = 10
                    enough_points = (
                        count_box_points_in_kitti_annotated_fov(
                            non_batched_pred_boxes,
                            sample_data_t0["pcl_full_w_ground_ta"][batch_idx].to(
                                non_batched_pred_boxes.pos.device
                            ),
                        )
                        >= MIN_NUM_PTS_PER_BOX
                    )
                    non_batched_pred_boxes.valid = (
                        enough_points & non_batched_pred_boxes.valid
                    )
                    non_batched_pred_boxes = non_batched_pred_boxes.drop_padding_boxes()
                non_batched_gt_boxes = gt_boxes[batch_idx].drop_padding_boxes()

                class_label_src = cfg.data.setdefault("class_label_source", "gt")
                if class_label_src == "gt":
                    class_transfer_matching_threshold = 3.0
                    (
                        idxs_into_gt,
                        idxs_into_preds,
                        _,
                        _,
                        _,
                    ) = slow_greedy_match_boxes_by_desending_confidence_by_dist(
                        non_batched_gt_boxes.pos,
                        non_batched_pred_boxes.pos,
                        non_batched_pred_confidence=torch.squeeze(
                            non_batched_pred_boxes.probs, dim=-1
                        ),
                        matching_threshold=class_transfer_matching_threshold,
                        match_in_nd=2,
                    )
                    if hasattr(val_loader.dataset, "movable_class_frequencies"):
                        class_ids_shape = non_batched_pred_boxes.class_id.shape
                        class_freqs = val_loader.dataset.movable_class_frequencies
                        dummy_class_ids = np.random.choice(
                            np.arange(0, len(class_freqs)),
                            size=class_ids_shape,
                            p=class_freqs,
                        )
                        non_batched_pred_boxes.class_id = (
                            torch.from_numpy(dummy_class_ids)
                            .to(non_batched_pred_boxes.class_id.device)
                            .to(non_batched_pred_boxes.class_id.dtype)
                        )
                    else:
                        non_batched_pred_boxes.class_id = torch.randint(
                            low=0,
                            high=len(val_loader.dataset.movable_class_names),
                            size=non_batched_pred_boxes.class_id.shape,
                            device=non_batched_pred_boxes.class_id.device,
                            dtype=non_batched_pred_boxes.class_id.dtype,
                        )
                    non_batched_pred_boxes.class_id[
                        idxs_into_preds
                    ] = non_batched_gt_boxes.class_id[idxs_into_gt]

                    if val_step < 10:
                        assert torch.all(
                            torch.linalg.norm(
                                non_batched_pred_boxes.pos[idxs_into_preds][:, :2]
                                - non_batched_gt_boxes.pos[idxs_into_gt][:, :2],
                                dim=-1,
                            )
                            <= class_transfer_matching_threshold
                        ), "if matching worked correctly, this cannot happen"

                else:
                    raise NotImplementedError(class_label_src)
                pred_boxes_after_nms_threshold_number_limit.append(
                    non_batched_pred_boxes
                )

                non_batched_benchmark_boxes = benchmark_gt_boxes[
                    batch_idx
                ].drop_padding_boxes()
                for metric_category in ("visible", "waymo_cropped", "benchmark"):
                    for range_bin_str in range_based_od_metrics[metric_category]:
                        for metr_coll in range_based_od_metrics[metric_category][
                            range_bin_str
                        ].values():
                            if metric_category in ("visible", "waymo_cropped"):
                                metr_coll.update(
                                    non_batched_gt_boxes=non_batched_gt_boxes,
                                    non_batched_pred_boxes=non_batched_pred_boxes,
                                    sample_token=meta_data["sample_id"][batch_idx],
                                )
                            elif metric_category == "benchmark":
                                metr_coll.update(
                                    non_batched_gt_boxes=non_batched_benchmark_boxes,
                                    non_batched_pred_boxes=non_batched_pred_boxes,
                                    sample_token=meta_data["sample_id"][batch_idx],
                                )
                            else:
                                raise NotImplementedError(metric_category)
                for w_metr in waymo_metrics.values():
                    w_metr.update(
                        non_batched_gt_boxes=non_batched_benchmark_boxes,
                        non_batched_pred_boxes=non_batched_pred_boxes,
                        sample_token=meta_data["sample_id"][batch_idx],
                    )
                nusc_metric_pred_boxes = non_batched_pred_boxes.clone()
                if (
                    cfg.box_prediction.activations.probs == "none"
                    and not isinstance(
                        box_predictor,
                        (FlowClusterDetector,),
                    )
                    and not (
                        isinstance(box_predictor, BoxLearner)
                        and isinstance(
                            box_predictor.model, (PointPillarsWrapper, PointRCNNWrapper)
                        )
                    )
                ):
                    nusc_metric_pred_boxes.probs = torch.sigmoid(
                        nusc_metric_pred_boxes.probs
                    )
                nusc_metrics.update(
                    non_batched_gt_boxes=non_batched_gt_boxes,
                    non_batched_pred_boxes=nusc_metric_pred_boxes,
                    sample_token=meta_data["sample_id"][batch_idx],
                )
                class_based_metrics.update(
                    non_batched_gt_boxes=non_batched_gt_boxes,
                    non_batched_pred_boxes=nusc_metric_pred_boxes,
                    sample_token=meta_data["sample_id"][batch_idx],
                )

            if export_predictions_for_visu and trigger_export_for_visu:
                sample_ids = meta_data["sample_id"]
                for batch_idx, sample_id in enumerate(sample_ids):
                    export_data = {
                        "points_xyzi": sample_data_t0["pcl_full_w_ground_ta"][batch_idx]
                        .cpu()
                        .numpy()
                        .astype(np.float32),
                        "pred": {
                            "boxes": pred_boxes_after_nms_threshold_number_limit[
                                batch_idx
                            ]
                            .clone()
                            .cpu()
                            .numpy()
                            .__dict__
                        },
                        "gt": {
                            "boxes": gt_boxes[batch_idx]
                            .clone()
                            .drop_padding_boxes()
                            .cpu()
                            .numpy()
                            .__dict__
                        },
                    }
                    export_sample_name = (
                        export_predictions_for_visu / sample_id.replace("/", "_")
                    )
                    np.savez_compressed(export_sample_name, export_data)

            if export_predictions_mmdet3d:
                for exp_batch_idx, sample_id in enumerate(meta_data["sample_id"]):
                    exp_boxes_sensor_coords: Shape = (
                        pred_boxes_after_nms_threshold_number_limit[exp_batch_idx]
                        .clone()
                        .cpu()
                    )
                    if isinstance(val_loader.dataset, NuscenesDataset):
                        boxes_official_coords = exp_boxes_sensor_coords.transform(
                            torch.from_numpy(np.linalg.inv(kitti_lidar_T_nusc_vehicle))
                        )
                    elif isinstance(
                        val_loader.dataset, (KittiObjectDataset, AV2Dataset)
                    ):
                        boxes_official_coords = exp_boxes_sensor_coords.clone()
                    elif isinstance(val_loader.dataset, WaymoDataset):
                        boxes_official_coords = exp_boxes_sensor_coords.transform(
                            torch.from_numpy(np.linalg.inv(vehicle_Twaymo_lidar))
                        )
                    else:
                        raise NotImplementedError(val_loader.dataset.__class__.__name__)

                    mmdet3d_boxes = convert_box_ours_to_mmdet3d(
                        boxes_official_coords,
                        val_loader.dataset.idx_to_class_name_mapping,
                    )
                    pkl_export_sample_name = (
                        export_predictions_mmdet3d / sample_id.replace("/", "_")
                    ).with_suffix(".pkl")
                    with open(pkl_export_sample_name, "wb") as file:
                        pickle.dump(mmdet3d_boxes, file)

            if trigger_img_logging:
                if (
                    num_extra_tbs > 0
                    and "sample_id" in meta_data
                    and meta_data["sample_id"] is not None
                ):
                    sample_ids = meta_data["sample_id"]
                    if cfg.data.source == "nuscenes":
                        tb_prefix = "|".join(
                            [
                                "_".join(el.split("-")[1].split("_")[:2])
                                for el in sample_ids
                            ]
                        )
                    elif cfg.data.source == "kitti":
                        tb_prefix = "_|_".join(sample_ids)
                    elif cfg.data.source == "waymo":
                        sids = [
                            sid.split("-")[-1].replace("_with_camera_labels/", "")
                            for sid in sample_ids
                        ]
                        tb_prefix = "_|_".join(sids)
                    else:
                        raise NotImplementedError(cfg.data.source)

                    prefix = writer_prefix + tb_prefix + "/"
                    num_extra_tbs -= 1
                    step = global_step
                else:
                    prefix = writer_prefix
                    step = global_step + val_step
                log_box_movement(
                    cfg=cfg,
                    writer=writer,
                    global_step=step,
                    sample_data_a=sample_data_t0,
                    sample_data_b=sample_data_t1,
                    pred_boxes=Shape.from_list_of_shapes(
                        pred_boxes_after_nms_threshold_number_limit
                    ),
                    writer_prefix=prefix,
                )
                if "flow_bev_ta_tb" in sample_data_t0["gt"]:
                    log_flow_image(
                        cfg=cfg,
                        writer=writer,
                        global_step=global_step,
                        flow_2d=sample_data_t0["gt"]["flow_bev_ta_tb"][
                            :, :, :, :2
                        ].permute((0, 3, 1, 2)),
                        prefix=prefix,
                        suffix="/flow/GT",
                    )
                if pred_boxes_maps is not None and cfg.network.name == "centerpoint":
                    gt_img_with_pred_logis = render_gt_boxes_with_predicted_logits(
                        cfg, mask_gt_renderer, sample_data_t0, gt_boxes, pred_boxes_maps
                    )

                    writer.add_images(
                        prefix + "gt_box_with_pred_logits",
                        gt_img_with_pred_logis,
                        global_step=step,
                        dataformats="NHWC",
                    )

                max_num_pred_visu_boxes = 20
                pred_visu_boxes = []
                for all_boxes in pred_boxes_after_nms_threshold_number_limit:
                    if all_boxes.shape[0] > max_num_pred_visu_boxes:
                        keep_top_box_idxs = torch.argsort(
                            torch.squeeze(all_boxes.probs, dim=-1), descending=True
                        )[:max_num_pred_visu_boxes]
                        pred_visu_boxes.append(all_boxes[keep_top_box_idxs])
                    else:
                        pred_visu_boxes.append(all_boxes)

                pred_visu_boxes = Shape.from_list_of_shapes(pred_visu_boxes)

                img_canvas = create_range_image_w_boxes(
                    pcls=get_network_input_pcls(
                        cfg, sample_data_t0, "ta", to_device=device
                    ),
                    boxes=gt_boxes,
                    fitted_boxes=pred_visu_boxes,
                )

                writer.add_images(
                    prefix + "range_images",
                    img_canvas,
                    dataformats="NHWC",
                    global_step=step,
                )

        if (
            incremental_log_every_n_hours is not None
            and (time.time() - time_last_logged) > 3600 * incremental_log_every_n_hours
        ):
            for metric_category in range_based_od_metrics:
                for range_bin_str in range_based_od_metrics[metric_category]:
                    for metr_coll in range_based_od_metrics[metric_category][
                        range_bin_str
                    ].values():
                        metr_coll.log(
                            global_step + val_step,
                            summary_writer=writer,
                            writer_prefix=f"{writer_prefix}interm_result/{metric_category}/detection_metrics/{range_bin_str}",
                        )
            flow_metrics.log_metrics_curves(
                global_step + val_step,
                summary_writer=writer,
                writer_prefix=writer_prefix + "interm_result/flow_metrics/",
            )
            nusc_metrics.log(
                global_step + val_step,
                summary_writer=writer,
                writer_prefix=writer_prefix
                + "interm_result/NUSC_OFFICIAL/detection_metrics/",
                render_curves_to=Path(writer.log_dir).parent.joinpath(
                    "intermediate_nuscenes_pdfs"
                ),
            )
            class_based_metrics.log(
                global_step + val_step,
                summary_writer=writer,
                writer_prefix=writer_prefix
                + f"interm_result/{metric_description}/detection_metrics/",
            )
            for range_str, w_metric in waymo_metrics.items():
                w_metric.log(
                    global_step + val_step,
                    summary_writer=writer,
                    writer_prefix=writer_prefix
                    + f"interm_result/WAYMO/detection_metrics/{range_str}",
                )
            time_last_logged = time.time()
    for metric_category in range_based_od_metrics:
        for range_bin_str in range_based_od_metrics[metric_category]:
            for metr_coll in range_based_od_metrics[metric_category][
                range_bin_str
            ].values():
                metr_coll.log(
                    global_step + val_step,
                    summary_writer=writer,
                    writer_prefix=f"{writer_prefix}final_result/{metric_category}/detection_metrics/{range_bin_str}",
                )
    nusc_metrics.log(
        global_step,
        summary_writer=writer,
        writer_prefix=writer_prefix + "final_result/NUSC_OFFICIAL/detection_metrics/",
        render_curves_to=Path(writer.log_dir).parent.joinpath("final_nuscenes_pdfs"),
    )
    class_based_metrics.log(
        global_step,
        summary_writer=writer,
        writer_prefix=writer_prefix
        + f"final_result/{metric_description}/detection_metrics/",
    )
    for range_str, w_metric in waymo_metrics.items():
        w_metric.log(
            global_step,
            summary_writer=writer,
            writer_prefix=writer_prefix
            + f"final_result/WAYMO/detection_metrics/{range_str}",
        )
    flow_metrics.log_metrics_curves(
        global_step,
        summary_writer=writer,
        writer_prefix=writer_prefix + "final_result/flow_metrics/",
    )
    eval_end_time = datetime.now()
    print(
        f"{eval_end_time} finished {val_step} eval step. Took {eval_end_time - eval_start_time}"
    )


def main():
    args = parse_cli_args()
    if args.summary_dir is None:
        maybe_slow_log_dir = Path("/tmp/od_custom_eval")
    else:
        maybe_slow_log_dir = Path(args.summary_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.load_checkpoint:
        args.load_checkpoint = Path(args.load_checkpoint)
        cfg_path_chkpt = args.load_checkpoint.parent.parent.joinpath("config.yml")
        exp_desc = Path(args.load_checkpoint.as_posix().split("/")[-5]).joinpath(
            args.load_checkpoint.stem
        )
        cfg = dumb_load_yaml_to_omegaconf(cfg_path_chkpt)

        box_predictor = select_network(cfg, device=torch.device("cuda:0"))

        box_predictor = load_checkpoint_check_sanity(
            args.load_checkpoint, cfg, box_predictor
        )
        default_cfg = parse_config(args.config_file)
        cfg.data.paths = default_cfg.data.paths
    else:
        cfg = parse_config(
            args.config_file,
            extra_cfg_args=args.configs,
            key_value_updates=args.keys_value,
        )

        assert cfg.network.name in (
            "flow_cluster_detector",
            "echo_gt",
        ), cfg.network.name
        exp_desc = cfg.network.name
        if cfg.network.name == "flow_cluster_detector":
            exp_desc = Path(exp_desc) / (cfg.data.flow_source + "_flow")
        if cfg.network.name in ("echo_gt",):
            exp_desc = Path(exp_desc) / cfg.network.name
        box_predictor = select_network(cfg, device=device)

    cfg.data.num_workers = 12
    start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    maybe_slow_log_dir = maybe_slow_log_dir.joinpath(exp_desc, start_time_str)
    maybe_slow_log_dir.mkdir(exist_ok=True, parents=True)
    save_config(cfg, maybe_slow_log_dir.joinpath("config.yml"))

    mask_gt_renderer = RecursiveDeviceMover(cfg).cuda()

    if cfg.data.source == "nuscenes":
        val_loader = get_nuscenes_val_dataset(
            cfg,
            use_skip_frames="never",
            size=None,
            shuffle=True,
        )
    elif cfg.data.source == "kitti":
        val_loader, _ = get_kitti_val_dataset(
            cfg,
            size=None,
            target="object",
            use_skip_frames="never",
            shuffle=True,
        )
    elif cfg.data.source == "waymo":
        val_loader = get_waymo_val_dataset(
            cfg,
            size=None,
            use_skip_frames="never",
            shuffle=True,
        )
    elif cfg.data.source == "av2":
        val_loader = get_av2_val_dataset(
            cfg,
            size=None,
            use_skip_frames="never",
            shuffle=True,
        )
    else:
        raise NotImplementedError(cfg.data.source)

    log_dir = maybe_slow_log_dir
    writer = SummaryWriter(log_dir)
    max_num_fast_test_steps = 5
    max_num_steps = max_num_fast_test_steps if args.fast_test else None

    writer.add_text("config", pretty_json(cfg), 0)

    run_val(
        cfg,
        val_loader,
        box_predictor,
        mask_gt_renderer,
        "online_val/",
        writer=writer,
        global_step=0,
        max_num_steps=100,
    )
    run_val(
        cfg,
        val_loader,
        box_predictor=box_predictor,
        mask_gt_renderer=mask_gt_renderer,
        writer_prefix="full_eval/",
        writer=writer,
        global_step=0,
        max_num_steps=max_num_steps,
        incremental_log_every_n_hours=4,
        export_predictions_for_visu=args.export_predictions_for_visu,
    )


if __name__ == "__main__":
    main()
