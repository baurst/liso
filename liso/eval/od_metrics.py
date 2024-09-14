import gc
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import numpy as np
import torch
from liso.kabsch.box_groundtruth_matching import (
    match_bboxes,
    slow_greedy_match_boxes_by_desending_confidence_by_dist,
)
from liso.kabsch.box_groundtruth_matching_iou import (
    match_boxes_by_descending_confidence_iou,
)
from liso.kabsch.shape_utils import Shape
from liso.utils.config_helper_helper import pretty_json
from liso.visu.utils import plot_to_np_image
from matplotlib import pyplot as plt
from sklearn.metrics import det_curve, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.metrics._ranking import _binary_clf_curve

matplotlib.use("agg")


def calc_ap(precisions, min_recall: float, min_precision: float) -> float:
    """Calculated average precision."""

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(precisions)
    if min_recall != 0.0:
        assert len(prec) == 101, len(prec)
    prec = prec[
        round(100 * min_recall) + 1 :
    ]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def get_conf_prec_rec(all_gt, all_scores, all_is_fn, use_interpolation=True):
    sortind = np.argsort(-all_scores)
    # COPIED FROM NUSCENES EVAL CODE:
    # sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    for idx in sortind:
        if all_is_fn[idx]:
            # is FN -> not relevant here
            continue
        if all_gt[idx]:
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
        conf.append(all_scores[idx])
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(np.count_nonzero(all_gt))
    if use_interpolation:
        rec_interp = np.linspace(0, 1, 101)  # 101 steps, from 0% to 100% recall.
        if prec.size > 0:
            # will crash on empty prec -> replace with nans in that case
            prec = np.interp(rec_interp, rec, prec, right=0)
        else:
            prec = np.nan * rec_interp
        if conf.size > 0:
            conf = np.interp(rec_interp, rec, conf, right=0)
        else:
            conf = np.nan * rec_interp
        rec = rec_interp
    return conf, prec, rec


def scale_iou(box_sizes_a: np.ndarray, box_sizes_b: np.ndarray) -> float:
    assert box_sizes_a.shape == box_sizes_b.shape, (
        box_sizes_a.shape,
        box_sizes_b.shape,
    )
    assert box_sizes_a.shape[-1] in (2, 3), box_sizes_a.shape

    # IOU
    min_wlh = np.minimum(box_sizes_a, box_sizes_b)
    volume_annotation = np.prod(box_sizes_a, axis=-1)
    volume_result = np.prod(box_sizes_b, axis=-1)
    intersection = np.prod(min_wlh, axis=-1)
    union = volume_annotation + volume_result - intersection
    iou = intersection / np.maximum(union, 1e-6)

    return iou


def angle_diff(
    gt_yaw: np.ndarray, pred_yaw: np.ndarray, period: float = 2 * np.pi
) -> float:
    diff = (gt_yaw - pred_yaw + period / 2) % period - period / 2
    diff = np.where(diff > np.pi, diff - (2 * np.pi), diff)  # wraparound angle
    return diff


def abs_yaw_diff(
    gt_yaw: np.ndarray, pred_yaw: np.ndarray, period: float = 2 * np.pi
) -> float:
    diff = angle_diff(gt_yaw, pred_yaw, period=period)
    return np.abs(diff)


class BaseODMetrics:
    def __init__(self) -> None:
        self.line_width = 1
        self.legend_font_size = 6
        self.dpi = 200
        self.figure_size = None  # (4, 3)
        self.bev_range_min_xy_m = None
        self.bev_range_max_xy_m = None

    def filter_boxes_to_be_in_bev_range(
        self,
        non_batched_boxes,
    ):
        box_is_in_eval_fov = torch.logical_and(
            non_batched_boxes.pos[:, :2]
            >= self.bev_range_min_xy_m[None, ...].to(non_batched_boxes.pos.device),
            non_batched_boxes.pos[:, :2]
            <= self.bev_range_max_xy_m[None, ...].to(non_batched_boxes.pos.device),
        ).all(axis=-1)

        non_batched_boxes.valid = non_batched_boxes.valid & box_is_in_eval_fov
        non_batched_boxes = non_batched_boxes.drop_padding_boxes()
        return non_batched_boxes

    def filter_boxes_to_be_in_abs_range(
        self,
        non_batched_boxes,
    ):
        box_range_xy = torch.linalg.norm(non_batched_boxes.pos[:, :2], dim=-1)
        box_is_in_range = (self.min_eval_range_m <= box_range_xy) & (
            box_range_xy < self.max_eval_range_m
        )

        non_batched_boxes.valid = non_batched_boxes.valid & box_is_in_range
        non_batched_boxes = non_batched_boxes.drop_padding_boxes()
        return non_batched_boxes

    def filter_boxes_to_be_of_class(self, non_batched_boxes: Shape, class_idx: int):
        non_batched_boxes.valid = non_batched_boxes.valid & (
            class_idx == torch.squeeze(non_batched_boxes.class_id, dim=-1)
        )
        non_batched_boxes = non_batched_boxes.drop_padding_boxes()
        return non_batched_boxes


class ObjectDetectionMetrics(BaseODMetrics):
    def __init__(
        self,
        moving_velocity_thresh: float,
        eval_movable_classes_as_one: bool = True,
        class_names: Tuple[str] = ("overall",),
        class_idxs: Tuple[int] = (0,),
        min_precision=0.1,
        min_recall=0.1,
        use_slow_nuscenes_matching=False,
        box_matching_criterion="dist",
        iou_matching_thresholds=(0.25, 0.3, 0.4, 0.5),
        filter_detections_by_bev_area_min_max_m=None,
        min_eval_range_m=None,
        max_eval_range_m=None,
    ) -> None:
        super().__init__()
        self.min_eval_range_m = min_eval_range_m
        self.max_eval_range_m = max_eval_range_m
        self.eval_movable_classes_as_one = eval_movable_classes_as_one
        if class_names == ("overall",):
            class_idxs = (0,)
        assert len(class_names) == len(class_idxs), (class_names, class_idxs)
        self.class_idxs = class_idxs
        self.class_names = class_names
        if box_matching_criterion == "dist":
            dist_matching_thresholds = (0.5, 1.0, 2.0, 4.0)
            self.tp_metric_thresh = 2.0
            self.threshold_unit = "m"
            self.matching_thresholds = dist_matching_thresholds
        elif box_matching_criterion in ("iou_3d", "iou_bev"):
            self.tp_metric_thresh = 0.5
            self.threshold_unit = box_matching_criterion
            self.matching_thresholds = iou_matching_thresholds
        else:
            raise NotImplementedError(box_matching_criterion)
        assert self.tp_metric_thresh in self.matching_thresholds
        self.box_matching_criterion = box_matching_criterion
        if filter_detections_by_bev_area_min_max_m is not None:
            self.bev_range_min_xy_m = torch.tensor(
                filter_detections_by_bev_area_min_max_m[:2]
            )
            self.bev_range_max_xy_m = torch.tensor(
                filter_detections_by_bev_area_min_max_m[2:]
            )
        self.filter_detections_by_bev_area_min_max = (
            filter_detections_by_bev_area_min_max_m
        )
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.use_slow_nuscenes_matching = use_slow_nuscenes_matching

        self.extra_categories = {"overall", "moving", "still"}

        self.per_class_per_thresh_per_category_gt_labels = {}
        self.per_class_per_thresh_per_category_scores = {}
        self.per_class_per_thresh_per_category_is_fn = {}
        self.per_class_per_thresh_label_stats = {}
        self.per_class_per_thresh_tp_errors_running_stats = {}
        for class_name in self.class_names:
            self.per_class_per_thresh_per_category_gt_labels[class_name] = {}
            self.per_class_per_thresh_per_category_scores[class_name] = {}
            self.per_class_per_thresh_per_category_is_fn[class_name] = {}
            self.per_class_per_thresh_label_stats[class_name] = {}
            self.per_class_per_thresh_tp_errors_running_stats[class_name] = {}
            for thresh in self.matching_thresholds:
                self.per_class_per_thresh_per_category_gt_labels[class_name][
                    thresh
                ] = {}
                self.per_class_per_thresh_per_category_scores[class_name][thresh] = {}
                self.per_class_per_thresh_per_category_is_fn[class_name][thresh] = {}
                self.per_class_per_thresh_label_stats[class_name][thresh] = {}
                self.per_class_per_thresh_tp_errors_running_stats[class_name][
                    thresh
                ] = {"AOE": 0.0, "ASE": 0.0, "ATE": 0.0, "tps": 0}
                for cat in self.extra_categories:
                    self.per_class_per_thresh_per_category_gt_labels[class_name][
                        thresh
                    ][cat] = []
                    self.per_class_per_thresh_per_category_scores[class_name][thresh][
                        cat
                    ] = []
                    self.per_class_per_thresh_per_category_is_fn[class_name][thresh][
                        cat
                    ] = []
                    self.per_class_per_thresh_label_stats[class_name][thresh][cat] = 0

        self.moving_velocity_thresh = moving_velocity_thresh

    def update(
        self,
        *,
        non_batched_gt_boxes: Shape,
        non_batched_pred_boxes: Shape,
        sample_token: str,
    ):
        if self.filter_detections_by_bev_area_min_max is not None:
            filtered_non_batched_gt_boxes = self.filter_boxes_to_be_in_bev_range(
                non_batched_gt_boxes.clone()
            )
            filtered_non_batched_pred_boxes = self.filter_boxes_to_be_in_bev_range(
                non_batched_pred_boxes.clone()
            )
        else:
            filtered_non_batched_gt_boxes = non_batched_gt_boxes.clone()
            filtered_non_batched_pred_boxes = non_batched_pred_boxes.clone()

        if self.max_eval_range_m is not None and self.min_eval_range_m is not None:
            filtered_non_batched_gt_boxes = self.filter_boxes_to_be_in_abs_range(
                filtered_non_batched_gt_boxes
            )
            filtered_non_batched_pred_boxes = self.filter_boxes_to_be_in_abs_range(
                filtered_non_batched_pred_boxes
            )

        assert len(self.class_idxs) == len(self.class_names), (
            self.class_idxs,
            self.class_names,
        )
        for class_idx, class_name in zip(self.class_idxs, self.class_names):
            if class_name == "overall":
                class_specific_gt_boxes = filtered_non_batched_gt_boxes
                class_specific_pred_boxes = filtered_non_batched_pred_boxes
            else:
                class_specific_gt_boxes = self.filter_boxes_to_be_of_class(
                    filtered_non_batched_gt_boxes.clone(), class_idx=class_idx
                )
                class_specific_pred_boxes = self.filter_boxes_to_be_of_class(
                    filtered_non_batched_pred_boxes.clone(), class_idx=class_idx
                )
            for matching_threshold in self.matching_thresholds:
                self.update_for_specific_class_and_threshold(
                    non_batched_gt_boxes=class_specific_gt_boxes,
                    non_batched_pred_boxes=class_specific_pred_boxes,
                    matching_threshold=matching_threshold,
                    class_name=class_name,
                )

    def update_for_specific_class_and_threshold(
        self,
        *,
        non_batched_gt_boxes: Shape,
        non_batched_pred_boxes: Shape,
        matching_threshold: float,
        class_name: str,
    ):
        if self.use_slow_nuscenes_matching:
            assert non_batched_gt_boxes.valid.all(), "invalid objects not supported!"
            assert non_batched_pred_boxes.valid.all(), "invalid objects not supported!"
            if self.box_matching_criterion == "dist":
                (
                    idxs_into_gt,
                    idxs_into_preds,
                    _,
                    matched_preds_mask,
                    det_gts_mask,
                ) = slow_greedy_match_boxes_by_desending_confidence_by_dist(
                    non_batched_gt_boxes.pos,
                    non_batched_pred_boxes.pos,
                    non_batched_pred_confidence=torch.squeeze(
                        non_batched_pred_boxes.probs, dim=-1
                    ),
                    matching_threshold=matching_threshold,
                    match_in_nd=2,
                )
            elif self.box_matching_criterion in ("iou_bev", "iou_3d"):
                (
                    idxs_into_gt,
                    idxs_into_preds,
                    _,
                    matched_preds_mask,
                    det_gts_mask,
                ) = match_boxes_by_descending_confidence_iou(
                    non_batched_gt_boxes,
                    non_batched_pred_boxes,
                    matching_threshold=matching_threshold,
                    iou_mode=self.box_matching_criterion,
                    matching_mode="greedy",
                )
            else:
                raise NotImplementedError(self.box_matching_criterion)

        else:
            assert self.box_matching_criterion == "dist", self.box_matching_criterion
            (
                idxs_into_gt,
                idxs_into_preds,
                _,
                matched_preds_mask,
                det_gts_mask,
            ) = match_bboxes(
                non_batched_gt_boxes.pos,
                non_batched_pred_boxes.pos,
                DIST_MATCHING_THRESHOLD=matching_threshold,
                match_in_nd=2,
            )
        np_non_batched_gt_boxes = non_batched_gt_boxes.numpy()
        np_non_batched_pred_boxes = non_batched_pred_boxes.numpy()
        pred_logits = np.squeeze(np_non_batched_pred_boxes.probs, axis=-1)

        gt_box_velos = np.linalg.norm(np_non_batched_gt_boxes.velo, axis=-1)
        gt_box_is_moving_mask = gt_box_velos > self.moving_velocity_thresh

        assert det_gts_mask.shape == gt_box_is_moving_mask.shape, (
            det_gts_mask.shape,
            gt_box_is_moving_mask.shape,
        )

        self.update_specific_metrics(
            det_gts_mask=det_gts_mask,
            matched_preds_mask=matched_preds_mask,
            pred_logits=pred_logits,
            idxs_into_gt=idxs_into_gt,
            idxs_into_preds=idxs_into_preds,
            ignore_gt_and_matched_preds_where_true_mask=~gt_box_is_moving_mask,
            matching_threshold=matching_threshold,
            category="moving",
            class_name=class_name,
        )
        self.update_specific_metrics(
            det_gts_mask=det_gts_mask,
            matched_preds_mask=matched_preds_mask,
            pred_logits=pred_logits,
            idxs_into_preds=idxs_into_preds,
            idxs_into_gt=idxs_into_gt,
            ignore_gt_and_matched_preds_where_true_mask=gt_box_is_moving_mask,
            matching_threshold=matching_threshold,
            category="still",
            class_name=class_name,
        )

        self.update_specific_metrics(
            det_gts_mask=det_gts_mask,
            matched_preds_mask=matched_preds_mask,
            pred_logits=pred_logits,
            idxs_into_preds=idxs_into_preds,
            idxs_into_gt=idxs_into_gt,
            ignore_gt_and_matched_preds_where_true_mask=np.zeros_like(
                gt_box_is_moving_mask
            ),
            matching_threshold=matching_threshold,
            category="overall",
            class_name=class_name,
        )

        num_det_success = np.count_nonzero(det_gts_mask)
        self.per_class_per_thresh_tp_errors_running_stats[class_name][
            matching_threshold
        ]["tps"] += num_det_success

        if num_det_success > 0:
            assert len(idxs_into_gt) == len(idxs_into_preds)
            assert len(idxs_into_gt) == num_det_success
            self.per_class_per_thresh_tp_errors_running_stats[class_name][
                matching_threshold
            ]["ATE"] += np.linalg.norm(
                np_non_batched_gt_boxes.pos[idxs_into_gt, :2]
                - np_non_batched_pred_boxes.pos[idxs_into_preds, :2],
                axis=-1,
            ).sum()

            self.per_class_per_thresh_tp_errors_running_stats[class_name][
                matching_threshold
            ]["ASE"] += (
                1.0
                - scale_iou(
                    np_non_batched_gt_boxes.dims[idxs_into_gt, ...],
                    np_non_batched_pred_boxes.dims[idxs_into_preds, ...],
                )
            ).sum()
            if (
                np_non_batched_gt_boxes.rot is not None
                and np_non_batched_pred_boxes.rot is not None
            ):
                self.per_class_per_thresh_tp_errors_running_stats[class_name][
                    matching_threshold
                ]["AOE"] += abs_yaw_diff(
                    np.squeeze(np_non_batched_gt_boxes.rot[idxs_into_gt, ...], axis=-1),
                    np.squeeze(
                        np_non_batched_pred_boxes.rot[idxs_into_preds, ...], axis=-1
                    ),
                ).sum()

    def update_specific_metrics(
        self,
        *,
        det_gts_mask: np.ndarray,
        matched_preds_mask: np.ndarray,
        pred_logits: np.ndarray,
        idxs_into_preds: np.ndarray,
        idxs_into_gt: np.ndarray,
        ignore_gt_and_matched_preds_where_true_mask: np.ndarray,
        matching_threshold: float,
        category: str,
        class_name: str,
    ):
        assert category in self.extra_categories, category
        self.per_class_per_thresh_label_stats[class_name][matching_threshold][
            category
        ] += np.count_nonzero(~ignore_gt_and_matched_preds_where_true_mask)
        # transfer ignore mask over to matched predictions
        ignore_gt_where_true = ignore_gt_and_matched_preds_where_true_mask[idxs_into_gt]
        # use transfered label and idx to to create mask for prediction
        use_this_prediction = np.ones_like(matched_preds_mask)
        use_this_prediction[idxs_into_preds] = ~ignore_gt_where_true

        if category == "overall":
            # must use all, ignore nothing
            assert np.all(
                ~ignore_gt_and_matched_preds_where_true_mask
            ), ignore_gt_and_matched_preds_where_true_mask
            assert np.all(use_this_prediction), use_this_prediction

        # disable predictions that match to ignored gt boxes
        # use all detections by default! do not change defualt, will loose FPs
        specific_pred_logits = pred_logits[use_this_prediction]
        specific_matched_preds_mask = matched_preds_mask[use_this_prediction]
        specific_det_gts_mask = det_gts_mask[
            ~ignore_gt_and_matched_preds_where_true_mask
        ]
        if category == "overall":
            # must use all, ignore nothing
            assert np.all(specific_det_gts_mask == det_gts_mask), (
                specific_det_gts_mask,
                det_gts_mask,
            )

        num_tps = np.count_nonzero(specific_det_gts_mask)
        assert num_tps == np.count_nonzero(specific_matched_preds_mask), "mismatch"
        num_gt_objs = len(specific_det_gts_mask)
        assert num_gt_objs == np.count_nonzero(
            ~ignore_gt_and_matched_preds_where_true_mask
        )
        # fps:
        num_fps = np.count_nonzero(~specific_matched_preds_mask)
        labels_for_fps = np.zeros(num_fps, dtype=bool)
        scores_for_fps = specific_pred_logits[~specific_matched_preds_mask]
        assert labels_for_fps.shape == scores_for_fps.shape
        self.per_class_per_thresh_per_category_gt_labels[class_name][
            matching_threshold
        ][category].append(labels_for_fps)
        self.per_class_per_thresh_per_category_scores[class_name][matching_threshold][
            category
        ].append(scores_for_fps)
        self.per_class_per_thresh_per_category_is_fn[class_name][matching_threshold][
            category
        ].append(np.zeros_like(labels_for_fps))

        # fns:
        num_fns = num_gt_objs - num_tps
        labels_for_fns = np.ones(num_fns, dtype=bool)
        scores_for_fns = -np.inf * np.ones(num_fns)
        assert labels_for_fns.shape == scores_for_fns.shape
        self.per_class_per_thresh_per_category_gt_labels[class_name][
            matching_threshold
        ][category].append(labels_for_fns)
        self.per_class_per_thresh_per_category_scores[class_name][matching_threshold][
            category
        ].append(scores_for_fns)
        self.per_class_per_thresh_per_category_is_fn[class_name][matching_threshold][
            category
        ].append(np.ones_like(labels_for_fns))

        if num_tps > 0:
            # tps
            labels_for_tps = np.ones(num_tps, dtype=bool)
            # get the scores from the matched predictions that are not ignored
            scores_for_tps = pred_logits[
                idxs_into_preds[~ignore_gt_where_true]
            ]  # BUG? should we use specific_pred_logits here?
            # okay it's probably not a bug:
            assert np.all(
                np.sort(specific_pred_logits[specific_matched_preds_mask])
                == np.sort(scores_for_tps)
            )
            assert labels_for_tps.shape == scores_for_tps.shape
            self.per_class_per_thresh_per_category_gt_labels[class_name][
                matching_threshold
            ][category].append(labels_for_tps)
            self.per_class_per_thresh_per_category_scores[class_name][
                matching_threshold
            ][category].append(scores_for_tps)
            self.per_class_per_thresh_per_category_is_fn[class_name][
                matching_threshold
            ][category].append(np.zeros_like(labels_for_tps))

    def log_roc_curves(
        self,
        all_gt: Dict[float, np.ndarray],
        all_scores: Dict[float, np.ndarray],
        global_step: int,
        metrics_dict: Dict[str, float],
        summary_writer=None,
        writer_prefix: str = "",
        path: str = None,
    ):
        # ROC_CURVE
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        plt.plot([0, 1], [0, 1], lw=self.line_width, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (fpr)")
        plt.ylabel("True Positive Rate (tpr)")
        plt.title("Receiver operating characteristic")
        per_thresh_fprs = {}
        tprs = {}
        conf_threshs = {}
        for matching_threshold in self.matching_thresholds:
            relevant_scores = map_scores_from_neg_infs_to_actual_min_score(
                all_scores[matching_threshold]
            )
            if relevant_scores.size == 0:
                continue
            fpr, tpr, confidence_thresholds_decreasing = roc_curve(
                all_gt[matching_threshold],
                relevant_scores,
            )
            if len(np.unique(all_gt[matching_threshold])) <= 1:
                area = 0.0  # not defined
            else:
                area = roc_auc_score(
                    all_gt[matching_threshold],
                    relevant_scores,
                )
            summary_label = (
                writer_prefix.rstrip("/")
                + f"/area_under_roc_curve@{matching_threshold:.1f}{self.threshold_unit}"
            )
            metrics_dict[summary_label] = area
            if summary_writer:
                summary_writer.add_scalar(summary_label, area, global_step)

            plt.plot(
                fpr[:-1],
                tpr[:-1],
                lw=self.line_width,
                label=f"Matching threshold: {matching_threshold:.1f}{self.threshold_unit} - ROC curve (area = {area:.4f})",
            )
            per_thresh_fprs[matching_threshold] = fpr[:-1]
            tprs[matching_threshold] = tpr[:-1]
            conf_threshs[matching_threshold] = confidence_thresholds_decreasing[:-1]

        plt.legend(loc="lower right", prop={"size": self.legend_font_size})
        if path:
            plt.savefig(Path(path).joinpath(f"ROC_curve_{global_step}.png"))
        fig = plt.figure(1)
        if summary_writer:
            summary_writer.add_image(
                writer_prefix + "/ROC_curve",
                plot_to_np_image(fig),
                global_step,
                dataformats="HWC",
            )
            summary_writer.flush()
        plt.close(fig)

        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        plt.ylim([0.0, 1.05])
        plt.xlabel("confidence threshold")
        plt.ylabel("tpr / fpr")
        plt.title("tpr/fpr at confidence thresholds")
        for matching_threshold in self.matching_thresholds:
            if matching_threshold not in conf_threshs:
                continue
            plt.plot(
                conf_threshs[matching_threshold],
                tprs[matching_threshold],
                lw=self.line_width,
                label=f"tpr@matching_threshold {matching_threshold:.1f}{self.threshold_unit}",
            )
            plt.plot(
                conf_threshs[matching_threshold],
                per_thresh_fprs[matching_threshold],
                lw=self.line_width,
                label=f"fpr@matching_threshold {matching_threshold:.1f}{self.threshold_unit}",
            )
        plt.legend(loc="lower right", prop={"size": self.legend_font_size})
        if path:
            plt.savefig(
                Path(path).joinpath(f"fpr_tpr_threshold_curve_{global_step}.png")
            )
        if summary_writer:
            fig = plt.figure(1)
            summary_writer.add_image(
                writer_prefix + "/tpr_fpr_threshold_curve",
                plot_to_np_image(fig),
                global_step,
                dataformats="HWC",
            )
            summary_writer.flush()

        plt.close(fig)

    def log_pr_curve_for_each_threshold(
        self,
        all_gt: np.ndarray,
        all_scores: np.ndarray,
        all_is_fn: np.ndarray,
        global_step: int,
        metrics_dict: Dict[str, float],
        summary_writer=None,
        writer_prefix: str = "",
        path: str = None,
        save_to_pdf=False,
    ):
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        if not save_to_pdf:
            plt.title("PR Curve")
        ax = plt.subplot()
        ax.axvline(x=self.min_recall, linestyle="--", color=(0, 0, 0, 0.3))
        ax.axhline(y=self.min_precision, linestyle="--", color=(0, 0, 0, 0.3))

        precs = {}
        recs = {}
        conf_threshs = {}
        for matching_threshold in self.matching_thresholds:
            confidence_thresholds, precisions, recalls = get_conf_prec_rec(
                all_gt[matching_threshold],
                all_scores[matching_threshold],
                all_is_fn[matching_threshold],
            )

            ap_score = calc_ap(
                precisions, min_recall=self.min_recall, min_precision=self.min_precision
            )

            summary_label = (
                writer_prefix.rstrip("/")
                + f"/AP@{matching_threshold:.1f}{self.threshold_unit}"
            )
            metrics_dict[summary_label] = ap_score
            if summary_writer:
                summary_writer.add_scalar(summary_label, ap_score, global_step)

            plt.plot(
                recalls,
                precisions,
                lw=self.line_width,
                label=f"AP@{matching_threshold:.1f}{self.threshold_unit}:{ap_score:.4f}",
            )
            precs[matching_threshold] = precisions
            recs[matching_threshold] = recalls
            conf_threshs[matching_threshold] = confidence_thresholds
        plt.legend(loc="upper right", prop={"size": self.legend_font_size})
        if path:
            plt.savefig(Path(path).joinpath(f"PR_curve_{global_step}.pdf"))
        fig = plt.figure(1)
        if summary_writer:
            summary_writer.add_image(
                writer_prefix.rstrip("/") + "/PR_curve",
                plot_to_np_image(fig),
                global_step,
                dataformats="HWC",
            )
            summary_writer.flush()

        plt.close(fig)
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        plt.ylim([0.0, 1.05])
        plt.xlabel("confidence threshold")
        plt.ylabel("Precision/Recall")
        if not save_to_pdf:
            plt.title("Precision/Recall @ confidence_thresholds")

        # THRESHOLDS
        for matching_threshold in self.matching_thresholds:
            plt.plot(
                conf_threshs[matching_threshold],
                recs[matching_threshold],
                lw=self.line_width,
                label=f"Recall @ matching threshold {matching_threshold:.1f}{self.threshold_unit}",
            )
            plt.plot(
                conf_threshs[matching_threshold],
                precs[matching_threshold],
                lw=self.line_width,
                label=f"Precision @ matching threshold {matching_threshold:.1f}{self.threshold_unit}",
            )
        plt.legend(loc="lower right", prop={"size": self.legend_font_size})
        if path:
            plt.savefig(
                Path(path).joinpath(
                    f"precision_recall_threshold_curve_{global_step}.png"
                )
            )
        if summary_writer:
            fig = plt.figure(1)
            summary_writer.add_image(
                writer_prefix + "/precision_recall_threshold_curve",
                plot_to_np_image(fig),
                global_step,
                dataformats="HWC",
            )
            summary_writer.flush()
        plt.close(fig)

    def log_sklearn_precision_curve_for_each_threshold(
        self,
        all_gt: np.ndarray,
        all_scores: np.ndarray,
        global_step: int,
        summary_writer=None,
        writer_prefix: str = "",
        path: str = None,
    ):
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        plt.ylim([0.0, 1.05])
        plt.xlabel("confidence threshold")
        plt.ylabel("Precision")
        plt.title("SKLearn Precision @ confidence_thresholds (no interp.)")
        for matching_threshold in self.matching_thresholds:
            if len(np.unique(all_gt[matching_threshold])) <= 1:
                continue
            relevant_scores = map_scores_from_neg_infs_to_actual_min_score(
                all_scores[matching_threshold]
            )
            precisions, _, increasing_thresholds = precision_recall_curve(
                # CAREFUL: thresholds are increasing here!!!
                all_gt[matching_threshold],
                relevant_scores,
                pos_label=1,
            )
            plt.plot(
                increasing_thresholds[1:],
                precisions[1:-1],
                lw=self.line_width,
                label=f"Precision @ matching threshold {matching_threshold:.1f}{self.threshold_unit}",
            )

        plt.legend(loc="upper right", prop={"size": self.legend_font_size})

        if path:
            plt.savefig(
                Path(path).joinpath(
                    f"sklearn_precision_threshold_curve_{global_step}.png"
                )
            )

        fig = plt.figure(1)
        if summary_writer:
            summary_writer.add_image(
                writer_prefix + "/sklearn_precision_threshold_curve",
                plot_to_np_image(fig),
                global_step,
                dataformats="HWC",
            )
            summary_writer.flush()
        plt.close(fig)

    def log_specific_pr_curve(
        self,
        all_gt: np.ndarray,
        all_scores: np.ndarray,
        all_is_fn: np.ndarray,
        global_step: int,
        metrics_dict: Dict[str, float],
        summary_writer=None,
        writer_prefix: str = "",
        path: str = None,
        save_to_pdf=False,
    ):
        precs = {}
        recs = {}
        conf_threshs = {}
        for matching_threshold in self.matching_thresholds:
            plt.figure(figsize=self.figure_size, dpi=self.dpi)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            if not save_to_pdf:
                plt.title("PR Curve")

            for category in self.extra_categories:
                confidence_thresholds, precisions, recalls = get_conf_prec_rec(
                    all_gt[matching_threshold][category],
                    all_scores[matching_threshold][category],
                    all_is_fn[matching_threshold][category],
                )

                ap_score = calc_ap(
                    precisions,
                    min_recall=self.min_recall,
                    min_precision=self.min_precision,
                )
                summary_label = (
                    writer_prefix
                    + f"{category}/AP@{matching_threshold:.1f}{self.threshold_unit}"
                )
                metrics_dict[summary_label] = ap_score
                if summary_writer:
                    summary_writer.add_scalar(summary_label, ap_score, global_step)

                plt.plot(
                    recalls,
                    precisions,
                    lw=self.line_width,
                    label=f"{category}:AP@{matching_threshold:.1f}{self.threshold_unit}:{ap_score:.4f}",
                )
                precs[matching_threshold] = precisions
                recs[matching_threshold] = recalls
                conf_threshs[matching_threshold] = confidence_thresholds
            plt.legend(loc="upper right", prop={"size": self.legend_font_size})
            if path:
                plt.savefig(
                    Path(path).joinpath(
                        f"PR_curve_thresh_{matching_threshold}_{global_step}.png"
                    )
                )
            fig = plt.figure(1)
            if summary_writer:
                summary_writer.add_image(
                    f"{writer_prefix}/PR_curve/{matching_threshold:.1f}{self.threshold_unit}",
                    plot_to_np_image(fig),
                    global_step,
                    dataformats="HWC",
                )
                summary_writer.flush()

            plt.close(fig)
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        plt.ylim([0.0, 1.05])
        plt.xlabel("confidence threshold")
        plt.ylabel("Precision/Recall")
        if not save_to_pdf:
            plt.title("Precision/Recall @ confidence_thresholds")

        # THRESHOLDS
        for matching_threshold in self.matching_thresholds:
            plt.plot(
                conf_threshs[matching_threshold],
                recs[matching_threshold],
                lw=self.line_width,
                label=f"Recall @ matching threshold {matching_threshold:.1f}{self.threshold_unit}",
            )
            plt.plot(
                conf_threshs[matching_threshold],
                precs[matching_threshold],
                lw=self.line_width,
                label=f"Precision @ matching threshold {matching_threshold:.1f}{self.threshold_unit}",
            )
        plt.legend(loc="lower right", prop={"size": self.legend_font_size})
        if path:
            plt.savefig(
                Path(path).joinpath(
                    f"precision_recall_threshold_curve_{global_step}.png"
                )
            )
        if summary_writer:
            fig = plt.figure(1)
            summary_writer.add_image(
                writer_prefix + "/precision_recall_threshold_curve",
                plot_to_np_image(fig),
                global_step,
                dataformats="HWC",
            )
            summary_writer.flush()
        plt.close(fig)

    def log_det_tp_fp_curves(
        self,
        all_gt: Dict[float, np.ndarray],
        all_scores: Dict[float, np.ndarray],
        global_step: int,
        summary_writer=None,
        writer_prefix: str = "",
        path: str = None,
    ):
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (fpr)")
        plt.ylabel("False Negative Rate (fnr)")
        plt.title("Detection Error Tradeoff Curve")
        per_thresh_fp_rates = {}
        per_thresh_fn_rates = {}
        per_thresh_conf_threshs_for_ratios = {}
        per_thresh_abs_num_fps = {}
        per_thresh_abs_num_tps = {}
        per_thresh_conf_threshs_abs = {}
        for matching_threshold in self.matching_thresholds:
            categories_of_matches = np.unique(all_gt[matching_threshold])
            if categories_of_matches.size == 0:
                continue
            relevant_scores = map_scores_from_neg_infs_to_actual_min_score(
                all_scores[matching_threshold]
            )
            if len(categories_of_matches) == 1 and categories_of_matches[0]:
                # only TPs present
                thresholds_for_ratios_decreasing = np.sort(relevant_scores)[::-1]
                fp_rate = np.zeros_like(relevant_scores)
                fn_rate = np.zeros_like(relevant_scores)
                thresholds_for_abs_decreasing = np.copy(
                    thresholds_for_ratios_decreasing
                )
                fps = all_gt[matching_threshold].size * np.zeros_like(
                    thresholds_for_ratios_decreasing
                )
                tps = all_gt[matching_threshold].size * np.ones_like(
                    thresholds_for_ratios_decreasing
                )
            elif len(categories_of_matches) == 1 and not categories_of_matches[0]:
                # only FPs present
                thresholds_for_ratios_decreasing = np.sort(relevant_scores)[::-1]
                fp_rate = np.ones_like(relevant_scores)
                fn_rate = np.ones_like(relevant_scores)
                thresholds_for_abs_decreasing = np.copy(
                    thresholds_for_ratios_decreasing
                )
                fps = all_gt[matching_threshold].size * np.ones_like(
                    thresholds_for_ratios_decreasing
                )
                tps = all_gt[matching_threshold].size * np.zeros_like(
                    thresholds_for_ratios_decreasing
                )
            else:
                # both TPs and FPs present
                fp_rate, fn_rate, thresholds_for_ratios_decreasing = det_curve(
                    all_gt[matching_threshold], relevant_scores
                )
                fps, tps, thresholds_for_abs_decreasing = _binary_clf_curve(
                    all_gt[matching_threshold], relevant_scores
                )
            # 1: assumes we have at least one FN!
            per_thresh_fp_rates[matching_threshold] = fp_rate[:-1]
            per_thresh_fn_rates[matching_threshold] = fn_rate[:-1]
            per_thresh_conf_threshs_for_ratios[
                matching_threshold
            ] = thresholds_for_ratios_decreasing[:-1]
            plt.plot(
                fp_rate,
                fn_rate,
                lw=self.line_width,
                label=f"Detection Error Tradeoff @ matching threshold {matching_threshold:.1f}{self.threshold_unit}",
            )

            per_thresh_abs_num_fps[matching_threshold] = fps[:-1]
            per_thresh_abs_num_tps[matching_threshold] = tps[:-1]
            per_thresh_conf_threshs_abs[
                matching_threshold
            ] = thresholds_for_abs_decreasing[:-1]

        plt.legend(loc="lower right", prop={"size": self.legend_font_size})

        if path:
            plt.savefig(Path(path).joinpath(f"DET_curve_{global_step}.png"))
        fig = plt.figure(1)
        if summary_writer:
            summary_writer.add_image(
                writer_prefix + "/DET_curve",
                plot_to_np_image(fig),
                global_step,
                dataformats="HWC",
            )
            summary_writer.flush()
        plt.close(fig)

        if len(np.unique(all_gt[self.matching_thresholds[-1]])) <= 1:
            print("No negative samples found, fnr makes no sense!")
            return

        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        plt.ylim([0.0, 1.05])
        plt.xlabel("thresholds")
        plt.ylabel("fpr/fnr")
        plt.title("fpr/fnr @ thresholds")
        for matching_threshold in self.matching_thresholds:
            if matching_threshold not in per_thresh_conf_threshs_for_ratios:
                print(
                    f"Skipping TP ratio plots for threshold {matching_threshold} as there are no TPs!"
                )
                continue
            plt.plot(
                per_thresh_conf_threshs_for_ratios[matching_threshold],
                per_thresh_fn_rates[matching_threshold],
                lw=self.line_width,
                label=f"fnr @ matching threshold {matching_threshold:.1f}",
            )
            plt.plot(
                per_thresh_conf_threshs_for_ratios[matching_threshold],
                per_thresh_fp_rates[matching_threshold],
                lw=self.line_width,
                label=f"fpr @ matching threshold {matching_threshold:.1f}",
            )
        plt.legend(loc="lower right", prop={"size": self.legend_font_size})
        if path:
            plt.savefig(
                Path(path).joinpath(f"fnr_fpr_thresholds_curve_{global_step}.png")
            )
        if summary_writer:
            fig = plt.figure(1)
            summary_writer.add_image(
                writer_prefix + "/fnr_fpr_thresholds_curve",
                plot_to_np_image(fig),
                global_step,
                dataformats="HWC",
            )
            summary_writer.flush()

        plt.close(fig)

        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        plt.xlabel("thresholds")
        plt.ylabel("num tps / num fps")
        plt.title("num tps / num fps @ thresholds")
        for matching_threshold in self.matching_thresholds:
            if matching_threshold not in per_thresh_conf_threshs_abs:
                print(
                    f"Skipping TP ratio plots for threshold {matching_threshold} as there are no TPs!"
                )
                continue
            plt.plot(
                per_thresh_conf_threshs_abs[matching_threshold],
                per_thresh_abs_num_fps[matching_threshold],
                lw=self.line_width,
                label=f"num fps @ matching threshold {matching_threshold:.1f}",
            )
            plt.plot(
                per_thresh_conf_threshs_abs[matching_threshold],
                per_thresh_abs_num_tps[matching_threshold],
                lw=self.line_width,
                label=f"num tps @ matching threshold {matching_threshold:.1f}",
            )
        plt.yscale("log")
        plt.legend(loc="best", prop={"size": self.legend_font_size})
        if path:
            plt.savefig(
                Path(path).joinpath(f"tps_fps_thresholds_curve_{global_step}.png")
            )
        if summary_writer:
            fig = plt.figure(1)
            summary_writer.add_image(
                writer_prefix + "/tps_fps_thresholds_curve",
                plot_to_np_image(fig),
                global_step,
                dataformats="HWC",
            )
            summary_writer.flush()

        plt.close(fig)

    def log(
        self,
        global_step: int,
        summary_writer=None,
        writer_prefix: str = "",
        path=None,
        save_to_pdf=False,
    ):
        assert path or summary_writer, "need at least either path or summary_writer"
        writer_prefix_with_crit = (
            writer_prefix.rstrip("/") + f"/{self.box_matching_criterion}/"
        )
        metrics_dict = {}

        all_per_class_specific_gt = {}
        all_per_class_specific_scores = {}
        all_per_class_specific_is_fn = {}
        all_per_class_specific_padding_scores = {}
        for class_name in self.class_names:
            all_per_class_specific_gt[class_name] = {}
            all_per_class_specific_scores[class_name] = {}
            all_per_class_specific_is_fn[class_name] = {}
            all_per_class_specific_padding_scores[class_name] = {}
            for matching_thresh in self.matching_thresholds:
                all_per_class_specific_gt[class_name][matching_thresh] = {}
                all_per_class_specific_scores[class_name][matching_thresh] = {}
                all_per_class_specific_is_fn[class_name][matching_thresh] = {}
                all_per_class_specific_padding_scores[class_name][matching_thresh] = {}

                for category in self.extra_categories:
                    all_per_class_specific_gt[class_name][matching_thresh][
                        category
                    ] = np.concatenate(
                        self.per_class_per_thresh_per_category_gt_labels[class_name][
                            matching_thresh
                        ][category]
                    )
                    all_per_class_specific_scores[class_name][matching_thresh][
                        category
                    ] = np.concatenate(
                        self.per_class_per_thresh_per_category_scores[class_name][
                            matching_thresh
                        ][category]
                    )
                    all_per_class_specific_is_fn[class_name][matching_thresh][
                        category
                    ] = np.concatenate(
                        self.per_class_per_thresh_per_category_is_fn[class_name][
                            matching_thresh
                        ][category]
                    )
                    assert (
                        all_per_class_specific_gt[class_name][matching_thresh][
                            category
                        ].shape
                        == all_per_class_specific_scores[class_name][matching_thresh][
                            category
                        ].shape
                    )
                    assert (
                        all_per_class_specific_gt[class_name][matching_thresh][
                            category
                        ].shape
                        == all_per_class_specific_is_fn[class_name][matching_thresh][
                            category
                        ].shape
                    )

        for class_name in self.class_names:
            writer_prefix_w_crit_and_class_name = (
                writer_prefix_with_crit.rstrip("/") + "/" + class_name + "/"
            )
            self.log_specific_pr_curve(
                all_per_class_specific_gt[class_name],
                all_per_class_specific_scores[class_name],
                all_per_class_specific_is_fn[class_name],
                global_step=global_step,
                summary_writer=summary_writer,
                writer_prefix=writer_prefix_w_crit_and_class_name,
                path=path,
                metrics_dict=metrics_dict,
                save_to_pdf=save_to_pdf,
            )
            self.log_recall_recall_curves(
                all_gt=all_per_class_specific_gt[class_name],
                all_scores=all_per_class_specific_scores[class_name],
                all_is_fn=all_per_class_specific_is_fn[class_name],
                per_thresh_num_instances=self.per_class_per_thresh_label_stats[
                    class_name
                ],
                global_step=global_step,
                summary_writer=summary_writer,
                writer_prefix=writer_prefix_w_crit_and_class_name,
                path=path,
            )

            overall_gts = {
                thresh: el["overall"]
                for thresh, el in all_per_class_specific_gt[class_name].items()
            }
            overall_scores = {
                thresh: el["overall"]
                for thresh, el in all_per_class_specific_scores[class_name].items()
            }
            overall_is_fn = {
                thresh: el["overall"]
                for thresh, el in all_per_class_specific_is_fn[class_name].items()
            }

            self.log_roc_curves(
                all_gt=overall_gts,
                all_scores=overall_scores,
                global_step=global_step,
                summary_writer=summary_writer,
                writer_prefix=writer_prefix_w_crit_and_class_name,
                path=path,
                metrics_dict=metrics_dict,
            )

            self.log_pr_curve_for_each_threshold(
                all_gt=overall_gts,
                all_scores=overall_scores,
                all_is_fn=overall_is_fn,
                global_step=global_step,
                summary_writer=summary_writer,
                writer_prefix=writer_prefix_w_crit_and_class_name,
                path=path,
                metrics_dict=metrics_dict,
                save_to_pdf=save_to_pdf,
            )
            self.log_sklearn_precision_curve_for_each_threshold(
                all_gt=overall_gts,
                all_scores=overall_scores,
                global_step=global_step,
                summary_writer=summary_writer,
                writer_prefix=writer_prefix_w_crit_and_class_name,
                path=path,
            )
            try:
                self.log_det_tp_fp_curves(
                    all_gt=overall_gts,
                    all_scores=overall_scores,
                    global_step=global_step,
                    summary_writer=summary_writer,
                    writer_prefix=writer_prefix_w_crit_and_class_name + "overall",
                    path=path,
                )
            except Exception as e:
                print(
                    "Could not log DET curves for class",
                    class_name,
                    "due to insufficient data",
                )
                print(e)
            if summary_writer:
                for metric_name, metric_value in metrics_dict.items():
                    if "NDS" in metric_name or "mAP" in metric_name:
                        summary_writer.add_scalar(
                            metric_name,
                            metric_value,
                            global_step,
                        )
                for (
                    matching_thresh,
                    label_stats,
                ) in self.per_class_per_thresh_label_stats[class_name].items():
                    for category, num_objs in label_stats.items():
                        summary_writer.add_scalar(
                            writer_prefix_w_crit_and_class_name
                            + f"/{category}/{matching_thresh:.1f}{self.threshold_unit}/num_objs",
                            num_objs,
                            global_step,
                        )

                    for (
                        tp_metr_cat,
                        tp_metr_value,
                    ) in self.per_class_per_thresh_tp_errors_running_stats[class_name][
                        matching_thresh
                    ].items():
                        if tp_metr_cat == "tps":
                            plot_val = tp_metr_value
                        else:
                            plot_val = tp_metr_value / max(
                                self.per_class_per_thresh_tp_errors_running_stats[
                                    class_name
                                ][matching_thresh]["tps"],
                                1e-6,
                            )
                        summary_writer.add_scalar(
                            writer_prefix_w_crit_and_class_name
                            + f"/{matching_thresh:.1f}{self.threshold_unit}/{tp_metr_cat}",
                            plot_val,
                            global_step,
                        )

                if summary_writer:
                    summary_writer.add_text(
                        writer_prefix_w_crit_and_class_name,
                        pretty_json(metrics_dict),
                        global_step,
                    )
        print(f"{global_step}/{writer_prefix_with_crit}:", metrics_dict)

        plt.close("all")
        gc.collect()
        return metrics_dict

    def log_recall_recall_curves(
        self,
        all_gt: Dict[float, Dict[str, np.ndarray]],
        all_scores: Dict[float, Dict[str, np.ndarray]],
        all_is_fn: Dict[float, Dict[str, np.ndarray]],
        per_thresh_num_instances: Dict[float, Dict[str, np.ndarray]],
        global_step: int,
        summary_writer=None,
        writer_prefix: str = "",
        path: str = None,
    ):
        for matching_threshold in self.matching_thresholds:
            plt.figure(figsize=self.figure_size, dpi=self.dpi)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("Recall[overall]")
            plt.ylabel("Recall[specific]")

            confidence_thresholds_overall, _, recalls_overall = get_conf_prec_rec(
                all_gt[matching_threshold]["overall"],
                all_scores[matching_threshold]["overall"],
                all_is_fn[matching_threshold]["overall"],
                use_interpolation=False,
            )

            plt.plot(
                recalls_overall,
                recalls_overall,
                lw=self.line_width,
                label=f"Recall[overall]@{matching_threshold:.1f}m\n#Instances: {per_thresh_num_instances[matching_threshold]['overall']}",
            )

            for category in ("moving", "still"):
                (
                    confidence_thresholds_for_category,
                    _,
                    recalls_for_category,
                ) = get_conf_prec_rec(
                    all_gt[matching_threshold][category],
                    all_scores[matching_threshold][category],
                    all_is_fn[matching_threshold][category],
                    use_interpolation=False,
                )
                if (
                    recalls_for_category.size == 0
                    or confidence_thresholds_for_category.size == 0
                ):
                    print(
                        f"Skipping Recall[{category}]@{matching_threshold:.1f}{self.threshold_unit} - no samples?"
                    )
                    continue
                category_recalls_at_overall_thresholds = np.interp(
                    confidence_thresholds_overall,
                    confidence_thresholds_for_category,
                    recalls_for_category,
                )

                plt.plot(
                    recalls_overall,
                    category_recalls_at_overall_thresholds,
                    lw=self.line_width,
                    label=f"Recall[{category}]@{matching_threshold:.1f}{self.threshold_unit}\n#Instances: {per_thresh_num_instances[matching_threshold][category]}",
                )
            plt.legend(loc="upper right", prop={"size": self.legend_font_size})
            if path:
                plt.savefig(
                    Path(path).joinpath(
                        f"Recall_Recall_curve_thresh_{matching_threshold}_{global_step}.pdf"
                    )
                )
            fig = plt.figure(1)
            if summary_writer:
                summary_writer.add_image(
                    f"{writer_prefix}/Recall_Recall_curve/{matching_threshold:.1f}{self.threshold_unit}",
                    plot_to_np_image(fig),
                    global_step,
                    dataformats="HWC",
                )
                summary_writer.flush()

            plt.close(fig)


class WaymoObjectDetectionMetrics(BaseODMetrics):
    def __init__(
        self,
        eval_movable_classes_as_one: bool = True,
        bev_range_min_xy_m=(-50, -20),
        bev_range_max_xy_m=(50, 20),
        class_names: Tuple[str] = ("overall",),
        class_idxs: Tuple[int] = (0,),
        min_eval_range_m: float = None,
        max_eval_range_m: float = None,
    ):
        super().__init__()
        self.min_eval_range_m = min_eval_range_m
        self.max_eval_range_m = max_eval_range_m
        self.eval_movable_classes_as_one = eval_movable_classes_as_one
        if class_names == ("overall",):
            class_idxs = (0,)
        assert len(class_names) == len(class_idxs), (class_names, class_idxs)
        self.class_idxs = class_idxs
        self.class_names = class_names
        self.box_matching_criterions = ("iou_3d", "iou_bev")
        self.extra_categories = ("L1", "L2")
        self.iou_matching_threshold = 0.4
        self.per_class_per_crit_per_category_gt_labels = {}
        self.per_class_per_crit_per_category_scores = {}
        self.per_class_per_crit_per_category_is_fn = {}

        self.bev_range_min_xy_m = torch.tensor(
            bev_range_min_xy_m,
        )
        self.bev_range_max_xy_m = torch.tensor(
            bev_range_max_xy_m,
        )
        self.report_prec_recall_at_confidences = [
            # 0.0,
            # 0.1,
            # 0.2,
            0.3,
            # 0.4,
            # 0.5,
            # 0.9,
            1.0,
        ]
        for class_name in self.class_names:
            self.per_class_per_crit_per_category_gt_labels[class_name] = {}
            self.per_class_per_crit_per_category_scores[class_name] = {}
            self.per_class_per_crit_per_category_is_fn[class_name] = {}
            for crit in self.box_matching_criterions:
                self.per_class_per_crit_per_category_gt_labels[class_name][crit] = {}
                self.per_class_per_crit_per_category_scores[class_name][crit] = {}
                self.per_class_per_crit_per_category_is_fn[class_name][crit] = {}
                for difficulty in self.extra_categories:
                    self.per_class_per_crit_per_category_gt_labels[class_name][crit][
                        difficulty
                    ] = []
                    self.per_class_per_crit_per_category_scores[class_name][crit][
                        difficulty
                    ] = []
                    self.per_class_per_crit_per_category_is_fn[class_name][crit][
                        difficulty
                    ] = []

    def update(
        self,
        *,
        non_batched_gt_boxes: Shape,
        non_batched_pred_boxes: Shape,
        sample_token: str,  # unused
    ):
        # filter to waymo range from paper
        filtered_non_batched_gt_boxes = self.filter_boxes_to_be_in_bev_range(
            non_batched_gt_boxes.clone(),
        )
        filtered_non_batched_pred_boxes = self.filter_boxes_to_be_in_bev_range(
            non_batched_pred_boxes.clone(),
        )

        if self.max_eval_range_m is not None and self.min_eval_range_m is not None:
            filtered_non_batched_gt_boxes = self.filter_boxes_to_be_in_abs_range(
                filtered_non_batched_gt_boxes
            )
            filtered_non_batched_pred_boxes = self.filter_boxes_to_be_in_abs_range(
                filtered_non_batched_pred_boxes
            )

        assert len(self.class_idxs) == len(self.class_names), (
            self.class_idxs,
            self.class_names,
        )
        for class_idx, class_name in zip(self.class_idxs, self.class_names):
            if class_name == "overall":
                class_specific_gt_boxes = filtered_non_batched_gt_boxes
                class_specific_pred_boxes = filtered_non_batched_pred_boxes
            else:
                class_specific_gt_boxes = self.filter_boxes_to_be_of_class(
                    filtered_non_batched_gt_boxes.clone(), class_idx=class_idx
                )
                class_specific_pred_boxes = self.filter_boxes_to_be_of_class(
                    filtered_non_batched_pred_boxes.clone(), class_idx=class_idx
                )
            for matching_crit in self.box_matching_criterions:
                self.update_for_specific_class_and_matching_crit(
                    non_batched_gt_boxes=class_specific_gt_boxes,
                    non_batched_pred_boxes=class_specific_pred_boxes,
                    matching_crit=matching_crit,
                    class_name=class_name,
                )

    def update_for_specific_class_and_matching_crit(
        self,
        *,
        non_batched_gt_boxes: Shape,
        non_batched_pred_boxes: Shape,
        matching_crit: str,
        class_name: str,
    ):
        (
            idxs_into_gt,
            idxs_into_preds,
            _,
            matched_preds_mask,
            det_gts_mask,
        ) = match_boxes_by_descending_confidence_iou(
            non_batched_gt_boxes,
            non_batched_pred_boxes,
            matching_threshold=self.iou_matching_threshold,
            iou_mode=matching_crit,
            matching_mode="hungarian",
        )

        np_non_batched_gt_boxes = non_batched_gt_boxes.numpy()
        np_non_batched_pred_boxes = non_batched_pred_boxes.numpy()
        pred_logits = np.squeeze(np_non_batched_pred_boxes.probs, axis=-1)

        gt_box_is_hard = np.squeeze(np_non_batched_gt_boxes.difficulty, axis=-1) > 0

        assert det_gts_mask.shape == gt_box_is_hard.shape, (
            det_gts_mask.shape,
            gt_box_is_hard.shape,
        )

        self.update_specific_metrics(
            det_gts_mask=det_gts_mask,
            matched_preds_mask=matched_preds_mask,
            pred_logits=pred_logits,
            idxs_into_gt=idxs_into_gt,
            idxs_into_preds=idxs_into_preds,
            ignore_gt_and_matched_preds_where_true_mask=np.zeros_like(
                gt_box_is_hard,
            ),
            matching_threshold=self.iou_matching_threshold,
            category="L2",
            matching_crit=matching_crit,
            class_name=class_name,
        )
        self.update_specific_metrics(
            det_gts_mask=det_gts_mask,
            matched_preds_mask=matched_preds_mask,
            pred_logits=pred_logits,
            idxs_into_preds=idxs_into_preds,
            idxs_into_gt=idxs_into_gt,
            ignore_gt_and_matched_preds_where_true_mask=gt_box_is_hard,
            matching_threshold=self.iou_matching_threshold,
            category="L1",
            matching_crit=matching_crit,
            class_name=class_name,
        )

    def update_specific_metrics(
        self,
        *,
        det_gts_mask: np.ndarray,
        matched_preds_mask: np.ndarray,
        pred_logits: np.ndarray,
        idxs_into_preds: np.ndarray,
        idxs_into_gt: np.ndarray,
        ignore_gt_and_matched_preds_where_true_mask: np.ndarray,
        matching_threshold: float,
        category: str,
        matching_crit: str,
        class_name: str,
    ):
        assert category in self.extra_categories, category
        # transfer ignore mask over to matched predictions
        ignore_gt_where_true = ignore_gt_and_matched_preds_where_true_mask[idxs_into_gt]
        # use transfered label and idx to to create mask for prediction
        use_this_prediction = np.ones_like(matched_preds_mask)
        use_this_prediction[idxs_into_preds] = ~ignore_gt_where_true

        if category == "L2":
            # must use all, ignore nothing
            assert np.all(
                ~ignore_gt_and_matched_preds_where_true_mask
            ), ignore_gt_and_matched_preds_where_true_mask
            assert np.all(use_this_prediction), use_this_prediction

        # disable predictions that match to ignored gt boxes
        # use all detections by default! do not change defualt, will loose FPs
        specific_pred_logits = pred_logits[use_this_prediction]
        specific_matched_preds_mask = matched_preds_mask[use_this_prediction]
        specific_det_gts_mask = det_gts_mask[
            ~ignore_gt_and_matched_preds_where_true_mask
        ]
        if category == "L2":
            # must use all, ignore nothing
            assert np.all(specific_det_gts_mask == det_gts_mask), (
                specific_det_gts_mask,
                det_gts_mask,
            )

        num_tps = np.count_nonzero(specific_det_gts_mask)
        assert num_tps == np.count_nonzero(specific_matched_preds_mask), "mismatch"
        num_gt_objs = len(specific_det_gts_mask)
        assert num_gt_objs == np.count_nonzero(
            ~ignore_gt_and_matched_preds_where_true_mask
        )
        # fps:
        num_fps = np.count_nonzero(~specific_matched_preds_mask)
        labels_for_fps = np.zeros(num_fps, dtype=bool)
        scores_for_fps = specific_pred_logits[~specific_matched_preds_mask]
        assert labels_for_fps.shape == scores_for_fps.shape
        self.per_class_per_crit_per_category_gt_labels[class_name][matching_crit][
            category
        ].append(labels_for_fps)
        self.per_class_per_crit_per_category_scores[class_name][matching_crit][
            category
        ].append(scores_for_fps)
        self.per_class_per_crit_per_category_is_fn[class_name][matching_crit][
            category
        ].append(np.zeros_like(labels_for_fps))

        # fns:
        num_fns = num_gt_objs - num_tps
        labels_for_fns = np.ones(num_fns, dtype=bool)
        scores_for_fns = -np.inf * np.ones(num_fns)
        assert labels_for_fns.shape == scores_for_fns.shape
        self.per_class_per_crit_per_category_gt_labels[class_name][matching_crit][
            category
        ].append(labels_for_fns)
        self.per_class_per_crit_per_category_scores[class_name][matching_crit][
            category
        ].append(scores_for_fns)
        self.per_class_per_crit_per_category_is_fn[class_name][matching_crit][
            category
        ].append(np.ones_like(labels_for_fns))

        if num_tps > 0:
            # tps
            labels_for_tps = np.ones(num_tps, dtype=bool)
            # get the scores from the matched predictions that are not ignored
            scores_for_tps = pred_logits[
                idxs_into_preds[~ignore_gt_where_true]
            ]  # BUG? should we use specific_pred_logits here?
            # okay it's probably not a bug:
            assert np.all(
                np.sort(specific_pred_logits[specific_matched_preds_mask])
                == np.sort(scores_for_tps)
            )
            assert labels_for_tps.shape == scores_for_tps.shape
            self.per_class_per_crit_per_category_gt_labels[class_name][matching_crit][
                category
            ].append(labels_for_tps)
            self.per_class_per_crit_per_category_scores[class_name][matching_crit][
                category
            ].append(scores_for_tps)
            self.per_class_per_crit_per_category_is_fn[class_name][matching_crit][
                category
            ].append(np.zeros_like(labels_for_tps))

    def log(
        self,
        global_step: int,
        summary_writer=None,
        writer_prefix: str = "",
        path=None,
    ):
        assert path or summary_writer, "need at least either path or summary_writer"
        metrics_dict = {}

        all_per_class_specific_gt = {}
        all_per_class_specific_scores = {}
        all_per_class_specific_is_fn = {}
        all_per_class_specific_padding_scores = {}
        for class_name in self.class_names:
            all_per_class_specific_gt[class_name] = {}
            all_per_class_specific_scores[class_name] = {}
            all_per_class_specific_is_fn[class_name] = {}
            all_per_class_specific_padding_scores[class_name] = {}
            for matching_crit in self.box_matching_criterions:
                all_per_class_specific_gt[class_name][matching_crit] = {}
                all_per_class_specific_scores[class_name][matching_crit] = {}
                all_per_class_specific_is_fn[class_name][matching_crit] = {}
                all_per_class_specific_padding_scores[class_name][matching_crit] = {}

                for category in self.extra_categories:
                    all_per_class_specific_gt[class_name][matching_crit][
                        category
                    ] = np.concatenate(
                        self.per_class_per_crit_per_category_gt_labels[class_name][
                            matching_crit
                        ][category]
                    )
                    all_per_class_specific_scores[class_name][matching_crit][
                        category
                    ] = np.concatenate(
                        self.per_class_per_crit_per_category_scores[class_name][
                            matching_crit
                        ][category]
                    )
                    all_per_class_specific_is_fn[class_name][matching_crit][
                        category
                    ] = np.concatenate(
                        self.per_class_per_crit_per_category_is_fn[class_name][
                            matching_crit
                        ][category]
                    )
                    assert (
                        all_per_class_specific_gt[class_name][matching_crit][
                            category
                        ].shape
                        == all_per_class_specific_scores[class_name][matching_crit][
                            category
                        ].shape
                    )
                    assert (
                        all_per_class_specific_gt[class_name][matching_crit][
                            category
                        ].shape
                        == all_per_class_specific_is_fn[class_name][matching_crit][
                            category
                        ].shape
                    )

        self.log_specific_pr_curve(
            all_per_class_specific_gt,
            all_per_class_specific_scores,
            all_per_class_specific_is_fn,
            global_step=global_step,
            summary_writer=summary_writer,
            writer_prefix=writer_prefix,
            path=path,
            metrics_dict=metrics_dict,
        )

        print(f"{global_step}/{writer_prefix}:", metrics_dict)

        return metrics_dict

    def log_specific_pr_curve(
        self,
        all_gt: np.ndarray,
        all_scores: np.ndarray,
        all_is_fn: np.ndarray,
        global_step: int,
        metrics_dict: Dict[str, float],
        summary_writer=None,
        writer_prefix: str = "",
        path: str = None,
    ):
        for class_name in self.class_names:
            for matching_crit in self.box_matching_criterions:
                plt.figure(figsize=self.figure_size, dpi=self.dpi)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title("PR Curve")

                for category in self.extra_categories:
                    confidences, precisions, recalls = get_conf_prec_rec(
                        all_gt[class_name][matching_crit][category],
                        all_scores[class_name][matching_crit][category],
                        all_is_fn[class_name][matching_crit][category],
                        use_interpolation=False,
                    )
                    for conf_thresh in self.report_prec_recall_at_confidences:
                        if len(confidences) == 0:
                            continue
                        closest_conf_idx = np.abs(confidences - conf_thresh).argmin()
                        prec_at_thresh = precisions[closest_conf_idx]
                        recall_at_thresh = recalls[closest_conf_idx]
                        pr_at_conf_thresh_label = (
                            writer_prefix
                            + f"{matching_crit}/{class_name}/{category}/Precision@{self.iou_matching_threshold:.1f}/@conf{conf_thresh:.1f}"
                        )
                        rec_at_conf_thresh_label = (
                            writer_prefix
                            + f"{matching_crit}/{class_name}/{category}/Recall@{self.iou_matching_threshold:.1f}/@conf{conf_thresh:.1f}"
                        )
                        metrics_dict[pr_at_conf_thresh_label] = prec_at_thresh
                        metrics_dict[rec_at_conf_thresh_label] = recall_at_thresh
                        if summary_writer:
                            summary_writer.add_scalar(
                                rec_at_conf_thresh_label, recall_at_thresh, global_step
                            )
                            summary_writer.add_scalar(
                                pr_at_conf_thresh_label, prec_at_thresh, global_step
                            )

                    ap_summary_label = (
                        writer_prefix
                        + f"{matching_crit}/{class_name}/{category}/AP@{self.iou_matching_threshold:.1f}"
                    )
                    _, _, ap_score = waymo_precisions_recalls_apscore(
                        precisions, recalls
                    )
                    metrics_dict[ap_summary_label] = ap_score
                    if summary_writer:
                        summary_writer.add_scalar(
                            ap_summary_label, ap_score, global_step
                        )

                    plt.plot(
                        recalls,
                        precisions,
                        lw=self.line_width,
                        label=f"{matching_crit}/{class_name}/{category}:AP@{self.iou_matching_threshold:.1f}:{ap_score:.4f}",
                    )
                plt.legend(loc="upper right", prop={"size": self.legend_font_size})
                if path:
                    plt.savefig(
                        Path(path).joinpath(
                            f"PR_curve_thresh_{matching_crit}/{class_name}_{self.iou_matching_threshold}_{global_step}.pdf"
                        )
                    )
                fig = plt.figure(1)
                if summary_writer:
                    summary_writer.add_image(
                        f"{writer_prefix}/{matching_crit}/{class_name}/PR_curve/{self.iou_matching_threshold:.1f}",
                        plot_to_np_image(fig),
                        global_step,
                        dataformats="HWC",
                    )
                    summary_writer.flush()

                plt.close(fig)
                summary_writer.flush()
                plt.close(fig)


def map_scores_from_neg_infs_to_actual_min_score(relevant_scores):
    num_valid_scores = np.count_nonzero(relevant_scores != -np.inf)
    if num_valid_scores == 0:
        print("Warning, no valid scores found: Replacing with magic number -666")
        replace_inf_with = -666
    else:
        min_score_not_inf = np.min(
            relevant_scores[relevant_scores != -np.inf],
        )
        max_score_not_inf = np.max(
            relevant_scores[relevant_scores != -np.inf],
        )
        replace_inf_with = min_score_not_inf - 0.1 * (
            # use slight offset between last non FN prediction and all FN predictions
            max_score_not_inf
            - min_score_not_inf
        )
    clean_scores = np.where(
        relevant_scores == -np.inf,
        replace_inf_with,
        relevant_scores,
    )
    return clean_scores


def waymo_precisions_recalls_apscore(precisions, recalls, max_recall_gap=0.05):
    # Paper:
    # If the gap between recall
    # values of two consecutive operating points on the PR curve
    # is larger than a preset threshold (set to 0.05), more p/r points
    # are explicitly inserted between with conservative precisions.
    # Example: p(r) : p(0) = 1.0, p(1) = 0.0, δ = 0.05. We
    # add p(0.95) = 0.0, p(0.90) = 0.0, ..., p(0.05) = 0.0.
    # The mAP = 0.05 after this augmentation.
    # recalls_orig = np.copy(recalls)
    # precisions_orig = np.copy(precisions)
    eps = 1e-6
    have_gap = (np.abs(np.diff(recalls)) - eps) > max_recall_gap
    max_num_tries = 1000
    while np.any(have_gap) and max_num_tries > 0:
        max_num_tries -= 1
        assert len(precisions) == len(recalls), (len(precisions), len(recalls))
        gap_location = np.where(have_gap)[0][0]
        precision_value_after_gap = precisions[gap_location + 1]
        recall_value_after_gap = recalls[gap_location + 1]
        recall_value_before_gap = recalls[gap_location]
        recall_gap_size = recall_value_after_gap - recalls[gap_location]
        assert np.all(recall_gap_size > 0.0), recall_gap_size
        num_points_to_insert = (recall_gap_size / max_recall_gap).astype(int) - 1
        insertion_loc_start = np.repeat(gap_location + 1, num_points_to_insert)
        recall_values_to_insert = np.linspace(
            start=recall_value_before_gap + max_recall_gap,
            stop=recall_value_after_gap - max_recall_gap,
            num=num_points_to_insert,
        )

        precision_values_to_insert = np.repeat(
            precision_value_after_gap, num_points_to_insert
        )

        precisions = np.insert(
            precisions, insertion_loc_start, precision_values_to_insert
        )
        recalls = np.insert(recalls, insertion_loc_start, recall_values_to_insert)

        have_gap = (np.abs(np.diff(recalls)) - eps) > max_recall_gap

    ap_score_trapz = np.trapz(precisions, recalls)

    return precisions, recalls, ap_score_trapz


def main():
    gt_rots = -np.pi + 2 * np.pi * np.random.rand(1000000)
    pred_rots = -np.pi + 2 * np.pi * np.random.rand(1000000)
    average_orientation_error = abs_yaw_diff(
        gt_rots, pred_rots, period=2 * np.pi
    ).mean()
    print(average_orientation_error)


if __name__ == "__main__":
    main()
