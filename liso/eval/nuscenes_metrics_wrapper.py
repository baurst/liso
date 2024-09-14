import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from liso.datasets.nuscenes_torch_dataset import NuscenesDataset
from liso.kabsch.shape_utils import Shape
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.render import setup_axis
from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import (
    DetectionBox,
    DetectionMetricDataList,
    DetectionMetrics,
)
from nuscenes.eval.detection.render import class_pr_curve
from pytorch3d.transforms import matrix_to_quaternion
from torch.utils.tensorboard import SummaryWriter


class DetectionConfig:
    def __init__(
        self,
        class_range: Dict[str, int],
        dist_fcn: str,
        dist_ths: List[float],
        dist_th_tp: float,
        min_recall: float,
        min_precision: float,
        max_boxes_per_sample: int,
        mean_ap_weight: int,
    ):
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight

        self.class_names = self.class_range.keys()

    @property
    def dist_fcn_callable(self):
        if self.dist_fcn == "center_distance":
            return center_distance
        else:
            raise Exception("Error: Unknown distance function %s!" % self.dist_fcn)

    def serialize(self) -> dict:
        return {
            "class_range": self.class_range,
            "dist_fcn": self.dist_fcn,
            "dist_ths": self.dist_ths,
            "dist_th_tp": self.dist_th_tp,
            "min_recall": self.min_recall,
            "min_precision": self.min_precision,
            "max_boxes_per_sample": self.max_boxes_per_sample,
            "mean_ap_weight": self.mean_ap_weight,
        }


class NuscenesObjectDetectionMetrics:
    def __init__(
        self,
        eval_movable_classes_as_one=True,
    ) -> None:
        self.eval_movable_classes_as_one = eval_movable_classes_as_one
        dist_matching_thresholds = (0.5, 1.0, 2.0, 4.0)
        self.tp_metric_thresh = 2.0
        self.threshold_unit = "m"
        self.matching_thresholds = dist_matching_thresholds
        self.gt_boxes = EvalBoxes()
        self.pred_boxes = EvalBoxes()
        if eval_movable_classes_as_one:
            class_ranges = {
                "movable": 50,
            }
        else:
            class_ranges = {
                "car": 50,
                "truck": 50,
                "bus": 50,
                "trailer": 50,
                "construction_vehicle": 50,
                "pedestrian": 40,
                "motorcycle": 40,
                "bicycle": 40,
                # "traffic_cone": 0.0,  # ignore these
                # "barrier": 0.0,  # ignore these
            }
            self.per_class_idx_max_dist = torch.tensor(
                [
                    class_ranges[class_name]
                    for class_name in NuscenesDataset.movable_class_names
                ]
            )
        self.target_class_names = list(class_ranges.keys())
        custom_movable_only_eval_config = {
            "class_range": class_ranges,
            "dist_fcn": "center_distance",
            "dist_ths": [0.5, 1.0, 2.0, 4.0],
            "dist_th_tp": 2.0,
            "min_recall": 0.1,
            "min_precision": 0.1,
            "max_boxes_per_sample": 500,
            "mean_ap_weight": 5,
        }
        self.verbose = True
        self.cfg = DetectionConfig(**custom_movable_only_eval_config)
        self.class_mapping = {}

    def filter_boxes_by_dist(self, boxes: Shape):
        if torch.count_nonzero(boxes.valid) > 0:
            box_dist = torch.linalg.norm(boxes.pos[:, :2], dim=-1)
            if self.eval_movable_classes_as_one:
                pred_too_far = box_dist > self.cfg.class_range["movable"]
            else:
                box_class_ids = torch.squeeze(boxes.class_id, dim=-1).long()
                pred_too_far = box_dist > self.per_class_idx_max_dist[box_class_ids].to(
                    box_dist.device
                )
            assert boxes.valid.shape == pred_too_far.shape, (
                boxes.valid.shape,
                pred_too_far.shape,
            )
            boxes.valid = ~pred_too_far & boxes.valid
        filtered_boxes = boxes.drop_padding_boxes()
        return filtered_boxes

    def update(
        self,
        *,
        non_batched_gt_boxes: Shape,
        non_batched_pred_boxes: Shape,
        sample_token: str,
    ):
        dist_filtered_non_batched_gt_boxes = self.filter_boxes_by_dist(
            non_batched_gt_boxes.clone()
        )
        dist_filtered_non_batched_pred_boxes = self.filter_boxes_by_dist(
            non_batched_pred_boxes.clone()
        )

        assert dist_filtered_non_batched_gt_boxes.valid.all()
        assert dist_filtered_non_batched_pred_boxes.valid.all()

        gt_nusc_boxes = self.translate_our_box_to_nusc_box(
            dist_filtered_non_batched_gt_boxes.clone().detach().cpu(), sample_token
        )
        self.gt_boxes.add_boxes(sample_token=sample_token, boxes=gt_nusc_boxes)
        pred_boxes = dist_filtered_non_batched_pred_boxes.clone().cpu().detach()
        pred_boxes.dims = torch.clip(
            pred_boxes.dims, min=0.0001
        )  # otherwise iou calculation will crash in nuscenes eval code
        pred_nusc_boxes = self.translate_our_box_to_nusc_box(pred_boxes, sample_token)
        self.pred_boxes.add_boxes(sample_token=sample_token, boxes=pred_nusc_boxes)

    def translate_our_box_to_nusc_box(
        self, non_batched_boxes: Shape, sample_token: str, is_gt=False
    ):
        boxes = non_batched_boxes.detach().cpu()
        boxes.change_order_confidence_descending()
        sensor_T_box = boxes.get_poses()
        sensor_quats_box = (
            matrix_to_quaternion(sensor_T_box[:, :3, :3]).detach().cpu().numpy()
        )
        sensor_T_box = sensor_T_box.numpy()
        boxes = boxes.numpy()
        gt_nusc_boxes = []
        for box_idx, quat in enumerate(sensor_quats_box):
            box = boxes[box_idx]

            score_args = {}
            if not is_gt:
                confidence_score = float(np.squeeze(box.probs))
                assert (
                    confidence_score >= 0.0 and confidence_score <= 1
                ), f"nuscenes does not like confidence {confidence_score}"
                score_args["detection_score"] = confidence_score

            if self.eval_movable_classes_as_one:
                class_name = "movable"
            else:
                class_name = NuscenesDataset.idx_to_class_name_mapping[
                    int(box.class_id)
                ]
            gt_nusc_box = DetectionBox(
                sample_token=sample_token,
                translation=box.pos,
                size=box.dims,
                rotation=quat,  # quat.elements,
                velocity=(0, 0),
                ego_translation=(0, 0, 0),
                detection_name=class_name,
                num_pts=-1,
                **score_args,
            )
            gt_nusc_boxes.append(gt_nusc_box)
        return gt_nusc_boxes

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        start_time = time.time()

        metric_data_list = DetectionMetricDataList()

        for class_name in self.target_class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(
                    self.gt_boxes,
                    self.pred_boxes,
                    class_name,
                    self.cfg.dist_fcn_callable,
                    dist_th,
                    self.class_mapping,
                    verbose=True,
                )
                metric_data_list.set(class_name, dist_th, md)

        print("Calculating metrics...")
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.target_class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ["traffic_cone"] and metric_name in [
                    "attr_err",
                    "vel_err",
                    "orient_err",
                ]:
                    tp = np.nan
                elif class_name in ["barrier"] and metric_name in [
                    "attr_err",
                    "vel_err",
                ]:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def log(
        self,
        global_step: int,
        summary_writer: SummaryWriter,
        writer_prefix: str = "",
        render_curves_to: str = "",
    ):
        metrics, metric_data_list = self.evaluate()

        if render_curves_to:
            self.plot_dir = render_curves_to
            Path(render_curves_to).mkdir(parents=True, exist_ok=True)
            self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        metrics_summary = metrics.serialize()

        # Print high-level metrics.
        summary_writer.add_scalar(
            writer_prefix.rstrip("/") + "/mAP", metrics_summary["mean_ap"], global_step
        )
        print(
            writer_prefix.rstrip("/") + "/mAP",
            metrics_summary["mean_ap"],
        )
        err_name_mapping = {
            "trans_err": "mATE",
            "scale_err": "mASE",
            "orient_err": "mAOE",
            "vel_err": "mAVE",
            "attr_err": "mAAE",
        }
        for tp_name, tp_val in metrics_summary["tp_errors"].items():
            summary_writer.add_scalar(
                "%s/%s" % (writer_prefix.rstrip("/"), err_name_mapping[tp_name]),
                tp_val,
                global_step,
            )
            print(
                "%s/%s: %.4f"
                % (
                    writer_prefix.rstrip("/"),
                    err_name_mapping[tp_name],
                    tp_val,
                )
            )
        print("%s/NDS: %.4f" % (writer_prefix.rstrip("/"), metrics_summary["nd_score"]))
        summary_writer.add_scalar(
            writer_prefix.rstrip("/") + "/NDS",
            (metrics_summary["nd_score"]),
            global_step,
        )
        print("Eval time: %.1fs" % metrics_summary["eval_time"])

        # Print per-class metrics.
        print()
        print("Per-class results:")

        class_aps = metrics_summary["mean_dist_aps"]
        class_tps = metrics_summary["label_tp_errors"]

        for class_name in class_aps.keys():
            metric_names_vals = {
                "AP": class_aps[class_name],
                "ATE": class_tps[class_name]["trans_err"],
                "ASE": class_tps[class_name]["scale_err"],
                "AOE": class_tps[class_name]["orient_err"],
                "AVE": class_tps[class_name]["vel_err"],
                "AAE": class_tps[class_name]["attr_err"],
            }
            for metr_name, metr_val in metric_names_vals.items():
                summary_writer.add_scalar(
                    "%s/%s/%s" % (writer_prefix.rstrip("/"), class_name, metr_name),
                    metr_val,
                    global_step,
                )
                print(
                    "%s/%s/%s: %.4f"
                    % (writer_prefix.rstrip("/"), class_name, metr_name, metr_val)
                )

        # print(metrics)
        # print(metric_data_list)

    def render(
        self, metrics: DetectionMetrics, md_list: DetectionMetricDataList
    ) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print("Rendering PR and TP curves")

        tueplots_available = True
        try:
            from tueplots import cycler, figsizes, fonts
            from tueplots.constants.color import palettes

            plt.rcParams.update(cycler.cycler(color=palettes.paultol_muted))
            # Increase the resolution of all the plots below
            plt.rcParams.update({"figure.dpi": 300})
            plt.rcParams.update(figsizes.cvpr2022_half())
            plt.rcParams.update(fonts.neurips2021())
            # plt.rcParams.update({"text.usetex": True})
        except ImportError as e:
            print("Error: Could not import tueplots. Please install it via pip.")
            print(e)
            tueplots_available = False

        suffix = ".pdf"

        def savepath(name):
            return os.path.join(self.plot_dir, name + suffix)

        summary_plot(
            md_list,
            metrics,
            min_precision=self.cfg.min_precision,
            min_recall=self.cfg.min_recall,
            dist_th_tp=self.cfg.dist_th_tp,
            savepath=savepath("summary_half"),
            target_class_names=self.target_class_names,
        )
        if tueplots_available:
            plt.rcParams.update(figsizes.cvpr2022_full())

        summary_plot(
            md_list,
            metrics,
            min_precision=self.cfg.min_precision,
            min_recall=self.cfg.min_recall,
            dist_th_tp=self.cfg.dist_th_tp,
            savepath=savepath("summary_full"),
            target_class_names=self.target_class_names,
        )


def summary_plot(
    md_list: DetectionMetricDataList,
    metrics: DetectionMetrics,
    min_precision: float,
    min_recall: float,
    dist_th_tp: float,
    target_class_names=None,
    savepath: str = None,
) -> None:
    """
    Creates a summary plot with PR and TP curves for each class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param min_precision: Minimum precision value.
    :param min_recall: Minimum recall value.
    :param dist_th_tp: The distance threshold used to determine matches.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """

    n_classes = len(target_class_names)
    _, axes = plt.subplots(nrows=n_classes, ncols=1)
    detection_name = target_class_names[0]

    ax1 = setup_axis(
        xlim=1,
        ylim=1,
        min_precision=min_precision,
        min_recall=min_recall,
        ax=axes,
        labelsize=None,
    )
    ax1.set_ylabel(
        "Precision"
        # "{} Precision".format(PRETTY_DETECTION_NAMES[detection_name]),
    )

    ax1.set_xlabel("Recall")

    class_pr_curve(md_list, metrics, detection_name, min_precision, min_recall, ax=ax1)

    if savepath is not None:
        plt.savefig(savepath)
        plt.close()
