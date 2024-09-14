from pathlib import Path
from typing import Tuple

import matplotlib
import numpy as np
from liso.visu.utils import plot_to_np_image
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("agg")


class FlowMetrics:
    def __init__(self, range_bins: Tuple[float] = None) -> None:
        if range_bins is None:
            range_bins = np.linspace(start=0.0, stop=100.0, num=11)
        self.range_bins = np.array(range_bins)
        self.categories = ("still", "moving", "overall")
        self.colors = ("green", "red", "blue")
        self.num_points_in_range_bin = {
            k: np.zeros(len(range_bins) - 1, dtype=np.int64) for k in self.categories
        }
        self.aee_per_range_bin = {
            k: np.zeros(len(range_bins) - 1, dtype=np.float64) for k in self.categories
        }
        self.total_aees = {k: 0.0 for k in self.categories}
        self.total_num_pts = {k: 0 for k in self.categories}

        self.line_width = 1
        self.legend_font_size = 6

    def update(
        self,
        points: np.ndarray,
        flow_pred: np.ndarray,
        flow_gt: np.ndarray,
        is_moving: np.ndarray,
        mask: np.ndarray,
    ):
        assert len(points.shape) == 2, points.shape
        assert len(is_moving.shape) == 1, is_moving.shape
        assert len(mask.shape) == 1, mask.shape

        range_m = np.linalg.norm(points[:, :3], axis=-1)
        end_point_errors_m = np.linalg.norm(flow_pred - flow_gt, axis=-1)
        category_masks = {
            "overall": mask,
            "still": mask & ~is_moving,
            "moving": mask & is_moving,
        }
        for bin_idx in range(len(self.range_bins) - 1):
            min_range = self.range_bins[bin_idx]
            max_range = self.range_bins[bin_idx + 1]
            point_is_in_range = (min_range <= range_m) & (range_m < max_range)
            for category_key, category_mask in category_masks.items():
                relevant_points_mask = point_is_in_range & category_mask
                num_pts = np.count_nonzero(relevant_points_mask)
                if num_pts > 0:
                    aee = np.mean(end_point_errors_m[relevant_points_mask])
                    old_avg = self.aee_per_range_bin[category_key][bin_idx]
                    old_num_pts = self.num_points_in_range_bin[category_key][bin_idx]
                    new_aee = (old_avg * old_num_pts + aee * num_pts) / (
                        old_num_pts + num_pts
                    )
                    self.aee_per_range_bin[category_key][bin_idx] = new_aee
                    self.num_points_in_range_bin[category_key][bin_idx] = (
                        old_num_pts + num_pts
                    )
        for category_key, relevant_points_mask in category_masks.items():
            num_pts = np.count_nonzero(relevant_points_mask)
            if num_pts > 0:
                aee = np.mean(end_point_errors_m[relevant_points_mask])
                old_avg = self.total_aees[category_key]
                old_num_pts = self.total_num_pts[category_key]
                new_aee = (old_avg * old_num_pts + aee * num_pts) / (
                    old_num_pts + num_pts
                )
                self.total_aees[category_key] = new_aee
                self.total_num_pts[category_key] = old_num_pts + num_pts

    def log_metrics_curves(
        self,
        global_step: int,
        summary_writer: SummaryWriter = None,
        writer_prefix: str = "",
        path: str = None,
    ):
        summary_prefix = writer_prefix.rstrip("/")
        plt.figure()
        plt.xlabel("Range [m]")
        plt.ylabel("Average-End-Point-Error [m]")
        for category_key, category_color in zip(self.categories, self.colors):
            plt.stairs(
                self.aee_per_range_bin[category_key],
                edges=self.range_bins,
                lw=self.line_width,
                color=category_color,
                label=category_key,
            )
            plt.axhline(
                self.total_aees[category_key],
                linestyle="dashdot",
                lw=self.line_width,
                color=category_color,
                label="overall "
                + category_key
                + f": {self.total_aees[category_key]:.3f}",
            )
        plt.legend(prop={"size": self.legend_font_size})
        if path:
            plt.savefig(Path(path).joinpath(f"aee_stats_{global_step}.png"))
        if summary_writer:
            fig = plt.figure(1)
            summary_writer.add_image(
                summary_prefix + "/aee_stats",
                plot_to_np_image(fig),
                global_step=global_step,
                dataformats="HWC",
            )
            summary_writer.flush()

        plt.close(fig)

        plt.figure()
        plt.xlabel("Range [m]")
        plt.ylabel("Number of Points in Bin")
        for category_key, category_color in zip(self.categories, self.colors):
            plt.stairs(
                self.num_points_in_range_bin[category_key],
                edges=self.range_bins,
                lw=self.line_width,
                color=category_color,
                label=category_key + " total:" + f"{self.total_num_pts[category_key]}",
            )
        plt.yscale("log")
        plt.legend(prop={"size": self.legend_font_size})
        if path:
            plt.savefig(Path(path).joinpath(f"points_per_bin_{global_step}.png"))
        if summary_writer:
            fig = plt.figure(1)
            summary_writer.add_image(
                summary_prefix + "/point_per_bin",
                plot_to_np_image(fig),
                global_step=global_step,
                dataformats="HWC",
            )
            summary_writer.flush()

        plt.close(fig)

        metrics = {}
        for category_key in self.categories:
            tag = summary_prefix + "/AEE/" + category_key
            metrics[tag] = self.total_aees[category_key]
            if summary_writer:
                summary_writer.add_scalar(
                    tag=tag,
                    scalar_value=self.total_aees[category_key],
                    global_step=global_step,
                )

        if summary_writer:
            summary_writer.flush()

        print(metrics)
        return metrics
