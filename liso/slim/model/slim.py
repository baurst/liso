import numpy as np
import torch
from liso.kabsch.main_utils import get_network_input_pcls
from liso.slim.model.head_decoder import HeadDecoder
from liso.slim.model.raft_mod import RAFT
from liso.slim.slim_loss.movavg_cls_threshold import MovingAverageThreshold
from torch import nn


class SLIM(nn.Module):  # type:ignore
    def __init__(self, cfg, num_train_samples, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.slim_cfg = cfg.SLIM
        bev_pc_range_half = 0.5 * np.array(self.cfg.data.bev_range_m)
        bev_pc_range = np.concatenate([-bev_pc_range_half, bev_pc_range_half], axis=0)
        self.head_decoder_fw = HeadDecoder(
            self.slim_cfg, bev_extent=bev_pc_range, name="head_decoder_forward"
        )
        self.head_decoder_bw = HeadDecoder(
            self.slim_cfg, bev_extent=bev_pc_range, name="head_decoder_backward"
        )
        assert self.slim_cfg.phases.train.mode in [
            "supervised",
            "unsupervised",
        ], self.config.mode

        if self.slim_cfg.phases.train.mode == "unsupervised":
            num_still_points = None
        else:
            num_still_points = self.slim_cfg.data.train.num_still_points
            num_still_points *= self.slim_cfg.model.num_iters
        self.moving_dynamicness_threshold = MovingAverageThreshold(
            num_train_samples=num_train_samples,
            num_moving=621013971,  # TODO: where did this number come from?
            num_still=num_still_points,
        )
        self.raft_network = RAFT(
            cfg=self.cfg,
            head_decoder_fw=self.head_decoder_fw,
            head_decoder_bw=self.head_decoder_bw,
        )

    def forward(
        self,
        sample_data_t0,
        sample_data_t1,
        summaries,
    ):
        # predictions is list across iterations
        outputs_fw, outputs_bw, aux_outputs = self.raft_network(
            get_network_input_pcls(
                self.cfg,
                sample_data_t0,
                "ta",
                to_device="cuda",
            ),
            get_network_input_pcls(
                self.cfg,
                sample_data_t1,
                "ta",
                to_device="cuda",
            ),
        )
        filled_pillar_mask_t0 = torch.squeeze(
            aux_outputs["t0"]["bev_net_input_dbg"] > 0.5, dim=1
        )
        filled_pillar_mask_t1 = torch.squeeze(
            aux_outputs["t1"]["bev_net_input_dbg"] > 0.5, dim=1
        )

        predictions_fw = []
        predictions_bw = []

        not_enough_points = 0

        for it, (net_output_0_1, net_output_1_0) in enumerate(
            zip(outputs_fw, outputs_bw)
        ):
            should_write_img_summaries = summaries["imgs_eval"] and (
                it == len(outputs_fw) - 1
            )
            should_write_metric_summaries = summaries["metrics_eval"] and (
                it == len(outputs_fw) - 1
            )
            cur_summaries = {
                "writer": summaries["writer"],
                "imgs_eval": should_write_img_summaries,
                "metrics_eval": should_write_metric_summaries,
                # "metrics_label_dict": summaries["metrics_label_dict"],
                # "label_mapping": summaries["label_mapping"],
            }
            gt_flow_bev_fw = sample_data_t0.get("gt", {}).get("flow_bev_ta_tb", None)
            if gt_flow_bev_fw is not None:
                gt_flow_bev_fw = gt_flow_bev_fw.to(net_output_0_1.device)

            prediction_fw = self.head_decoder_fw(
                net_output_0_1,
                dynamicness_threshold=self.moving_dynamicness_threshold.value(),
                pointwise_valid_mask=sample_data_t0["pcl_ta"]["pcl_is_valid"].to(
                    net_output_0_1.device
                ),
                pointwise_voxel_coordinates=sample_data_t0["pcl_ta"]["pillar_coors"].to(
                    net_output_0_1.device
                ),
                pc=sample_data_t0["pcl_ta"]["pcl"].to(net_output_0_1.device),
                filled_pillar_mask=filled_pillar_mask_t0,
                odom=sample_data_t0["gt"]["odom_ta_tb"].to(net_output_0_1.device),
                inv_odom=sample_data_t1["gt"]["odom_ta_tb"].to(net_output_0_1.device),
                summaries=cur_summaries,
                gt_flow_bev=gt_flow_bev_fw,
                ohe_gt_stat_dyn_ground_label_bev_map=sample_data_t0.get(
                    "ohe_gt_stat_dyn_ground_label_bev_map_t0", None
                ),
                dynamic_flow_is_non_rigid_flow=self.slim_cfg.model.dynamic_flow_is_non_rigid_flow,
            )
            gt_flow_bev_bw = sample_data_t1.get("gt", {}).get("flow_bev_ta_tb", None)
            if gt_flow_bev_bw is not None:
                gt_flow_bev_bw = gt_flow_bev_bw.to(net_output_1_0.device)
            prediction_bw = self.head_decoder_bw(
                net_output_1_0,
                dynamicness_threshold=self.moving_dynamicness_threshold.value(),
                pointwise_valid_mask=sample_data_t1["pcl_ta"]["pcl_is_valid"].to(
                    net_output_1_0.device
                ),
                pointwise_voxel_coordinates=sample_data_t1["pcl_ta"]["pillar_coors"].to(
                    net_output_1_0.device
                ),
                pc=sample_data_t1["pcl_ta"]["pcl"].to(net_output_1_0.device),
                filled_pillar_mask=filled_pillar_mask_t1,
                odom=sample_data_t1["gt"]["odom_ta_tb"].to(net_output_1_0.device),
                inv_odom=sample_data_t0["gt"]["odom_ta_tb"].to(net_output_1_0.device),
                gt_flow_bev=gt_flow_bev_bw,
                summaries=cur_summaries,
                ohe_gt_stat_dyn_ground_label_bev_map=sample_data_t1.get(
                    "ohe_gt_stat_dyn_ground_label_bev_map_t1", None
                ),
                dynamic_flow_is_non_rigid_flow=self.slim_cfg.model.dynamic_flow_is_non_rigid_flow,
            )
            not_enough_points += (
                prediction_fw["not_enough_points"].detach().cpu().numpy().sum()
            )
            not_enough_points += (
                prediction_bw["not_enough_points"].detach().cpu().numpy().sum()
            )

            predictions_fw.append(prediction_fw)
            predictions_bw.append(prediction_bw)

        if summaries["writer"] is not None:
            summaries["writer"].add_scalar("not_enough_points", not_enough_points)

        self.predictions_fw = predictions_fw
        self.predictions_bw = predictions_bw

        return self.predictions_fw, self.predictions_bw
