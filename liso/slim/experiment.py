import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from liso.datasets.argoverse2.av2_torch_dataset import (
    get_av2_train_dataset,
    get_av2_val_dataset,
)
from liso.datasets.kitti_raw_torch_dataset import (
    KittiRawDataset,
    get_kitti_train_dataset,
)
from liso.datasets.kitti_tracking_torch_dataset import get_kitti_val_dataset
from liso.datasets.nuscenes_torch_dataset import (
    get_nuscenes_train_dataset,
    get_nuscenes_val_dataset,
)
from liso.datasets.torch_dataset_commons import LidarDataset
from liso.datasets.waymo_torch_dataset import (
    get_waymo_train_dataset,
    get_waymo_val_dataset,
)
from liso.eval.flow_metrics import FlowMetrics
from liso.kabsch.main_utils import (
    get_datasets,
    get_network_input_pcls,
    log_bce_loss_img,
)
from liso.kabsch.mask_dataset import RecursiveDeviceMover
from liso.slim.model.slim import SLIM
from liso.slim.slim_loss.slim_loss_adaptor import (  # supervisedSlimSingleScaleLoss,
    selfsupervisedSlimSingleScaleLoss,
)
from liso.slim.utils.metrics import (
    aggregate_metrics,
    compute_scene_flow_metrics_for_points_in_this_mask,
)
from liso.slim.utils.pointwise2bev import scatter_pointwise2bev
from liso.slim.utils.tb_factory import TBFactory
from liso.utils.config_helper_helper import pretty_json
from liso.utils.learning_rate import get_polynomial_decay_schedule_with_warmup
from liso.visu.bbox_image import (
    draw_boxes_on_2d_projection,
    log_box_movement,
    render_pcl_range_image,
)
from liso.visu.flow_image import log_flow_image, pytorch_create_flow_image
from tqdm import tqdm
from yaml import dump


def list_of_dicts_to_dict_of_lists(in_list):
    if len(in_list) == 0:
        return {}
    else:
        out_dict = {}
        keys = set(in_list[0].keys())
        for k in keys:
            out_dict[k] = []
        for d in in_list:
            for k, v in d.items():
                out_dict[k].append(v)

        return out_dict


class Experiment:
    def __init__(
        self,
        cfg,
        slim_cfg,
        log_dir,
        maybe_slow_log_dir=None,
        fast_test=False,
        debug=False,
        global_step=0,
        world_size=1,
        worker_id=0,
    ):
        self.cfg = cfg
        self.fast_test = fast_test
        self.debug_mode = debug
        self.log_dir = log_dir
        self.maybe_slow_log_dir = maybe_slow_log_dir

        self.checkpoint_dir = Path(log_dir).joinpath("checkpoints")
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # self.logger = logger
        self.tb_factory = TBFactory(Path(log_dir).joinpath("tb"))
        self.slim_cfg = slim_cfg
        self.initial_cfg_check(debug)
        self.set_version()
        self.global_step = global_step
        self.cfg_was_tb_logged = False
        bev_pc_range_half = 0.5 * np.array(self.cfg.data.bev_range_m)
        self.bev_extent = np.concatenate(
            [-bev_pc_range_half, bev_pc_range_half], axis=0
        )
        self.path_to_loaded_model_weights = None

        self.world_size = int(world_size)
        self.worker_id = int(worker_id)

    def initial_cfg_check(self, debug):
        # print(self.cfg.model.output_modification)
        if "supervised" in self.slim_cfg.losses:
            if self.slim_cfg.losses.supervised.mode == "total_flow":
                assert (
                    self.slim_cfg.model.output_modification.disappearing_logit is False
                )
                assert self.slim_cfg.model.output_modification.static_logit is True
                assert self.slim_cfg.model.output_modification.dynamic_logit is False
                assert self.slim_cfg.model.output_modification.ground_logit is False
                stat_flow_source = ["net"]
                if debug:
                    stat_flow_source.append("gt")
                assert (
                    self.slim_cfg.model.output_modification.static_flow
                    in stat_flow_source
                )
                dyn_flow_source = ["zero"]
                if debug:
                    dyn_flow_source.append("gt")
                assert (
                    self.slim_cfg.model.output_modification.dynamic_flow
                    in dyn_flow_source
                )
            elif self.slim_cfg.losses.supervised.mode == "dyn_and_stat_with_cls":
                assert (
                    self.slim_cfg.model.output_modification.disappearing_logit is False
                )
                assert self.slim_cfg.model.output_modification.static_logit == "net"
                assert self.slim_cfg.model.output_modification.dynamic_logit == "net"
                assert self.slim_cfg.model.output_modification.ground_logit is False
                assert self.slim_cfg.model.output_modification.static_flow == "net"

        assert "pretrain" in self.slim_cfg.iterations
        if set(self.slim_cfg.phases.keys()) == {"train"}:
            assert self.slim_cfg.iterations.pretrain == 0
        elif set(self.slim_cfg.phases.keys()) == {"train", "pretrain"}:
            assert self.slim_cfg.iterations.pretrain > 0
        else:
            raise ValueError(
                "unknown combination of training phases: %s"
                % self.slim_cfg.phases.keys()
            )

    def set_version(self):
        if hasattr(self, "version"):
            if "version" in self.slim_cfg:
                self.version = self.slim_cfg.version + "_" + self.version
                del self.slim_cfg["version"]
        elif "version" in self.slim_cfg:
            self.version = self.slim_cfg.version
            del self.slim_cfg["version"]
        else:
            self.version = "unversioned"

    def prepare(self, for_training):
        print("START PREPARING EXPERIMENT!")
        self.global_step = 1

        # pp_voxel_cfg = PointPillarsLayer.get_voxel_config(
        #     bev_extent=self.cfg.data.bev_extent,
        #     **{
        #         k: v
        #         for k, v in self.cfg.model.point_pillars.items()
        #         if k in {"nbr_pillars", "inf_distance", "max_points_per_pillar"}
        #     },
        # )
        # self.train_loader_labelmap = get_train_dataset_and_labelmap(
        #     self.cfg, pp_voxel_cfg
        # )
        if for_training:
            (
                self.train_loader,
                _,
                self.val_loader,
                self.val_on_train_loader,
            ) = get_datasets(
                self.cfg, self.fast_test, target="flow", shuffle_validation=True
            )
            num_train_samples = len(self.train_loader)

        else:
            num_train_samples = 15000  # TODO: this should not be necessary, we will overwrite moving threshold anyway during loading of model weights

        self.mask_gt_renderer = RecursiveDeviceMover(self.cfg).cuda()
        self.model = SLIM(
            cfg=self.cfg,
            num_train_samples=num_train_samples,
        )
        self.model.cuda()

        if for_training:
            if self.slim_cfg.optimizer == "rmsprop":
                self.optimizer = torch.optim.RMSprop(
                    self.model.parameters(),
                    lr=self.slim_cfg.learning_rate.initial,
                )
            elif self.slim_cfg.optimizer == "adam":
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.slim_cfg.learning_rate.initial,
                )  # network/fnet/layer2/layer_with_weights-0/norm3/gamma/.ATTRIBUTES/VARIABLE_VALUE
            else:
                raise AssertionError("only rmsprop/adam supported")

            self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.slim_cfg.learning_rate.warm_up.step_length,
                num_training_steps=self.slim_cfg.iterations.train,
                lr_end=self.slim_cfg.learning_rate.initial * 0.05,
            )

    def load_model_weights(self, path_to_weights: Path):
        self.model.load_state_dict(torch.load(path_to_weights))
        self.path_to_loaded_model_weights = path_to_weights

    def run_inference_only(self, skip_existing: bool):
        target_dir = self.maybe_slow_log_dir.joinpath("preds")
        target_dir.mkdir(parents=True, exist_ok=True)

        info = {"model_weights": str(self.path_to_loaded_model_weights)}
        with open(target_dir.joinpath("pred_info.yml"), "w") as yaml_file:
            dump(info, yaml_file, default_flow_style=False)

        ds_args = {
            "use_geom_augmentation": False,
            "shuffle": False,
            "size": None,
            "verbose": False,
            "get_only_these_specific_samples": None,
        }
        if self.cfg.data.source == "nuscenes":
            t0_t1_loader, t0_t1_ds = get_nuscenes_train_dataset(
                cfg=self.cfg, use_skip_frames="never", **ds_args
            )
            t0_t2_loader, _ = get_nuscenes_train_dataset(
                cfg=self.cfg, use_skip_frames="only", **ds_args
            )
            val_loader = get_nuscenes_val_dataset(
                self.cfg,
                use_skip_frames="never",
                size=None,
                shuffle=True,
            )
            val_on_train_loader = None
        elif self.cfg.data.source == "kitti":
            t0_t1_loader, t0_t1_ds = get_kitti_train_dataset(
                cfg=self.cfg, use_skip_frames="never", **ds_args
            )
            t0_t2_loader, _ = get_kitti_train_dataset(
                cfg=self.cfg, use_skip_frames="only", **ds_args
            )
            val_loader, _ = get_kitti_val_dataset(
                self.cfg,
                size=None,
                target="flow",  # get tracking dataset
                use_skip_frames="never",
                shuffle=False,
                mode="val",
            )
            val_on_train_loader = None  # kitti tracking dataloader loads all data
        elif self.cfg.data.source == "waymo":
            t0_t1_loader, t0_t1_ds = get_waymo_train_dataset(
                cfg=self.cfg,
                use_skip_frames="never",
                **ds_args,
                need_flow_during_training=False,
            )
            t0_t2_loader, _ = get_waymo_train_dataset(
                cfg=self.cfg,
                use_skip_frames="only",
                **ds_args,
                need_flow_during_training=False,
            )
            print("Skipping export of T0->T2 samples Waymo to speed up export!")
            t0_t2_loader = [
                None,
            ] * len(t0_t1_loader)
            val_loader = get_waymo_val_dataset(
                self.cfg,
                size=None,
                use_skip_frames="never",
                shuffle=False,
                need_flow=False,
            )
            val_on_train_loader = None
        elif self.cfg.data.source == "av2":
            t0_t1_loader, t0_t1_ds = get_av2_train_dataset(
                cfg=self.cfg,
                use_skip_frames="never",
                **ds_args,
                need_flow_during_training=False,
            )
            print("Skipping export of T0->T2 samples AV2 to speed up export!")
            t0_t2_loader = [
                None,
            ] * len(t0_t1_loader)
            val_loader = get_av2_val_dataset(
                self.cfg,
                size=None,
                use_skip_frames="never",
                shuffle=False,
            )
            val_on_train_loader = None

        assert len(t0_t1_loader) == len(t0_t2_loader), "missed frames"

        no_summaries = {
            "writer": None,
            "imgs_eval": False,
            "metrics_eval": False,
            "aggregated_metrics": False,
        }

        with torch.no_grad():
            for vl in (val_loader, val_on_train_loader):
                if vl is None:
                    continue
                for sample_idx, val_el in tqdm(
                    enumerate(vl), total=len(vl), disable=False
                ):
                    if self.world_size > 1:
                        if sample_idx % self.world_size != self.worker_id:
                            continue

                    self.slim_inference_and_save_result(
                        target_dir,
                        t0_t1_ds,
                        no_summaries,
                        val_el,
                        t0_t2_el=None,
                        skip_existing=skip_existing,
                    )
            # print("NOT EXPORTING ON TRAIN AGAIN")
            # print("REMOVE EXIT BELOW TO RUN EXPORT ON TRAIN SET")
            # print("DONE")
            # sys.exit(0)
            for sample_idx, (t0_t1_el, t0_t2_el) in tqdm(
                enumerate(zip(t0_t1_loader, t0_t2_loader)),
                total=len(t0_t1_loader),
                disable=False,
            ):
                if self.world_size > 1:
                    if sample_idx % self.world_size != self.worker_id:
                        continue
                self.slim_inference_and_save_result(
                    target_dir,
                    t0_t1_ds,
                    no_summaries,
                    t0_t1_el,
                    t0_t2_el,
                    skip_existing=skip_existing,
                )

    @torch.no_grad()
    def slim_inference_and_save_result(
        self,
        target_dir: Path,
        t0_t1_ds: LidarDataset,
        no_summaries: Dict[str, str],
        t0_t1_el: Dict[str, torch.FloatTensor],
        t0_t2_el: Dict[str, torch.FloatTensor] = None,
        skip_existing=False,
    ):
        (
            sample_data_t0,
            sample_data_t1,
            _,
            meta_data_t0_t1,
        ) = self.mask_gt_renderer(t0_t1_el)

        target_file = target_dir.joinpath(meta_data_t0_t1["sample_id"][0]).with_suffix(
            ".npz"
        )
        if skip_existing and target_file.exists():
            return

        preds_fw, preds_bw = self.model(
            sample_data_t0,
            sample_data_t1,
            summaries=no_summaries,
        )
        preds = {}
        preds["bev_raw_flow_t0_t1"] = preds_fw[-1].modified_network_output.static_flow
        preds["bev_raw_flow_t1_t0"] = preds_bw[-1].modified_network_output.static_flow
        preds["bev_dynamicness_t0_t1"] = preds_fw[
            -1
        ].modified_network_output.dynamicness
        preds["bev_dynamicness_t1_t0"] = preds_bw[
            -1
        ].modified_network_output.dynamicness
        save_stuff = {
            "static_threshold": self.model.moving_dynamicness_threshold.value(),
        }

        if t0_t2_el is not None:
            (
                sample_data_t0_2,
                sample_data_t2,
                _,
                meta_data_t0_t2,
            ) = self.mask_gt_renderer(t0_t2_el)

            assert meta_data_t0_t1["sample_id"] == meta_data_t0_t2["sample_id"], (
                meta_data_t0_t1["sample_id"],
                meta_data_t0_t2["sample_id"],
            )
            assert len(meta_data_t0_t1["sample_id"]) == 1, len(
                meta_data_t0_t1["sample_id"]
            )
            assert torch.all(
                sample_data_t0["pcl_ta"]["pcl"] == sample_data_t0_2["pcl_ta"]["pcl"]
            )
            preds_fw, preds_bw = self.model(
                sample_data_t0_2,
                sample_data_t2,
                summaries=no_summaries,
            )
            preds["bev_raw_flow_t0_t2"] = preds_fw[
                -1
            ].modified_network_output.static_flow
            preds["bev_raw_flow_t2_t0"] = preds_bw[
                -1
            ].modified_network_output.static_flow
            preds["bev_dynamicness_t0_t2"] = preds_fw[
                -1
            ].modified_network_output.dynamicness
            preds["bev_dynamicness_t2_t0"] = preds_bw[
                -1
            ].modified_network_output.dynamicness

            preds_fw, preds_bw = self.model(
                sample_data_t1,
                sample_data_t2,
                summaries=no_summaries,
            )
            preds["bev_raw_flow_t1_t2"] = preds_fw[
                -1
            ].modified_network_output.static_flow
            preds["bev_raw_flow_t2_t1"] = preds_bw[
                -1
            ].modified_network_output.static_flow
            preds["bev_dynamicness_t1_t2"] = preds_fw[
                -1
            ].modified_network_output.dynamicness
            preds["bev_dynamicness_t2_t1"] = preds_bw[
                -1
            ].modified_network_output.dynamicness

        # save:
        save_stuff = {
            **save_stuff,
            **preds,
        }
        save_stuff_cpu = {
            k: torch.squeeze(v, dim=0).detach().cpu().numpy()
            for k, v in save_stuff.items()
        }
        save_stuff_cpu["bev_range_m"] = t0_t1_ds.bev_range_m_np
        target_file.parent.mkdir(
            exist_ok=True, parents=True
        )  # for waymo we have subfolders
        np.savez_compressed(target_file, **save_stuff_cpu)

    def run(self):
        self.model.train()

        train_writer = self.tb_factory("train", self.cfg.data.source + "/")
        train_summaries = {
            "writer": train_writer,
            "imgs_eval": self.global_step % 100 == 0 or self.debug_mode,
            "metrics_eval": self.global_step % 100 == 0 or self.debug_mode,
            "aggregated_metrics": False,
        }
        train_iterator = iter(self.train_loader)
        while self.global_step < self.slim_cfg.iterations.train:
            self.optimizer.zero_grad()
            try:
                full_train_data = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_loader)
                full_train_data = next(train_iterator)
            (
                sample_data_t0,
                sample_data_t1,
                _,
                meta_data,
            ) = self.mask_gt_renderer(full_train_data)
            pred_fw, _ = self.train_one_step(
                sample_data_t0,
                sample_data_t1,
                train_summaries,
            )
            # if self.global_step % 100 == 0 or self.debug_mode:
            #     monitor_input_output_data(train_writer, train_el, pred_fw)
            if self.debug_mode or (
                (self.global_step % self.cfg.logging.img_log_interval) == 0
            ):
                log_flow_image(
                    train_writer,
                    self.cfg,
                    self.global_step,
                    (pred_fw.dense_maps.aggregated_flow[..., 0:2])
                    .clone()
                    .permute(0, 3, 1, 2),
                    suffix="aggregated",
                )
                gt_bev_flow = sample_data_t0.get("gt", {}).get("flow_bev_ta_tb", None)
                if gt_bev_flow is not None:
                    log_flow_image(
                        train_writer,
                        self.cfg,
                        self.global_step,
                        gt_bev_flow[..., :2].permute((0, 3, 1, 2)),
                        suffix="gt",
                    )
                dynamicness_bev = pred_fw.modified_network_output.dynamicness
                log_bce_loss_img(
                    self.cfg,
                    "pred_dynamicness",
                    train_writer,
                    self.global_step,
                    dynamicness_bev[:, None, :, :],
                )
            if (
                self.global_step % self.slim_cfg.iterations.full_eval_every == 0
                or self.debug_mode
            ):
                max_train_eval_iter = self.cfg.validation.num_val_on_train_steps
                max_val_eval_iter = self.cfg.validation.num_val_steps
                if self.debug_mode:
                    max_train_eval_iter = 5
                    max_val_eval_iter = 5
                    print("Warning: EVAL ON SUPER FEW SAMPLES")
                if isinstance(self.val_on_train_loader.dataset, KittiRawDataset):
                    print(
                        f"Skipping validation on {self.val_on_train_loader.dataset.__class__.__name__}: No GT available!"
                    )
                else:
                    self.eval_model(
                        "train", self.val_on_train_loader, max_train_eval_iter
                    )
                self.eval_model("valid", self.val_loader, max_val_eval_iter)
                self.model.train()
                torch.save(
                    self.model.state_dict(),
                    self.checkpoint_dir.joinpath(f"{self.global_step}.pth"),
                )
            if self.debug_mode and self.global_step > 3:
                break

    def eval_model(self, name, data_loader, max_iterations=None):
        curr_time = datetime.now()
        print(
            f"{curr_time}:Running validation on {name} {data_loader.dataset.__class__.__name__}: Dataset size: {len(data_loader)}, eval on {max_iterations} batches."
        )

        self.model.eval()
        img_writer = self.tb_factory("valid", name + "/")
        eval_metrics = self.run_eval_on_this_dataset(
            val_loader=data_loader,
            img_writer=img_writer,
            max_iterations=max_iterations,
        )
        for flow_category, v in eval_metrics.items():
            val_tb_writer = self.tb_factory("valid", name + "/" + flow_category + "/")
            for key, scalar_metric in v.items():
                if np.isscalar(scalar_metric):
                    val_tb_writer.add_scalar(key, scalar_metric)
                else:
                    pass

    def run_eval_on_this_dataset(self, val_loader, img_writer, max_iterations=None):
        val_summaries = {
            "writer": None,
            "imgs_eval": False,
            "metrics_eval": False,
            "aggregated_metrics": False,
        }
        list_of_metrics_dicts = {
            "raw/overall": [],
            "agg/overall": [],
            "rig/overall": [],
            "raw/moving": [],
            "agg/moving": [],
            "rig/moving": [],
            "raw/still": [],
            "agg/still": [],
            "rig/still": [],
        }
        flow_metrics = {k: FlowMetrics() for k in ("raw", "agg", "rig")}
        with torch.no_grad():
            num_val_steps = 0
            for val_el in tqdm(val_loader, disable=False):
                (
                    sample_data_t0,
                    sample_data_t1,
                    _,
                    meta_data,
                ) = self.mask_gt_renderer(val_el)
                # print("Val on el: {0}".format(meta_data["sample_id"][0]))
                preds_fw, _ = self.model(
                    sample_data_t0,
                    sample_data_t1,
                    val_summaries,
                )
                num_val_steps += 1
                if self.debug_mode and num_val_steps > 3:
                    break
                if max_iterations is not None and num_val_steps > max_iterations:
                    break

                pred = preds_fw[-1]
                gt_flow = sample_data_t0["gt"]["flow_ta_tb"].cpu().numpy()
                moving_mask = (
                    sample_data_t0["gt"]["moving_mask"].cpu().numpy()
                    & sample_data_t0["pcl_ta"]["pcl_is_valid"].cpu().numpy()
                    & sample_data_t0["gt"]["point_has_valid_flow_label"].cpu().numpy()
                )
                static_mask = (
                    np.logical_not(moving_mask)
                    & sample_data_t0["pcl_ta"]["pcl_is_valid"].cpu().numpy()
                    & sample_data_t0["gt"]["point_has_valid_flow_label"].cpu().numpy()
                )
                # TODO: make sure this is correct
                pred_flows_for_eval = {"raw": pred.static_flow.cpu().numpy()}
                if pred.aggregated_flow is not None:
                    pred_agg_flow = pred.aggregated_flow.cpu().numpy()
                    pred_flows_for_eval["agg"] = pred_agg_flow
                if pred.static_aggr_flow is not None:
                    pred_rig_flow = pred.static_aggr_flow.cpu().numpy()
                    pred_flows_for_eval["rig"] = pred_rig_flow
                if self.debug_mode or num_val_steps % 20 == 0:
                    # monitor_input_output_data(img_writer, train_data_t0_t1, pred)
                    log_flow_image(
                        img_writer,
                        self.cfg,
                        self.global_step + num_val_steps,
                        (pred.dense_maps.aggregated_flow[..., 0:2])
                        .clone()
                        .permute(0, 3, 1, 2),
                        suffix="aggregated",
                    )
                    log_flow_image(
                        img_writer,
                        self.cfg,
                        self.global_step + num_val_steps,
                        (pred.dense_maps.static_flow[..., 0:2])
                        .clone()
                        .permute(0, 3, 1, 2),
                        suffix="raw",
                    )
                    log_flow_image(
                        img_writer,
                        self.cfg,
                        self.global_step + num_val_steps,
                        (
                            sample_data_t0["gt"]["flow_bev_ta_tb"][:, :, :, :2].permute(
                                (0, 3, 1, 2)
                            )
                        ).clone(),
                        suffix="gt",
                    )

                    dynamicness_bev = pred.modified_network_output.dynamicness
                    log_bce_loss_img(
                        self.cfg,
                        "pred_dynamicness",
                        img_writer,
                        self.global_step + num_val_steps,
                        dynamicness_bev[:, None, :, :],
                    )
                    masked_pillar_coors = sample_data_t0["pcl_ta"][
                        "pillar_coors"
                    ].clone()
                    bev_masked_pred_flow, bev_pred_mask = scatter_pointwise2bev(
                        pred.static_flow.cpu(),
                        masked_pillar_coors,
                        sample_data_t0["pcl_ta"]["pcl_is_valid"],
                        self.cfg.data.img_grid_size,
                    )
                    rgb_img = pytorch_create_flow_image(
                        bev_masked_pred_flow[..., :2].permute(0, -1, 1, 2)
                    ).permute(0, 2, 3, 1)
                    img_writer.add_image(
                        "masked_pred_flow",
                        torch.where(
                            bev_pred_mask[..., None], rgb_img, torch.zeros_like(rgb_img)
                        )[0, ...],
                        dataformats="HWC",
                        global_step=self.global_step + num_val_steps,
                    )
                    masked_pillar_coors[
                        ~sample_data_t0["gt"]["point_has_valid_flow_label"]
                    ] = 0
                    bev_masked_gt_flow, bev_mask = scatter_pointwise2bev(
                        sample_data_t0["gt"]["flow_ta_tb"],
                        masked_pillar_coors,
                        sample_data_t0["pcl_ta"]["pcl_is_valid"]
                        & sample_data_t0["gt"]["point_has_valid_flow_label"],
                        self.cfg.data.img_grid_size,
                    )

                    rgb_img = pytorch_create_flow_image(
                        bev_masked_gt_flow[..., :2].permute(0, -1, 1, 2)
                    ).permute(0, 2, 3, 1)
                    img_writer.add_image(
                        "masked_gt_flow",
                        torch.where(
                            bev_mask[..., None], rgb_img, torch.zeros_like(rgb_img)
                        )[0, ...],
                        dataformats="HWC",
                        global_step=self.global_step + num_val_steps,
                    )

                    pcls = get_network_input_pcls(
                        self.cfg, sample_data_t0, "ta", to_device="cuda"
                    )
                    elevation_rad_max = np.deg2rad(10.0)
                    elevation_rad_min = np.deg2rad(-30.0)
                    range_image_height = 128
                    range_image_width = 1024
                    img_canvas = render_pcl_range_image(
                        pcls,
                        elevation_rad_max,
                        elevation_rad_min,
                        range_image_height,
                        range_image_width,
                    )

                    if (
                        "boxes" in sample_data_t0["gt"]
                        and "boxes" in sample_data_t1["gt"]
                    ):
                        log_box_movement(
                            cfg=self.cfg,
                            writer=img_writer,
                            global_step=self.global_step + num_val_steps,
                            sample_data_a=sample_data_t0,
                            sample_data_b=sample_data_t1,
                            pred_boxes=None,
                            writer_prefix="gt",
                        )
                    if "boxes" in sample_data_t0["gt"]:
                        gt_boxes = sample_data_t0["gt"]["boxes"]
                        draw_boxes_on_2d_projection(
                            img_canvas,
                            gt_boxes,
                            elevation_rad_max,
                            elevation_rad_min,
                            box_color=np.array([1.0, 0.0, 0.0]),
                        )

                    img_canvas = np.repeat(
                        img_canvas, repeats=2, axis=1  # increase vertical resolution
                    )
                    img_writer.add_images(
                        "range_images",
                        img_canvas,
                        dataformats="NHWC",
                        global_step=self.global_step + num_val_steps,
                    )
                    # monitor_layer_flow_logits(
                    #    img_writer,
                    #    gt_flow=gt_flow[0, ...],
                    #    gt_flow_bev=gt_flow_bev[0, ...],
                    #    raw_flows=pred_flows_for_eval["raw"][0, ...],
                    #    agg_flows=pred_flows_for_eval["agg"][0, ...],
                    #    static_agg_flows=pred_flows_for_eval["rig"][0, ...],
                    #    static_logits=pred["staticness"][0, ...].detach().cpu().numpy(),
                    #    confidence_logits=None,
                    #    static_threshold=self.model.moving_dynamicness_threshold.value()
                    #    .detach()
                    #    .cpu()
                    #    .numpy(),
                    #    pc1=val_el["pcl_t0"]["pc"][0, ...].numpy(),
                    #    pc2=val_el["pcl_t1"]["pc"][0, ...].numpy(),
                    # )
                for flow_name, eval_flow in pred_flows_for_eval.items():
                    for batch_idx in range(sample_data_t0["pcl_ta"]["pcl"].shape[0]):
                        flow_metrics[flow_name].update(
                            points=sample_data_t0["pcl_ta"]["pcl"][batch_idx]
                            .cpu()
                            .numpy(),
                            flow_pred=eval_flow[batch_idx],
                            flow_gt=gt_flow[batch_idx],
                            is_moving=moving_mask[batch_idx],
                            mask=sample_data_t0["pcl_ta"]["pcl_is_valid"][batch_idx]
                            .cpu()
                            .numpy(),
                        )
                    list_of_metrics_dicts["%s/overall" % flow_name].append(
                        compute_scene_flow_metrics_for_points_in_this_mask(
                            eval_flow, gt_flow, np.logical_or(moving_mask, static_mask)
                        )
                    )
                    if np.count_nonzero(moving_mask) > 0:
                        list_of_metrics_dicts["%s/moving" % flow_name].append(
                            compute_scene_flow_metrics_for_points_in_this_mask(
                                eval_flow,
                                gt_flow,
                                moving_mask,
                            )
                        )
                    if np.count_nonzero(static_mask) > 0:
                        list_of_metrics_dicts["%s/still" % flow_name].append(
                            compute_scene_flow_metrics_for_points_in_this_mask(
                                eval_flow,
                                gt_flow,
                                static_mask,
                            )
                        )

        eval_metrics = {}
        for k, list_of_metrics in list_of_metrics_dicts.items():
            if len(list_of_metrics):
                eval_metrics[k] = aggregate_metrics(list_of_metrics)
        for flow_type, metr in flow_metrics.items():
            metr.log_metrics_curves(
                global_step=self.global_step,
                summary_writer=img_writer,
                writer_prefix=flow_type,
            )
        return eval_metrics

    def train_one_step(
        self,
        sample_data_t0,
        sample_data_t1,
        summaries,
    ):
        self.tb_factory.global_step = self.global_step
        metrics_dicts = []
        slim_loss = torch.zeros(1).cuda()
        summaries["writer"] = self.tb_factory("train", "monitoring/")

        # print("Training on el: {0}".format(train_el["name"][0].decode("utf-8")))

        preds_fw, preds_bw = self.model(
            sample_data_t0,
            sample_data_t1,
            summaries,
        )

        pc1 = sample_data_t0["pcl_ta"]["pcl"].to("cuda")
        valid_mask_pc1 = sample_data_t0["pcl_ta"]["pcl_is_valid"].to("cuda")
        pc2 = sample_data_t1["pcl_ta"]["pcl"].to("cuda")
        valid_mask_pc2 = sample_data_t1["pcl_ta"]["pcl_is_valid"].to("cuda")
        for pred_fw, pred_bw in zip(preds_fw, preds_bw):
            intermediate_metrics_dict = {}
            if self.slim_cfg.phases.train.mode == "unsupervised":
                slim_loss = slim_loss + selfsupervisedSlimSingleScaleLoss(
                    pc1=pc1,
                    valid_mask_pc1=valid_mask_pc1,
                    pc2=pc2,
                    valid_mask_pc2=valid_mask_pc2,
                    pred_fw=pred_fw,
                    pred_bw=pred_bw,
                    moving_thresh_module=self.model.moving_dynamicness_threshold,
                    loss_cfg=self.slim_cfg.losses.unsupervised,
                    model_cfg=self.slim_cfg.model,
                    bev_extent=self.bev_extent,
                    metrics_collector=intermediate_metrics_dict,
                )
            elif self.slim_cfg.phases.train.mode == "supervised":
                raise NotImplementedError()
                # slim_loss = slim_loss + supervisedSlimSingleScaleLoss(
                #    train_el,
                #    pred_fw,
                #    pred_bw,
                #    moving_thresh_module=self.model.moving_dynamicness_threshold,
                #    loss_cfg=self.slim_cfg.losses.supervised,
                #    model_cfg=self.slim_cfg.model,
                #    metrics_collector=intermediate_metrics_dict,
                #    summaries=summaries,
                #    training=True,
                # )
            else:
                raise NotImplementedError(
                    "Don't know mode {0}".format(self.slim_cfg.phases.train.mode)
                )
            metrics_dicts.append(intermediate_metrics_dict)
        self.optimizer.zero_grad()
        slim_loss.backward()
        if not self.cfg_was_tb_logged:
            self.tb_factory("train", "cfg/").add_text(
                "cfg",
                "\n\nCommand: `$ %s`\n\n    %s"
                % (
                    os.path.abspath(sys.argv[0]) + " " + " ".join(sys.argv[1:]),
                    pretty_json(self.cfg),
                ),
            )
            self.cfg_was_tb_logged = True
        self.optimizer.step()
        self.lr_scheduler.step()
        # print(dict(self.model.named_parameters())["network.fnet.layer2.0.conv2.weight"])
        acc_metrics = list_of_dicts_to_dict_of_lists(metrics_dicts)
        acc_metrics = {k: sum(v) for k, v in acc_metrics.items()}
        for name, value in acc_metrics.items():
            self.tb_factory("train", "metrics/").add_scalar(name, value)

        self.tb_factory("train", "training/").add_scalar(
            "lr",
            self.lr_scheduler.get_last_lr()[0],
            global_step=self.global_step,
        )

        self.global_step += 1

        return preds_fw[-1], preds_bw[-1]
