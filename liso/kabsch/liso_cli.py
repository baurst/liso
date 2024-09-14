import gc
import time
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp

import matplotlib
import torch
from config_helper.config import get_config_hash
from liso.datasets.create_gt_augm_database import (
    build_augmentation_db_from_actual_groundtruth,
)
from liso.datasets.kitti_raw_torch_dataset import KittiRawDataset
from liso.datasets.torch_dataset_commons import lidar_dataset_collate_fn, worker_init_fn
from liso.eval.eval_ours import run_val
from liso.kabsch.main_utils import (
    apply_rotation_regularization_loss,
    get_box_dbs_path,
    get_datasets,
    get_network_input_pcls,
    log_box_image,
    log_gt_bev_maps,
    sanity_check_cfg,
    sanity_check_flow,
    sv_hungarian_loss,
)
from liso.kabsch.mask_dataset import RecursiveDeviceMover
from liso.losses.centerpoint_loss import centerpoint_loss
from liso.losses.transfusion_loss import compute_transfusion_heatmap_loss
from liso.networks.flow_cluster_detector.flow_cluster_detector import (
    FlowClusterDetector,
)
from liso.networks.simple_net.simple_net import select_network
from liso.networks.simple_net.simple_net_utils import load_checkpoint_check_sanity
from liso.tracker.mined_box_db_utils import load_mined_boxes_db
from liso.tracker.tracking import (
    copy_box_db_to_dir,
    get_clean_train_dataset_single_batch,
    track_boxes_on_data_sequence,
)
from liso.utils.config_helper_helper import load_handle_args_cfg_logdir, pretty_json
from liso.visu.visualize_box_augmentation_database import (
    visualize_augm_boxes_with_points_inside_them,
)
from torch.utils.tensorboard.writer import SummaryWriter

matplotlib.use("agg")


def get_train_data_source(cfg, sample_data_t0, augm_sample_data_t0):
    if cfg.data.augmentation.boxes.active:
        train_data_source = augm_sample_data_t0
    else:
        train_data_source = sample_data_t0
    return train_data_source


def main():
    log_dir_prefix = None
    args, cfg, maybe_slow_log_dir = load_handle_args_cfg_logdir(
        log_dir_prefix=log_dir_prefix,
        # create_log_dir=False,
        # save_cfg=False,
    )

    cfg.data.setdefault("train_on_box_source", "mined")
    if cfg.data.train_on_box_source == "gt":
        # this method takes forever during tracking
        # but we can use it to create a gt box db
        # which is much faster and happens only once
        # torch.multiprocessing.set_start_method("spawn")
        pass

    max_size_of_sv_db_mb = 100  # only active during supervised training
    if args.profile or args.cprofile:
        max_size_of_sv_db_mb = 3
        cfg.checkpoint.save_model_every = 500
        cfg.validation.val_every_n_steps = 500
        cfg.validation.num_val_steps = 2
        cfg.validation.num_val_on_train_steps = 5
        cfg.logging.img_log_interval = 100
        cfg.optimization.num_training_steps = 250
        cfg.optimization.rounds.steps_per_round = (
            2 * cfg.optimization.num_training_steps
        )  # we dont want to profile the tracking
        cfg.optimization.abs_num_warmup_steps = 10
        cfg.optimization.rounds.active = False

        if cfg.optimization.rounds.active:
            cfg.data.tracking_cfg.min_track_age = min(
                cfg.data.tracking_cfg.min_track_age, 2
            )
            print(
                f"WARNING: Setting min_track_age to {cfg.data.tracking_cfg.min_track_age}, because we don't train long enough to find plausible tracks which leads to crash"
            )

    fast_test = args.fast_test
    if fast_test:
        cfg.data.paths.nuscenes.box_dbs.local = "/tmp/box_dbs/nuscenes"
        cfg.data.paths.waymo.box_dbs.local = "/tmp/box_dbs/nuscenes"
        cfg.data.paths.kitti.box_dbs.local = "/tmp/box_dbs/nuscenes"
        max_size_of_sv_db_mb = 1
        _num_rounds = 2
        cfg.optimization.rounds.steps_per_round = 3
        cfg.validation.val_every_n_steps = cfg.optimization.rounds.steps_per_round
        cfg.checkpoint.save_model_every = cfg.optimization.rounds.steps_per_round
        cfg.validation.num_val_steps = 2
        cfg.validation.num_val_on_train_steps = 2
        cfg.logging.img_log_interval = cfg.optimization.rounds.steps_per_round // 2
        cfg.optimization.num_training_steps = (
            _num_rounds * cfg.optimization.rounds.steps_per_round
        )
        cfg.optimization.abs_num_warmup_steps = 1

        if cfg.optimization.rounds.active:
            cfg.data.tracking_cfg.min_track_age = min(
                cfg.data.tracking_cfg.min_track_age, 2
            )
            print(
                f"WARNING: Setting min_track_age to {cfg.data.tracking_cfg.min_track_age}, because we don't train long enough to find plausible tracks which leads to crash"
            )

    sanity_check_cfg(cfg)

    log_dir = maybe_slow_log_dir

    checkpoint_dir = log_dir.joinpath("checkpoints")
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cuda0 = torch.device("cuda:0")

    recursive_device_mover = RecursiveDeviceMover(cfg).to(cuda0)

    fwd_writer = SummaryWriter(log_dir.joinpath("fwd"))
    fwd_writer.add_text("config", pretty_json(cfg), 0)
    fwd_writer.flush()

    train_loader, _, val_loader, val_on_train_loader = get_datasets(
        cfg,
        fast_test,
        target="object",
        shuffle_validation=True,
        need_flow_during_training=False,
    )
    val_cfg = cfg

    (
        box_predictor,
        optimizer,
        lr_scheduler,
        resume_from_step,
    ) = get_network_optimizer_scheduler(
        cfg,
        device=cuda0,
        path_to_checkpoint=args.load_checkpoint,
        finetune=args.finetune,
    )
    box_predictor.train()
    train_iterator = iter(train_loader)

    if args.load_checkpoint and not args.finetune:
        assert resume_from_step > 0, resume_from_step
    else:
        assert resume_from_step == 0, "this will break all mining triggering logic!"
    for global_step in range(resume_from_step, cfg.optimization.num_training_steps + 1):
        trigger_reset_network_optimizer_scheduler_after_val = False
        number_of_current_round = global_step // cfg.optimization.rounds.steps_per_round
        if (
            cfg.optimization.rounds.active
            and number_of_current_round
            % cfg.optimization.rounds.drop_net_weights_every_nth_round
            == 0
            and global_step > 0
            and (global_step % cfg.optimization.rounds.steps_per_round == 0)
            and cfg.data.train_on_box_source == "mined"
        ):
            trigger_reset_network_optimizer_scheduler_after_val = True

        if (
            cfg.data.train_on_box_source == "mined"
            and (
                cfg.data.augmentation.boxes.active
                and global_step == cfg.data.augmentation.boxes.start_augm_at_step
            )
            or (
                cfg.optimization.rounds.active
                and (global_step % cfg.optimization.rounds.steps_per_round == 0)
            )
            or (
                cfg.loss.supervised.supervised_on_clusters.active
                and global_step == resume_from_step
            )
        ):
            print(
                f"Step: {global_step} - Deleting datasets to save RAM before starting tracking!"
            )
            del train_loader
            del val_on_train_loader
            gc.collect()

            skip_db_generation = False
            if global_step == resume_from_step:
                clean_dataset_for_db_creation = get_clean_train_dataset_single_batch(
                    cfg
                )
                if cfg.data.tracking_cfg.bootstrap_detector == "flow_cluster_detector":
                    box_predictor_for_tracking = FlowClusterDetector(cfg)
                else:
                    raise NotImplementedError(cfg.data.tracking_cfg.bootstrap_detector)
                box_db_base_dir = get_box_dbs_path(cfg)
                path_to_box_augm_db = box_db_base_dir / "boxes_db_global_step_0.npy"
                assert cfg.optimization.rounds.raw_or_tracked in {
                    "tracked",
                    "raw",
                }, cfg.optimization.rounds.raw_or_tracked
                path_to_mined_boxes_db = (
                    box_db_base_dir / f"{cfg.optimization.rounds.raw_or_tracked}.npz"
                )
                tracking_args = {"export_raw_tracked_detections_to": box_db_base_dir}
                if (
                    Path(path_to_box_augm_db).exists()
                    and Path(path_to_mined_boxes_db).exists()
                    and not cfg.data.force_redo_box_mining
                ):
                    skip_db_generation = True
            else:
                clean_dataset_for_db_creation = get_clean_train_dataset_single_batch(
                    cfg
                )
                cfg_hash = get_config_hash(cfg)[:5]
                datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                box_db_base_dir = (
                    get_box_dbs_path(cfg)
                    / f"round_{number_of_current_round}_step_{global_step}_{cfg_hash}_{datetime_str}"
                )
                fwd_writer.add_text(
                    "save_mined_box", box_db_base_dir.as_posix(), global_step
                )

                box_predictor_for_tracking = box_predictor
                tracking_args = {
                    "export_raw_tracked_detections_to": box_db_base_dir,
                }

            if skip_db_generation:
                print("Skipping DB generation!")
            else:
                if fast_test or args.profile:
                    tracking_args["min_num_boxes"] = 2
                    tracking_args["timeout_s"] = 60
                    tracking_args["max_augm_db_size_mb"] = 1
                else:
                    tracking_args[
                        "max_augm_db_size_mb"
                    ] = cfg.data.tracking_cfg.setdefault("max_augm_db_size_mb", 250)
                (
                    path_to_box_augm_db,
                    paths_to_mined_boxes_dbs,
                ) = track_boxes_on_data_sequence(
                    cfg=cfg,
                    dataset=clean_dataset_for_db_creation,
                    box_predictor=box_predictor_for_tracking,
                    writer=fwd_writer,
                    global_step=global_step,
                    writer_prefix="tracking",
                    tracking_cfg=cfg.data.tracking_cfg,
                    **tracking_args,
                )
                path_to_mined_boxes_db = paths_to_mined_boxes_dbs[
                    cfg.optimization.rounds.raw_or_tracked
                ]

            visualize_augm_boxes_with_points_inside_them(
                path_to_augm_box_db=path_to_box_augm_db,
                num_boxes_to_visualize=200,
                writer=fwd_writer,
                global_step=global_step,
                writer_prefix="augm_boxes_from_tracking",
            )

            # copy the box db to log dir
            copy_box_db_to_dir(
                path_to_box_augm_db,
                log_dir=log_dir,
                global_step=global_step,
            )
            copy_box_db_to_dir(
                path_to_mined_boxes_db,
                log_dir=log_dir,
                global_step=global_step,
            )

            if not isinstance(clean_dataset_for_db_creation, KittiRawDataset):
                # we don't have boxes or flow in the kitti raw to evaluate against
                eval_mined_boxes_loader = torch.utils.data.DataLoader(
                    clean_dataset_for_db_creation,
                    pin_memory=True,
                    batch_size=1,
                    num_workers=cfg.data.num_workers,
                    collate_fn=lidar_dataset_collate_fn,
                    shuffle=not fast_test,  # shuffle to get more diversity!!
                    # but during fast test only 3 samples are in the box db! -> 0 predictions
                    worker_init_fn=worker_init_fn,
                )
                run_val(
                    cfg,
                    eval_mined_boxes_loader,
                    load_mined_boxes_db(path_to_mined_boxes_db),
                    recursive_device_mover,
                    "mined_boxes_val/",
                    fwd_writer,
                    global_step,
                    max_num_steps=cfg.validation.num_val_steps,
                )

            train_loader, _, _, val_on_train_loader = get_datasets(
                cfg,
                fast_test,
                path_to_augmentation_db=path_to_box_augm_db,
                path_to_mined_boxes_db=path_to_mined_boxes_db,
                target="object",
                shuffle_validation=True,
                need_flow_during_training=False,
            )

            train_iterator = iter(train_loader)

        elif (
            global_step == resume_from_step  # first train iteration
            and cfg.data.train_on_box_source == "gt"
            and cfg.data.augmentation.boxes.active
        ):
            # we need to generate the box augmentation db from the groundtruth
            target_dir_box_augm_db = mkdtemp()
            path_to_box_augm_db = build_augmentation_db_from_actual_groundtruth(
                cfg,
                target_dir_box_augm_db,
                save_every_n_samples=100,
                min_num_points_in_box=20,
                max_size_of_db_mb=max_size_of_sv_db_mb,
            )
            visualize_augm_boxes_with_points_inside_them(
                path_to_augm_box_db=path_to_box_augm_db,
                num_boxes_to_visualize=200,
                writer=fwd_writer,
                global_step=global_step,
                writer_prefix="augm_boxes_from_gt",
            )
            train_loader, _, _, val_on_train_loader = get_datasets(
                cfg,
                fast_test,
                path_to_augmentation_db=path_to_box_augm_db,
                path_to_mined_boxes_db=None,
                target="object",
                shuffle_validation=True,
                need_flow_during_training=False,
            )

            train_iterator = iter(train_loader)

        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=cuda0)
        start_dataloading_time = time.perf_counter()
        try:
            full_train_data = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            full_train_data = next(train_iterator)
            print("=" * 20)
            print("END OF DATASET")
            print("=" * 20)
        end_dataloading_time = time.perf_counter()

        if (
            cfg.loss.supervised.supervised_on_clusters.active
            or cfg.data.augmentation.boxes.active
        ):
            need_augm_sample_data = True
            need_sample_data_t0 = False
        else:
            need_augm_sample_data = False
            need_sample_data_t0 = True
        trigger_img_logging = (global_step % cfg.logging.img_log_interval) == 0

        if trigger_img_logging:
            need_sample_data_t0 = True

        (
            sample_data_t0,
            _,
            augm_sample_data_t0,
            _,
        ) = recursive_device_mover(
            full_train_data,
            need_sample_data_t0=need_sample_data_t0,
            need_sample_data_t1=False,
            need_augm_sample_data_t0=need_augm_sample_data,
        )

        if cfg.loss.pointrcnn_loss.active or cfg.loss.pointpillars_loss.active:
            assert cfg.network.name in ("pointrcnn", "pointpillars"), cfg.network.name
            train_data_source = get_train_data_source(
                cfg, sample_data_t0, augm_sample_data_t0
            )

            forward_start_time = time.perf_counter()
            (pointrcnn_losses_dict) = box_predictor(
                None,
                get_network_input_pcls(
                    cfg, train_data_source, time_key="ta", to_device=cuda0
                ),
                gt_boxes=train_data_source[cfg.data.train_on_box_source]["boxes"],
                centermaps_gt=None,
            )
            forward_end_time = time.perf_counter()
            backward_start_time = time.perf_counter()
            for loss_name, loss_val in pointrcnn_losses_dict.items():
                loss = loss + loss_val
                fwd_writer.add_scalar(
                    loss_name,
                    loss_val,
                    global_step=global_step,
                )

        elif (
            cfg.loss.supervised.centermaps.active
            or cfg.loss.supervised.hungarian.active
            or cfg.optimization.rounds.active
            or cfg.loss.supervised.supervised_on_clusters.active
        ):
            assert (
                not cfg.network.name == "point_rcnn"
            ), "does loss computation on its own"
            assert cfg.network.name in (
                "transfusion",
                "centerpoint",
            ), cfg.network.name
            assert (
                cfg.loss.supervised.centermaps.active
                or cfg.loss.supervised.supervised_on_clusters.active
                or cfg.loss.pointrcnn_loss.active
            )

            train_data_source = get_train_data_source(
                cfg, sample_data_t0, augm_sample_data_t0
            )

            augm_loss_tag = f"{cfg.data.train_on_box_source}_augm_boxes"

            forward_start_time = time.perf_counter()
            (
                pred_boxes_on_augm_t0,
                pred_boxes_maps_on_augm_t0,
                raw_activated_box_attrs_on_augm_t0,
                aux_net_outputs_on_augm_t0,
            ) = box_predictor(
                None,
                get_network_input_pcls(
                    cfg, train_data_source, time_key="ta", to_device=cuda0
                ),
                None,
                centermaps_gt=None,
            )
            forward_end_time = time.perf_counter()
            backward_start_time = time.perf_counter()

            if cfg.loss.supervised.centermaps.confidence_target == "gaussian":
                cm_loss_weight = cfg.loss.supervised.supervised_on_clusters.weight

                cp_augm_regression_maps = {
                    attr_name: train_data_source[cfg.data.train_on_box_source][
                        f"centermaps_{attr_name}"
                    ]
                    for attr_name in cfg.loss.supervised.supervised_on_clusters.attrs
                }

                cp_augm_center_mask = train_data_source[cfg.data.train_on_box_source][
                    "centermaps_center_bool_mask"
                ]
            else:
                raise NotImplementedError(
                    cfg.loss.supervised_centermaps.confidence_target
                )
            if cfg.data.train_on_box_source == "gt" and cfg.data.source == "kitti":
                cp_ignore_region_is_true_mask = train_data_source[
                    cfg.data.train_on_box_source
                ]["ignore_region_is_true_mask"]
            else:
                cp_ignore_region_is_true_mask = torch.zeros_like(cp_augm_center_mask)

            rotation_loss_weights_map = torch.ones_like(
                cp_augm_regression_maps["probs"]
            )

            if trigger_img_logging:
                log_gt_bev_maps(
                    fwd_writer,
                    "AUGM",
                    global_step,
                    cp_augm_regression_maps,
                    cp_augm_center_mask,
                    cp_ignore_region_is_true_mask,
                )
                if cfg.network.name == "centerpoint" and (trigger_img_logging):
                    for map_name in raw_activated_box_attrs_on_augm_t0:
                        visu_map = raw_activated_box_attrs_on_augm_t0[map_name].mean(
                            keepdims=True, dim=-1
                        )
                        visu_map_normed = (
                            visu_map - torch.amin(visu_map, dim=(1, 2), keepdim=True)
                        ) / (
                            torch.amax(visu_map, dim=(1, 2), keepdim=True)
                            - torch.amin(visu_map, dim=(1, 2), keepdim=True)
                        )
                        fwd_writer.add_images(
                            f"PRED/{map_name}",
                            visu_map_normed,
                            global_step=global_step,
                            dataformats="NHWC",
                        )

                bev_occup_mask = aux_net_outputs_on_augm_t0["bev_net_input_dbg"]

                log_box_image(
                    cfg=cfg,
                    max_num_batches=min(cfg.logging.max_log_img_batches, 2),
                    writer=fwd_writer,
                    writer_prefix="AUGM/augmented_boxes/",
                    step=global_step,
                    gt_boxes=sample_data_t0["gt"]["boxes"],
                    occupancy_f32=bev_occup_mask,
                    pred_boxes=train_data_source[cfg.data.train_on_box_source]["boxes"],
                )
            if cfg.network.name == "centerpoint":
                centermap_losses = centerpoint_loss(
                    loss_cfg=cfg.loss,
                    # decoded_pred_boxes=pred_boxes_on_augm_t0,
                    raw_activated_pred_box_maps=raw_activated_box_attrs_on_augm_t0,
                    decoded_pred_box_maps=pred_boxes_maps_on_augm_t0,
                    gt_maps=cp_augm_regression_maps,
                    gt_center_mask=cp_augm_center_mask,
                    rotation_loss_weights_map=rotation_loss_weights_map,
                    box_prediction_cfg=cfg.box_prediction,
                    ignore_region_is_true_mask=cp_ignore_region_is_true_mask,
                )
            elif cfg.network.name == "transfusion":
                centermap_losses = compute_transfusion_heatmap_loss(
                    loss_cfg=cfg.loss,
                    box_vars=aux_net_outputs_on_augm_t0,
                    gt_maps=cp_augm_regression_maps,
                    gt_center_mask=cp_augm_center_mask,
                    ignore_region_is_true_mask=cp_ignore_region_is_true_mask,
                )
                assert (
                    cfg.loss.supervised.centermaps.confidence_target == "gaussian"
                ), cfg.loss.supervised.centermaps.confidence_target
                cluster_boxes_a = train_data_source[cfg.data.train_on_box_source][
                    "boxes"
                ]

                sv_hungarian_losses_dict = sv_hungarian_loss(
                    cfg=cfg,
                    writer=fwd_writer,
                    gt_boxes_a=cluster_boxes_a,
                    supervised_loss_boxes=pred_boxes_on_augm_t0.clone(),
                    raw_activated_box_attrs_a=raw_activated_box_attrs_on_augm_t0,
                    losses_dict={},
                    global_step=global_step,
                    train_these_box_attrs=cfg.loss.supervised.supervised_on_clusters.attrs,
                    hungarian_loss_weight=cfg.loss.supervised.hungarian.weight,
                    loss_extra_descr=augm_loss_tag,
                    ignore_region_is_true_mask=cp_ignore_region_is_true_mask,
                    # rotation_loss_weights=cluster_boxes_a.velo,  # this is expected to crash, not implemented yet
                )
                for loss_name, loss_val in sv_hungarian_losses_dict.items():
                    loss = loss + loss_val
                    fwd_writer.add_scalar(
                        loss_name,
                        loss_val,
                        global_step=global_step,
                    )
            else:
                raise NotImplementedError()

            sv_augm_loss = 0.0
            for loss_name, loss_val in centermap_losses.items():
                sv_augm_loss = sv_augm_loss + cm_loss_weight * loss_val
                fwd_writer.add_scalar(
                    f"loss/{augm_loss_tag}/{loss_name}",
                    cm_loss_weight * loss_val,
                    global_step=global_step,
                )
            regul_loss_dict = {}
            apply_rotation_regularization_loss(
                cfg,
                raw_activated_box_attrs_on_augm_t0,
                pred_boxes_on_augm_t0,
                regul_loss_dict,
            )
            for loss_name, loss_val in regul_loss_dict.items():
                sv_augm_loss = sv_augm_loss + loss_val
                fwd_writer.add_scalar(
                    f"loss/{augm_loss_tag}/{loss_name}",
                    cm_loss_weight * loss_val,
                    global_step=global_step,
                )

            loss = loss + sv_augm_loss
            fwd_writer.add_scalar(
                f"loss/{augm_loss_tag}/total_sv",
                sv_augm_loss,
                global_step=global_step,
            )
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        backward_end_time = time.perf_counter()

        timings = {
            "fwd_time": forward_end_time - forward_start_time,
            "bwd_time": backward_end_time - backward_start_time,
            "dataloading_time": end_dataloading_time - start_dataloading_time,
        }
        for time_desc, actual_time in timings.items():
            fwd_writer.add_scalar(
                f"timing/{time_desc}_s",
                actual_time,
                global_step=global_step,
            )
        if fast_test:
            print(timings)
        fwd_writer.add_scalar("loss/total", loss, global_step=global_step)
        fwd_writer.add_scalar(
            "lr",
            lr_scheduler.get_last_lr()[0],
            global_step=global_step,
        )

        if global_step % cfg.checkpoint.save_model_every == 0:
            save_experiment_state(
                checkpoint_dir, box_predictor, optimizer, lr_scheduler, global_step
            )

        if (global_step > 0) and global_step % cfg.validation.val_every_n_steps == 0:
            if cfg.data.flow_source != "gt" and global_step == 0:
                sanity_check_flow(
                    cfg,
                    train_loader,
                    fwd_writer,
                    global_step,
                    writer_prefix="train/sanity_check_flow",
                    max_num_steps=20 if fast_test else 200,
                )
                sanity_check_flow(
                    cfg,
                    val_on_train_loader,
                    fwd_writer,
                    global_step,
                    writer_prefix="val_on_train/sanity_check_flow",
                    max_num_steps=20 if fast_test else 200,
                )
            box_predictor.eval()
            run_val(
                val_cfg,
                val_loader,
                box_predictor,
                recursive_device_mover,
                "online_val/",
                fwd_writer,
                global_step,
                max_num_steps=cfg.validation.num_val_steps,
            )
            run_val(
                cfg,
                val_on_train_loader,
                box_predictor,
                recursive_device_mover,
                "val_on_train/",
                fwd_writer,
                global_step,
                max_num_steps=cfg.validation.num_val_on_train_steps,
            )
            torch.cuda.empty_cache()  # let's hope that fixes OOM?
            box_predictor.train()
        if trigger_reset_network_optimizer_scheduler_after_val:
            assert cfg.data.train_on_box_source != "gt", cfg.data.train_on_box_source
            print(f"{global_step}: RESETTING NETWORK, OPTIMIZER, SCHEDULER")
            box_predictor, optimizer, lr_scheduler, _ = get_network_optimizer_scheduler(
                cfg,
                path_to_checkpoint=None,
                device=cuda0,
            )
            box_predictor.train()
            trigger_reset_network_optimizer_scheduler_after_val = False

    if not (args.profile or args.cprofile):
        run_val(
            val_cfg,
            val_loader,
            box_predictor,
            recursive_device_mover,
            "online_val/",
            fwd_writer,
            global_step,
            max_num_steps=cfg.validation.num_val_steps,
        )
        # run final validation
        run_val(
            val_cfg,
            val_loader,
            box_predictor,
            recursive_device_mover,
            "complete_eval/",
            fwd_writer,
            global_step,
            max_num_steps=cfg.validation.num_val_steps,
        )

        _ = save_experiment_state(
            checkpoint_dir, box_predictor, optimizer, lr_scheduler, global_step
        )


def save_experiment_state(
    checkpoint_dir: Path, box_predictor, optimizer, lr_scheduler, global_step: int
):
    save_to = checkpoint_dir.joinpath(f"{global_step}.pth")
    torch.save(
        {
            "network": box_predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "global_step": global_step,
        },
        save_to,
    )
    return save_to


def get_network_optimizer_scheduler(
    cfg,
    device,
    path_to_checkpoint=None,
    finetune=False,
):
    """
    finetune: if True, we only load network weights, but not optimizer/scheduler state
    """
    # do not change order here:
    # see https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html#load-the-general-checkpoint
    box_predictor = select_network(cfg, device)
    if path_to_checkpoint is not None and not finetune:
        resume_from_step = int(Path(path_to_checkpoint).stem)
    else:
        resume_from_step = 0
    optimizer, lr_scheduler = get_optimizer_scheduler(
        cfg,
        box_predictor,
    )
    if path_to_checkpoint is not None:
        box_predictor = load_checkpoint_check_sanity(
            path_to_checkpoint=path_to_checkpoint,
            cfg=cfg,
            box_predictor=box_predictor,
        )
    if path_to_checkpoint is not None and not finetune:
        checkpoint_content = torch.load(path_to_checkpoint)
        if "optimizer" in checkpoint_content:
            optimizer.load_state_dict(checkpoint_content["optimizer"])
            print("Successfully loaded optimizer state dict")
        if "lr_scheduler" in checkpoint_content:
            lr_scheduler.load_state_dict(checkpoint_content["lr_scheduler"])
            print("Successfully loaded learning rate scheduler state dict")

        num_scheduler_steps = (
            resume_from_step
            if "global_step" not in checkpoint_content
            else checkpoint_content["global_step"]
        )
        try:
            for _ in range(num_scheduler_steps):
                lr_scheduler.step()
        except ValueError as e:
            print(e)
            print("(this should only happen with fast test and resume)")

    return box_predictor, optimizer, lr_scheduler, resume_from_step


def get_optimizer_scheduler(cfg, box_predictor):
    optimizer = torch.optim.AdamW(
        box_predictor.parameters(),
        lr=cfg.optimization.learning_rate,
        weight_decay=0.01,  # 0.01 is default
    )
    common_scheduler_kwargs = {
        "optimizer": optimizer,
        "max_lr": cfg.optimization.learning_rate,
        "pct_start": 0.4,
        "base_momentum": 0.85,
        "max_momentum": 0.95,
        "div_factor": 10.0,
    }
    if cfg.data.train_on_box_source == "gt":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            total_steps=cfg.optimization.num_training_steps
            + 2,  # for some reason this crashes if I dont have this extra two steps
            **common_scheduler_kwargs,
        )
    elif cfg.data.train_on_box_source == "mined":
        assert cfg.optimization.rounds.active, "assuming this"
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            total_steps=(
                cfg.optimization.rounds.steps_per_round
                * cfg.optimization.rounds.drop_net_weights_every_nth_round
            )
            + 2,  # for some reason this crashes if I dont have this extra two steps
            final_div_factor=10,
            **common_scheduler_kwargs,
        )
    return optimizer, lr_scheduler


if __name__ == "__main__":
    main()
