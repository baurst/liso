import time
from collections import defaultdict, namedtuple
from datetime import datetime
from pathlib import Path
from shutil import copy2
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from config_helper.config import save_config
from liso.box_fitting.box_fitting import fit_2d_box_modest
from liso.datasets.argoverse2.av2_torch_dataset import AV2Dataset
from liso.datasets.kitti_object_torch_dataset import KittiObjectDataset
from liso.datasets.kitti_raw_torch_dataset import KittiRawDataset
from liso.datasets.kitti_tracking_torch_dataset import (
    KittiTrackingDataset,
    get_kitti_val_dataset,
)
from liso.datasets.nuscenes_torch_dataset import NuscenesDataset
from liso.datasets.torch_dataset_commons import (
    LidarDataset,
    get_points_in_boxes_mask,
    lidar_dataset_collate_fn,
    worker_init_fn,
)
from liso.datasets.waymo_torch_dataset import WaymoDataset
from liso.eval.eval_ours import count_box_points_in_kitti_annotated_fov, run_val
from liso.kabsch.main_utils import get_network_input_pcls
from liso.kabsch.mask_dataset import RecursiveDeviceMover
from liso.kabsch.shape_utils import (
    Shape,
    extract_box_motion_transform_without_sensor_odometry,
    is_boxes_clearly_in_bev_range,
    soft_align_box_flip_orientation_with_motion_trafo,
)
from liso.networks.flow_cluster_detector.flow_cluster_detector import (
    FlowClusterDetector,
)
from liso.networks.simple_net.point_rcnn import PointRCNNWrapper
from liso.networks.simple_net.pointpillars import PointPillarsWrapper
from liso.networks.simple_net.simple_net import BoxLearner, select_network
from liso.networks.simple_net.simple_net_utils import load_checkpoint_check_sanity
from liso.slim.experiment import list_of_dicts_to_dict_of_lists
from liso.tracker.augm_box_db_utils import (
    drop_boxes_from_augmentation_db,
    estimate_augm_db_size_mb,
    get_empty_augm_box_db,
    save_augmentation_database,
)
from liso.tracker.box_tracker import NotATracker
from liso.tracker.global_box_tracker import FlowBasedBoxTracker
from liso.tracker.mined_box_db_utils import load_mined_boxes_db
from liso.tracker.track_smoothing import (
    MIN_TRACK_LEN_FOR_SMOOTHING,
    batch_box_data_for_batched_smoothing,
    batched_displacement_from_pos,
    smooth_track_bike_model,
    smooth_track_jerk,
    split_batched_padded_tensor_into_list,
)
from liso.tracker.tracking_helpers import (
    accumulate_pcl,
    aggregate_odometry_to_world_poses,
)
from liso.utils.config_helper_helper import (
    dumb_load_yaml_to_omegaconf,
    load_handle_args_cfg_logdir,
    parse_cli_args,
)
from liso.utils.nms_iou import iou_based_nms
from liso.utils.torch_transformation import homogenize_pcl, torch_decompose_matrix
from liso.visu.bbox_image import (
    draw_box_image,
    draw_box_onto_image,
    plot_text_on_canvas_at_position,
)
from liso.visu.pcl_image import (
    create_topdown_f32_pcl_image_variable_extent,
    project_2d_pcl_to_rowcol_nonsquare_bev_range,
)
from liso.visu.visualize_box_augmentation_database import (
    visualize_augm_boxes_with_points_inside_them,
)
from matplotlib.cm import gist_rainbow
from PIL import Image
from pynanoflann import KDTree
from tensorboard.compat.proto.summary_pb2 import Summary
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

BoxDataForSmoothing = namedtuple(
    "BoxDataForSmoothing",
    [
        "track_id",
        "track_age",
        "start_time_idx",
        "box_sequence_world_for_track_id",
        "box_sequence_sensor_for_track_id",
        "extra_attributes_for_this_box",
    ],
)


def copy_box_db_to_dir(path_to_mined_boxes_db: Path, log_dir: Path, global_step: int):
    print(f"Backing up mined boxes db to dir {log_dir}")
    path_to_tracking_db_folder = log_dir / "tracking_dbs" / str(global_step)
    path_to_tracking_db_folder.mkdir(parents=True, exist_ok=True)
    return copy2(path_to_mined_boxes_db, path_to_tracking_db_folder)


def main():
    args = parse_cli_args()
    assert len(args.keys_value) == 0, "will be ignored anyway"
    assert len(args.configs) == 0, "will be ignored anyway"
    args, _, log_dir = load_handle_args_cfg_logdir(
        args=args,
        save_cfg=False,
    )
    assert (
        args.load_checkpoint
    ), "please use a checkpoint with a config that makes sense, you can override the actual model!"
    cfg_path_chkpt = Path(args.load_checkpoint).parent.parent.joinpath("config.yml")
    cfg = dumb_load_yaml_to_omegaconf(cfg_path_chkpt)
    global_step = int(Path(args.load_checkpoint).stem)

    if args.override_network:
        cfg.network.name = args.override_network
        assert cfg.network.name in (
            "flow_cluster_detector",
            "echo_gt",
        ), cfg.network.name
        global_step = 0

    if args.export_predictions_for_tcr:
        print("WARNING")
        print("OVERRIDING DATASET CHOICE- CHOOSING KITTI TRACKING DATASET FOR TCR")
        cfg.data.batch_size = 1
        _, dataset = get_kitti_val_dataset(
            cfg,
            mode="val",
            size=None,
            target="flow",  # this triggers tracking to be used
            use_skip_frames="never",
            shuffle=False,
        )
    else:
        dataset = get_clean_train_dataset_single_batch(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    box_predictor = select_network(cfg, device=device)
    if not args.override_network:
        box_predictor = load_checkpoint_check_sanity(
            args.load_checkpoint, cfg, box_predictor
        )
        exp_desc = Path(
            args.load_checkpoint.split("/")[-5]
            + "_it"
            + Path(args.load_checkpoint).stem
        )
    else:
        exp_desc = cfg.network.name

    if isinstance(box_predictor, FlowClusterDetector):
        exp_desc = f"{cfg.data.flow_source}_flow_{exp_desc}"

    if args.export_predictions_to_dir:
        export_raw_tracked_detections_to = (
            Path(args.export_predictions_to_dir) / exp_desc
        )
        # export_raw_tracked_detections_to.mkdir(exist_ok=False, parents=True)
        export_raw_tracked_detections_to.mkdir(exist_ok=True, parents=True)
    else:
        export_raw_tracked_detections_to = None

    tb_writer = SummaryWriter(log_dir.joinpath("fwd"))

    print("Copying-config to new log dir")

    save_config(cfg, log_dir.joinpath("config_tracking.yml"))

    copy2(cfg_path_chkpt, log_dir)

    box_db_path, paths_to_mined_box_dbs = track_boxes_on_data_sequence(
        cfg=cfg,
        dataset=dataset,
        box_predictor=box_predictor,
        writer=tb_writer,
        global_step=global_step,
        writer_prefix="tracking",
        verbose=False,
        log_freq=1,
        export_raw_tracked_detections_to=export_raw_tracked_detections_to,
        export_detections_only_in_annotated_fov=True,
        tracking_cfg=cfg.data.tracking_cfg,
        timeout_s=5 * 60 if args.fast_test else None,
        max_augm_db_size_mb=1 if args.fast_test else 200,
        log_gifs_to_disk=args.dump_sequences_for_visu is not None,
        dump_sequences_for_visu=args.dump_sequences_for_visu,
    )

    visualize_augm_boxes_with_points_inside_them(
        path_to_augm_box_db=box_db_path,
        num_boxes_to_visualize=400,
        writer=tb_writer,
        global_step=global_step,
        writer_prefix="tracked_boxes_augm_db",
    )
    if not isinstance(dataset, KittiRawDataset):
        # we don't have boxes or flow in the kitti raw to evaluate against
        prefetch_args = {}
        eval_mined_boxes_loader = torch.utils.data.DataLoader(
            dataset,
            pin_memory=True,
            batch_size=1,
            num_workers=cfg.data.num_workers,
            collate_fn=lidar_dataset_collate_fn,
            shuffle=False,
            worker_init_fn=worker_init_fn,
            **prefetch_args,
        )
        mask_gt_renderer = RecursiveDeviceMover(cfg).cuda()

        run_val(
            cfg,
            eval_mined_boxes_loader,
            load_mined_boxes_db(paths_to_mined_box_dbs["tracked"]),
            mask_gt_renderer,
            "mined_boxes_val/",
            tb_writer,
            global_step=global_step,
            max_num_steps=cfg.validation.num_val_steps,
            img_log_interval=2 if args.fast_test else None,
        )


def set_box_size_keep_closest_point_constant(
    boxes: Shape, new_box_dims: torch.FloatTensor
) -> Shape:
    boxes.assert_attr_shapes_compatible()
    bottom_corner_idxs = torch.tensor(
        Shape.get_bottom_corner_idxs(), device=boxes.pos.device
    )
    box_corners_sensor, _ = boxes.get_box_corners()
    box_corners_sensor = box_corners_sensor[..., bottom_corner_idxs, :]
    dist_from_sensor = torch.linalg.norm(box_corners_sensor[..., :2], dim=-1)
    closest_corner_pt_idx = torch.argmin(dist_from_sensor, dim=-1)
    num_boxes = boxes.shape[0]
    closest_corner_pt_sensor = box_corners_sensor[
        torch.arange(num_boxes, device=closest_corner_pt_idx.device),
        closest_corner_pt_idx,
        :,
    ]
    shift_m = new_box_dims / boxes.dims * (boxes.pos - closest_corner_pt_sensor)
    boxes.pos = closest_corner_pt_sensor + shift_m
    boxes.dims = torch.ones_like(boxes.dims) * new_box_dims
    boxes.assert_attr_shapes_compatible()
    return boxes


def update_world_boxes_from_sensor_boxes(
    *,
    box_sequence_sensor: Shape,
    box_sequence_world: Shape,
    w_T_sensor_ti: List[torch.DoubleTensor],
):
    assert len(w_T_sensor_ti) == box_sequence_sensor.shape[0], (
        len(w_T_sensor_ti),
        box_sequence_sensor.shape[0],
    )
    box_sequence_sensor.assert_attr_shapes_compatible()
    box_sequence_world.assert_attr_shapes_compatible()
    assert box_sequence_sensor.shape == box_sequence_world.shape, (
        box_sequence_sensor.shape,
        box_sequence_world.shape,
    )
    # dims
    box_sequence_world.dims = box_sequence_sensor.dims
    s_T_b = box_sequence_sensor.get_poses()
    w_T_b = w_T_sensor_ti @ s_T_b
    boxes_pos, boxes_rot = torch_decompose_matrix(w_T_b)
    box_sequence_world.pos = boxes_pos
    box_sequence_world.rot = boxes_rot
    box_sequence_world.probs = box_sequence_sensor.probs
    return box_sequence_world


def update_sensor_boxes_from_world_boxes(
    *,
    box_sequence_world: Shape,
    box_sequence_sensor: Shape,
    w_T_sensor_ti: List[torch.DoubleTensor],
):
    assert len(w_T_sensor_ti) == box_sequence_sensor.shape[0], (
        len(w_T_sensor_ti),
        box_sequence_sensor.shape[0],
    )
    box_sequence_sensor.assert_attr_shapes_compatible()
    box_sequence_world.assert_attr_shapes_compatible()
    assert box_sequence_sensor.shape == box_sequence_world.shape, (
        box_sequence_sensor.shape,
        box_sequence_world.shape,
    )

    w_T_b = box_sequence_world.get_poses()

    s_T_b = torch.linalg.inv(w_T_sensor_ti) @ w_T_b

    boxes_pos, boxes_rot = torch_decompose_matrix(s_T_b)

    box_sequence_sensor.pos = boxes_pos
    box_sequence_sensor.rot = boxes_rot
    box_sequence_sensor.probs = box_sequence_world.probs
    return box_sequence_sensor


def get_sequence_id(cfg, meta_data):
    sample_ids = meta_data["sample_id"]
    if cfg.data.source in ("nuscenes", "waymo"):
        return [sid.split("_")[0] for sid in sample_ids]
    elif cfg.data.source == "kitti":
        return sample_ids
    elif cfg.data.source == "av2":
        return [sid.split("/")[2] for sid in sample_ids]
    else:
        raise NotImplementedError(cfg.data.source)


def get_odometry_gt_odmetry_and_flow(dataset, cfg, sample_data_t0, device):
    pointwise_flow_t0_t1 = sample_data_t0[cfg.data.flow_source]["flow_ta_tb"].to(device)
    odom_t0_t1 = sample_data_t0[cfg.data.odom_source]["odom_ta_tb"][0]
    gt_odom_t0_t1 = sample_data_t0["gt"]["odom_ta_tb"][0]
    ptwise_gt_flow_t0_t1 = sample_data_t0.get("gt", {}).get("flow_ta_tb", None)
    if ptwise_gt_flow_t0_t1 is None:
        ptwise_gt_flow_t0_t1 = pointwise_flow_t0_t1
    if isinstance(dataset, NuscenesDataset):
        # extrapolate motion, since our nuscenes is sampled at 10Hz
        assert (
            dataset.data_use_skip_frames == "never"
        ), "need to adapt extrapolation factor below!"
        motion_extrapolation_factor = 5.0
        pointwise_flow_t0_t1 = motion_extrapolation_factor * pointwise_flow_t0_t1
        ptwise_gt_flow_t0_t1 = motion_extrapolation_factor * ptwise_gt_flow_t0_t1
        nusc_odom_key = "odom_ta_tx"
        odom_t0_t1 = sample_data_t0[cfg.data.odom_source][nusc_odom_key][0]
        gt_odom_t0_t1 = sample_data_t0["gt"][nusc_odom_key][0]

    return (
        odom_t0_t1,
        pointwise_flow_t0_t1,
        gt_odom_t0_t1,
        ptwise_gt_flow_t0_t1,
    )


def normalize_data_to_new_range(
    data: np.ndarray, old_min: float, new_min: float, old_max: float, new_max: float
):
    assert old_max != old_min, ("will div by zero", old_max, old_min)
    return (data - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


class PCLBoxGifSummary:
    def __init__(
        self,
        cfg: Dict,
        summary_writer: SummaryWriter,
        img_grid_size: Tuple[int, int] = (1024, 1024),
        time_between_frames_s: float = 0.1,
        log_to_disk=False,
    ) -> None:
        self.cfg = cfg
        self.summary_writer = summary_writer
        self.pcl_npy_imgs_f32 = []
        self.sample_ids = []
        self.img_grid_size = np.array(img_grid_size)
        self.bev_range_m = np.array(cfg.data.bev_range_m)
        self.time_between_frames_s = time_between_frames_s
        self.log_to_disk = log_to_disk

    def update_pcl_npy_imgs(self, pcl: torch.FloatTensor, sample_id: str):
        assert pcl.shape[-1] == 4, pcl.shape
        assert len(pcl.shape) == 2, ("batching not supported", pcl.shape)
        self.sample_ids.append(sample_id)
        bev_range_m = torch.from_numpy(self.bev_range_m).to(pcl.device)
        (
            pixel_intensity_img,
            pixel_occup_mask,
        ) = create_topdown_f32_pcl_image_variable_extent(
            pcl,
            pcl[:, -1],
            coords_min=-bev_range_m / 2,
            coords_max=bev_range_m / 2,
            img_grid_size=torch.from_numpy(self.img_grid_size).to(pcl.device),
        )
        intensity_values = pixel_intensity_img[pixel_occup_mask].cpu().numpy()
        blank_canvas = np.zeros(
            (self.img_grid_size[0], self.img_grid_size[1], 3), dtype=np.float32
        )

        intensity_color_u8 = (
            plt.cm.jet(
                normalize_data_to_new_range(
                    intensity_values,
                    old_min=0.0,
                    new_min=0.35,
                    old_max=1.0,
                    new_max=1.0,
                )
            )[..., :3]
        ).astype(np.float32)
        blank_canvas[pixel_occup_mask.cpu().numpy()] = intensity_color_u8
        self.pcl_npy_imgs_f32.append(blank_canvas)

    def log_gif(self, npy_imgs_f32, writer_prefix: str, global_step: int) -> None:
        # self.summary_writer.add_video(
        #     writer_prefix,
        #     (255 * np.array(npy_imgs_f32)[None]).astype(np.uint8),
        #     global_step=global_step,
        #     fps=10,
        # )
        images = [Image.fromarray((img * 255).astype(np.uint8)) for img in npy_imgs_f32]

        # dest = (
        #     Path(self.summary_writer.get_logdir())
        #     / f"global_step_{global_step}"
        #     / writer_prefix
        # )
        # dest.mkdir(exist_ok=True, parents=True)
        duration = int(100 * self.time_between_frames_s / 0.1)
        if self.log_to_disk:
            path_to_gif = (
                Path(self.summary_writer.log_dir).parent / "gifs" / writer_prefix
            )
            path_to_gif.parent.mkdir(exist_ok=True, parents=True)
            images[0].save(
                path_to_gif.with_suffix(".gif"),
                save_all=True,
                append_images=images[1:],
                # optimize=True,
                duration=duration,
                loop=0,
                # subsampling=0,
                # quality=100,
            )
        with NamedTemporaryFile(suffix=".gif") as tmp_gif:
            # dest = dest / "boxes.gif"
            images[0].save(
                tmp_gif.name,
                save_all=True,
                append_images=images[1:],
                # optimize=True,
                duration=duration,
                loop=0,
                # subsampling=0,
                # quality=100,
            )
            with open(tmp_gif.name, "rb") as f:
                tensor_string = f.read()
            height, width, num_channels = npy_imgs_f32[0].shape
            video = Summary.Image(
                height=height,
                width=width,
                colorspace=num_channels,
                encoded_image_string=tensor_string,
            )

            self.summary_writer.file_writer.add_summary(
                Summary(value=[Summary.Value(tag=writer_prefix, image=video)]),
                global_step,
            )

    @torch.no_grad()
    def log_box_gif(
        self,
        box_db,
        writer_prefix: str,
        global_step: int,
        confidence_threshold: float = 0.0,
    ) -> None:
        pcl_imgs = np.array(self.pcl_npy_imgs_f32).copy()
        for time_idx, sample_id in enumerate(self.sample_ids):
            if sample_id in box_db:
                boxes = Shape(**box_db[sample_id]["raw_box"]).to_tensor()
                boxes.valid = boxes.valid & (
                    torch.squeeze(boxes.probs, dim=-1) >= confidence_threshold
                )
                boxes = boxes.drop_padding_boxes()
                pcl_img_f32 = pcl_imgs[time_idx][None, ...]  # add batch_dim
                pcl_img_w_box = draw_box_onto_image(
                    boxes[None],  # add batch dim
                    pcl_img_f32,
                    bev_range_m=self.bev_range_m,
                    color="confidence",
                )
                pcl_imgs[time_idx] = pcl_img_w_box[0]

        self.log_gif(pcl_imgs, writer_prefix, global_step)


def track_boxes_on_data_sequence(
    *,
    cfg: Dict,
    dataset: LidarDataset,
    box_predictor: BoxLearner,
    tracking_cfg: Dict[str, float],
    writer: SummaryWriter = None,
    min_num_boxes: int = None,  # abort after this many boxes have been tracked
    timeout_s: int = None,
    global_step: int = None,
    writer_prefix: str = "",
    verbose=False,
    log_freq=None,
    export_raw_tracked_detections_to: str = "",
    export_detections_only_in_annotated_fov=False,
    max_augm_db_size_mb=200,
    log_gifs_to_disk=False,
    dump_sequences_for_visu=False,
):
    if min_num_boxes is None:
        min_num_boxes = np.iinfo(np.uint64).max
    if log_freq is None:
        if verbose:
            log_freq = 1
        else:
            log_freq = {"waymo": 10, "kitti": 10, "nuscenes": 20, "av2": 20}[
                cfg.data.source
            ]
    if writer is not None:
        assert global_step is not None, global_step
    if timeout_s is None:
        timeout_s = float("inf")

    align_predicted_boxes_using_flow = cfg.data.tracking_cfg.setdefault(
        "align_predicted_boxes_using_flow", False
    )

    num_successfull_tracks = 0
    num_tracked_sequences = 0
    taboo_dataset_indexes = set()
    min_track_obj_speed_mps = 0.0

    if (
        isinstance(box_predictor, FlowClusterDetector)
        and cfg.data.tracking_cfg.tracker_model != "None"
    ):
        # FlowClusterDetector cannot detect still objects anyway
        # noisy SLIM flow creates FP clusters
        min_track_obj_speed_mps = tracking_cfg.flow_cluster_detector_min_obj_speed_mps

    if export_raw_tracked_detections_to:
        tracked_boxes_conf_stats = {}
        tracked_boxes_db = {}
    gt_boxes_db = {}
    timeout_at = time.time() + timeout_s
    box_points_snippets_db = get_empty_augm_box_db()
    max_track_id = 0
    if hasattr(dataset, "sequence_lens"):
        max_tqdm_count = len(dataset.sequence_lens)
    else:
        max_tqdm_count = min_num_boxes

    print(f"{datetime.now()} start tracking at step {global_step}.")

    if dump_sequences_for_visu:
        assert export_raw_tracked_detections_to is not None
        # this is will export these sequences for blender rendering
        waymo_plot_sequences = [
            "segment-17752423643206316420",
            "segment-1265122081809781363",
            "segment-9696413700515401320",
        ]

        intersting_kitti_sequences = [
            "2011_09_28_0038",
            "2011_09_26_0005",
            "2011_09_26_0018",
        ]
        interesting_nusc_sequence_ids = ["scene-0019", "scene-0181", "scene-0158"]

        interesting_sequences = {
            "waymo": waymo_plot_sequences,
            "kitti": intersting_kitti_sequences,
            "nusces": interesting_nusc_sequence_ids,
        }

        visualize_these_sequences = interesting_sequences[cfg.data.source]
    time_between_frames_s = [0.1, 0.5][isinstance(dataset, NuscenesDataset)]
    cuda0 = torch.device("cuda:0")

    with tqdm(total=max_tqdm_count, disable=False) as pbar:
        while num_successfull_tracks < min_num_boxes and time.time() < timeout_at:
            raw_boxes_db = {}
            visu_boxes_db = {}
            trigger_gif_logging = (
                writer is not None and num_tracked_sequences % log_freq == 0
            )
            trigger_img_logging = (
                writer is not None and num_tracked_sequences % log_freq == 0
            )
            if trigger_gif_logging:
                pcl_box_gif_summary = PCLBoxGifSummary(
                    cfg,
                    writer,
                    img_grid_size=(1024, 1024),
                    time_between_frames_s=time_between_frames_s,
                    log_to_disk=log_gifs_to_disk,
                )
            if dump_sequences_for_visu:
                if len(visualize_these_sequences) == 0:
                    seq = None
                else:
                    seq_id = visualize_these_sequences.pop()
                    seq = dataset.get_consecutive_sample_idxs_for_sequence(
                        dataset.get_scene_index_for_scene_name(seq_id),
                    )
            else:
                seq = dataset.get_consecutive_sample_idxs_for_sequence(
                    num_tracked_sequences
                )
            if seq is None:
                print("Ran out of sequences, stopping!")
                break
            dataset_idxs = [el.idx for el in seq]
            if any(ds_idx in taboo_dataset_indexes for ds_idx in dataset_idxs):
                num_sequences_visited = len(taboo_dataset_indexes)
                print(
                    f"Prevented revisiting of sequence! Visited {num_sequences_visited} sequences in total."
                )
                continue
            else:
                taboo_dataset_indexes.update(dataset_idxs)
            subset = torch.utils.data.Subset(dataset, dataset_idxs)
            prefetch_args = {}

            subset_loader = torch.utils.data.DataLoader(
                subset,
                pin_memory=True,
                batch_size=1,
                num_workers=cfg.data.num_workers,
                collate_fn=lidar_dataset_collate_fn,
                shuffle=False,
                worker_init_fn=worker_init_fn,
                **prefetch_args,
            )

            tracker_box_matching_threshold = cfg.data.tracking_cfg.setdefault(
                "track_matching_threshold_m", 1.0
            )

            tracker_model_name = cfg.data.tracking_cfg.setdefault(
                "tracker_model", "flow_tracker"
            )
            use_pred_future_box_poses_for_matching = cfg.data.tracking_cfg.setdefault(
                "use_pred_future_box_poses_for_matching", True
            )
            if tracker_model_name == "flow_tracker":
                simple_tracker = FlowBasedBoxTracker(
                    use_propagated_boxes=use_pred_future_box_poses_for_matching,
                    box_matching_threshold_m=tracker_box_matching_threshold,
                    association_strategy="ours",
                )
                gt_tracker = FlowBasedBoxTracker(
                    use_propagated_boxes=use_pred_future_box_poses_for_matching,
                    box_matching_threshold_m=tracker_box_matching_threshold,
                    association_strategy="ours",
                )
            elif tracker_model_name == "None":
                simple_tracker = NotATracker()
                gt_tracker = NotATracker()
            else:
                raise NotImplementedError(tracker_model_name)
            point_clouds_sensor_cosy = []
            point_cloud_row_idxs = []
            sample_ids_in_seq = []
            odoms_t0_t1 = []
            for time_idx, data_el in enumerate(tqdm(subset_loader, disable=False)):
                sample_data_t0, _, _, meta = data_el
                sample_ids = meta["sample_id"]
                assert len(sample_ids) == 1, "batch size 1 required"
                assert sample_ids[0] == seq[time_idx].sample_name, (
                    sample_ids[0],
                    seq[time_idx].sample_name,
                )
                sample_id = sample_ids[0]
                sample_ids_in_seq.append(sample_id)
                network_input_pcls_ta = get_network_input_pcls(
                    cfg,
                    sample_data_t0,
                    time_key="ta",
                    to_device=cuda0,
                )

                pcl_no_ground = sample_data_t0["pcl_ta"]["pcl"][0].to(cuda0)
                if isinstance(box_predictor, (FlowClusterDetector,)):
                    pred_boxes = box_predictor(
                        sample_data_t0,
                        writer=writer,
                        writer_prefix=writer_prefix,
                        global_step=global_step + num_successfull_tracks,
                    )
                    pred_boxes = pred_boxes.to(cuda0)
                else:
                    if cfg.network.name == "echo_gt":
                        gt_echo_boxes = sample_data_t0["gt"]["boxes"].to(cuda0)
                    else:
                        gt_echo_boxes = None
                    pred_boxes, _, _, _ = box_predictor(
                        img_t0=None,
                        pcls_t0=network_input_pcls_ta,
                        gt_boxes=gt_echo_boxes,
                        centermaps_gt=None,
                        train=False,
                    )
                    del gt_echo_boxes
                    if (
                        cfg.box_prediction.activations.probs == "none"
                        and not isinstance(
                            box_predictor,
                            (FlowClusterDetector,),
                        )
                        and not (
                            isinstance(box_predictor, BoxLearner)
                            and isinstance(
                                box_predictor.model,
                                (PointPillarsWrapper, PointRCNNWrapper),
                            )
                        )
                    ):
                        pred_boxes.probs = torch.sigmoid(pred_boxes.probs)
                pred_boxes = pred_boxes[0]

                nms_pred_box_idxs = iou_based_nms(
                    pred_boxes,
                    overlap_threshold=cfg.nms_iou_threshold,
                    pre_nms_max_boxes=tracking_cfg.max_num_boxes_before_nms,
                    post_nms_max_boxes=tracking_cfg.max_num_boxes_after_nms,
                )
                pred_boxes = pred_boxes[nms_pred_box_idxs].detach()

                assert len(pred_boxes.shape) == 1, "batching not supported"

                if (
                    pred_boxes.shape[0] > 0
                    and tracking_cfg.drop_boxes_on_bev_boundaries
                ):
                    is_box_fully_visible_in_bev = is_boxes_clearly_in_bev_range(
                        pred_boxes,
                        bev_range_m=torch.tensor(
                            cfg.data.bev_range_m, device=pred_boxes.pos.device
                        ),
                    )
                    pred_boxes.valid = is_box_fully_visible_in_bev
                    if verbose:
                        print(
                            "Dropped ",
                            torch.count_nonzero(~is_box_fully_visible_in_bev)
                            .cpu()
                            .numpy(),
                            "/",
                            pred_boxes.shape[0],
                            " boxes outside of BEV at time ",
                            time_idx,
                        )
                    pred_boxes = pred_boxes.drop_padding_boxes()

                if pred_boxes.shape[0] > 0 and tracking_cfg.min_points_in_box > 0:
                    pred_boxes_for_num_points_filtering = pred_boxes.clone()
                    if pred_boxes.dims.shape[-1] == 2:
                        dummy_height = 2.0 * torch.ones_like(pred_boxes.dims[..., [0]])
                        dummy_dims = torch.cat(
                            [
                                pred_boxes_for_num_points_filtering.dims,
                                dummy_height,
                            ],
                            dim=-1,
                        )
                        pred_boxes_for_num_points_filtering.dims = dummy_dims
                    if pred_boxes.pos.shape[-1] == 2:
                        dummy_z_coord = -1.0 * torch.ones_like(
                            pred_boxes.dims[..., [0]]
                        )
                        dummy_pos = torch.cat(
                            [
                                pred_boxes_for_num_points_filtering.pos,
                                dummy_z_coord,
                            ],
                            dim=-1,
                        )
                        pred_boxes_for_num_points_filtering.pos = dummy_pos

                    point_is_in_boxes_mask = get_points_in_boxes_mask(
                        pred_boxes_for_num_points_filtering,
                        homogenize_pcl(pcl_no_ground[:, :3]),
                    )
                    num_points_in_box = point_is_in_boxes_mask.sum(dim=0)
                    box_has_enough_points = (
                        num_points_in_box >= tracking_cfg.min_points_in_box
                    )
                    if verbose:
                        print(
                            "Dropped ",
                            torch.count_nonzero(~box_has_enough_points).cpu().numpy(),
                            "/",
                            pred_boxes.shape[0],
                            " boxes with less than ",
                            tracking_cfg.min_points_in_box,
                            " points at time: ",
                            time_idx,
                        )
                    # print(num_points_in_box[box_has_enough_points])
                    pred_boxes.valid = box_has_enough_points
                    pred_boxes = pred_boxes.drop_padding_boxes()
                # SECTION export detections
                box_is_in_annotated_fov = torch.ones_like(pred_boxes.valid)
                if export_raw_tracked_detections_to:
                    batch_idx = 0
                    raw_export_boxes = pred_boxes.clone()
                    if export_detections_only_in_annotated_fov and isinstance(
                        dataset, (KittiTrackingDataset, KittiObjectDataset)
                    ):
                        # filter detections that fell into areas that have no labels
                        box_is_in_annotated_fov = (
                            count_box_points_in_kitti_annotated_fov(
                                raw_export_boxes,
                                sample_data_t0["pcl_full_ta"][batch_idx].to(cuda0),
                            )
                            >= tracking_cfg.min_points_in_box
                        )
                        raw_export_boxes.valid = (
                            box_is_in_annotated_fov & raw_export_boxes.valid
                        )
                        raw_export_boxes = raw_export_boxes.drop_padding_boxes()
                    assert box_is_in_annotated_fov.shape == pred_boxes.shape, (
                        box_is_in_annotated_fov.shape,
                        pred_boxes.shape,
                    )

                    num_raw_export_boxes = raw_export_boxes.shape[0]
                    max_confidence_in_sample = float("-inf")
                    if num_raw_export_boxes > 0:
                        assert (
                            sample_id not in raw_boxes_db
                        ), f"overwriting occurs for {sample_id}!"
                        raw_boxes_db[sample_id] = {
                            "lidar_T_box": raw_export_boxes.get_poses().cpu().numpy(),
                            "raw_box": raw_export_boxes.cpu().numpy().__dict__,
                        }
                        max_confidence_in_sample = float(
                            raw_export_boxes.probs.cpu().numpy().max()
                        )
                    elif verbose:
                        print(f"No RAW boxes found in {sample_id} - skipping")

                if dump_sequences_for_visu:
                    maybe_gt_boxes = sample_data_t0.get("gt", {}).get("boxes", None)
                    if maybe_gt_boxes is not None:
                        maybe_gt_boxes = (
                            maybe_gt_boxes[batch_idx]
                            .clone()
                            .drop_padding_boxes()
                            .cpu()
                            .numpy()
                            .__dict__
                        )

                    pcl_no_ground_for_visu = (
                        sample_data_t0["pcl_ta"]["pcl"][0].clone().cpu().numpy()
                    )
                    flow_no_ground = (
                        sample_data_t0[cfg.data.flow_source]["flow_ta_tb"][batch_idx]
                        .clone()
                        .cpu()
                        .numpy()
                    )
                    dyn_flow_no_ground = (
                        flow_no_ground
                        - np.einsum(
                            "ij,nj->ni",
                            sample_data_t0[cfg.data.odom_source]["odom_tb_ta"][
                                batch_idx
                            ]
                            - np.eye(4),
                            np.concatenate(
                                [
                                    pcl_no_ground_for_visu[:, :3],
                                    np.ones_like(pcl_no_ground_for_visu[:, [0]]),
                                ],
                                axis=-1,
                            ),
                        )[:, :3]
                    )
                    full_pcl_with_ground = (
                        sample_data_t0["pcl_full_w_ground_ta"][batch_idx]
                        .clone()
                        .cpu()
                        .numpy()
                    )

                    kdt = KDTree(n_neighbors=1, metric="L2", leaf_size=20)
                    kdt.fit(pcl_no_ground_for_visu[:, :3])
                    dist, pcl_knn_idxs = kdt.kneighbors(full_pcl_with_ground[:, :3])
                    dist = np.squeeze(dist, axis=-1)
                    pcl_knn_idxs = np.squeeze(pcl_knn_idxs, axis=-1)
                    dist_too_far = dist > 0.1
                    dyn_flow_w_ground = np.zeros(
                        (full_pcl_with_ground.shape[0], 3), dtype=np.float32
                    )
                    dyn_flow_w_ground[~dist_too_far] = dyn_flow_no_ground[
                        pcl_knn_idxs[~dist_too_far]
                    ]

                    visu_boxes_db[sample_id] = {
                        "flow": dyn_flow_w_ground.astype(np.float32),
                        "points_xyzi": sample_data_t0["pcl_full_w_ground_ta"][batch_idx]
                        .clone()
                        .cpu()
                        .numpy()
                        .astype(np.float32),
                        "pred": {"boxes": pred_boxes.clone().cpu().numpy().__dict__},
                        "gt": {
                            "boxes": maybe_gt_boxes,
                        },
                    }

                # END SECTION export detections
                pred_boxes = pred_boxes[None]
                point_cloud_ta = sample_data_t0["pcl_ta"]["pcl"][..., :3].to(cuda0)
                valid_mask_ta = sample_data_t0["pcl_ta"]["pcl_is_valid"].to(cuda0)

                (
                    odom_ta_tb,
                    pointwise_flow_ta_tb,
                    gt_odom_ta_tb,
                    gt_flow_ta_tb,
                ) = get_odometry_gt_odmetry_and_flow(
                    dataset, cfg, sample_data_t0, cuda0
                )

                (
                    fg_kabsch_trafos_t0_t1,
                    odom_t0_t1,
                    bg_kabsch_trafo_t0_t1,
                    _,
                    st1_T_pred_bt1,
                ) = propagate_boxes_forward_using_flow(
                    pred_boxes,
                    point_cloud_ta,
                    valid_mask_ta,
                    pointwise_flow_ta_tb=pointwise_flow_ta_tb,
                    odom_t0_t1=odom_ta_tb,
                    device=cuda0,
                )

                (
                    _,
                    _,
                    _,
                    _,
                    st_minus_1_T_pred_bt_minus1,
                ) = propagate_boxes_forward_using_flow(
                    pred_boxes,
                    point_cloud_ta,
                    valid_mask_ta,
                    pointwise_flow_ta_tb=-1.0 * pointwise_flow_ta_tb,
                    odom_t0_t1=torch.linalg.inv(odom_ta_tb),
                    device=cuda0,
                )

                if align_predicted_boxes_using_flow and not isinstance(
                    box_predictor, (FlowClusterDetector,)
                ):
                    pred_boxes = soft_align_box_flip_orientation_with_motion_trafo(
                        boxes=pred_boxes,
                        fg_kabsch_trafos=fg_kabsch_trafos_t0_t1,
                        bg_kabsch_trafo=bg_kabsch_trafo_t0_t1,
                    )

                pred_boxes = pred_boxes[0].cpu()

                odom_t0_t1 = odom_t0_t1.detach().cpu()

                pcl_no_ground_sensor_cosy = pcl_no_ground.detach()
                point_clouds_sensor_cosy.append(pcl_no_ground_sensor_cosy.cpu())
                point_cloud_row_idxs.append(
                    sample_data_t0["lidar_rows_ta"][0].detach().cpu()
                )
                odoms_t0_t1.append(odom_t0_t1.detach().cpu())
                assert len(pred_boxes.pos.shape) == 2
                assert len(pred_boxes.rot.shape) == 2
                assert odom_t0_t1.shape == (4, 4), odom_t0_t1.shape
                assert odom_t0_t1.dtype == torch.float64, odom_t0_t1.dtype

                if export_raw_tracked_detections_to:
                    per_box_extra_attributes = []
                    box_is_in_annotated_fov = box_is_in_annotated_fov.cpu().numpy()
                    for box_idx in range(pred_boxes.shape[0]):
                        per_box_extra_attributes.append(
                            {
                                "is_in_annotated_fov": box_is_in_annotated_fov[box_idx],
                                "sample_id": sample_id,
                            }
                        )
                else:
                    per_box_extra_attributes = [
                        None,
                    ] * pred_boxes.shape[0]

                simple_tracker.update(
                    pred_boxes,
                    predicted_box_poses_stiii=st1_T_pred_bt1[0].cpu(),
                    predicted_box_poses_sti=st_minus_1_T_pred_bt_minus1[0].cpu(),
                    odom_stii_stiii=odom_t0_t1,
                    per_box_extra_attributes_tii=per_box_extra_attributes,
                )
                gt_boxes = sample_data_t0.get("gt", {}).get("boxes", None)
                if gt_boxes is not None:
                    gt_boxes_db[sample_id] = {
                        "raw_box": gt_boxes[0].detach().cpu().numpy().__dict__
                    }
                if trigger_gif_logging:
                    pcl_box_gif_summary.update_pcl_npy_imgs(
                        pcl_no_ground_sensor_cosy, sample_id
                    )
                if trigger_img_logging:
                    if gt_boxes:
                        gt_boxes = gt_boxes.to(cuda0)
                    else:
                        gt_boxes = Shape.createEmpty().to_tensor().to(cuda0)
                    (
                        _,
                        _,
                        _,
                        _,
                        st1_T_gt_pred_bt1,
                    ) = propagate_boxes_forward_using_flow(
                        gt_boxes,
                        point_cloud_ta,
                        valid_mask_ta,
                        gt_flow_ta_tb.to(cuda0),
                        odom_t0_t1=gt_odom_ta_tb,
                        device=cuda0,
                    )
                    (
                        _,
                        _,
                        _,
                        _,
                        st_minus_1_T_gt_bt_minus1,
                    ) = propagate_boxes_forward_using_flow(
                        gt_boxes,
                        point_cloud_ta,
                        valid_mask_ta,
                        pointwise_flow_ta_tb=-1.0 * gt_flow_ta_tb.to(cuda0),
                        odom_t0_t1=torch.linalg.inv(gt_odom_ta_tb),
                        device=cuda0,
                    )
                    gt_boxes = gt_boxes[0].detach().cpu()
                    gt_tracker.update(
                        gt_boxes,
                        st1_T_gt_pred_bt1[0].detach().cpu(),
                        st_minus_1_T_gt_bt_minus1[0].detach().cpu(),
                        gt_odom_ta_tb.detach().cpu(),
                        per_box_extra_attributes_tii=[
                            None,
                        ]
                        * gt_boxes.shape[0],
                    )
            if trigger_img_logging:
                gt_tracker.run_tracker()
            # up to here
            simple_tracker.run_tracker()
            if tracker_model_name != "None":
                (
                    longest_track_ids,
                    track_ages,
                ) = simple_tracker.get_ids_lengths_of_longest_tracks()
                boxes_world_w_Ts_box = simple_tracker.get_boxes_in_world_coordinates()
                boxes_sensor_Ts_box = (
                    simple_tracker.get_boxes_in_sensor_coordinates_at_each_timestamp()
                )
                w_T_sensor_poses_ti = aggregate_odometry_to_world_poses(
                    simple_tracker.sti_T_stii
                )
                extra_attrs_box = (
                    simple_tracker.get_extra_attributes_at_each_timestamp()
                )
                keep_these_track_ids_timestamps_boxes_extra_attrs = {
                    "world_raw": {},
                    "world_refined": {},
                    "sensor_raw": {},
                    "sensor_refined": {},
                    "extra_attributes": {},
                }

                box_data_for_smoothing = []
                for track_id, track_age in zip(longest_track_ids, track_ages):
                    if track_age >= tracking_cfg.min_track_age:
                        (
                            box_indices_for_track_id,
                            start_time_idx,
                        ) = simple_tracker.get_box_indices_start_time_for_track_id(
                            track_id
                        )
                        box_sequence_world_for_track_id = Shape.from_list_of_shapes(
                            [
                                boxes_world_w_Ts_box[start_time_idx + time_step][
                                    box_idx
                                ]
                                for time_step, box_idx in enumerate(
                                    box_indices_for_track_id
                                )
                            ]
                        )
                        if (
                            torch.median(box_sequence_world_for_track_id.probs)
                            < cfg.optimization.rounds.confidence_threshold_mined_boxes
                        ):
                            # don't even bother with refining it, since we will filter it away anyways
                            continue

                        (
                            keep,
                            dist_covered_by_this_track_m,
                        ) = decide_keep_or_drop_box(
                            tracking_cfg=tracking_cfg,
                            box_sequence_world_for_specific_track_id=box_sequence_world_for_track_id,
                            min_track_obj_speed_mps=min_track_obj_speed_mps,
                            time_between_frames_s=time_between_frames_s,
                            track_id=track_id,
                            verbose=verbose,
                            is_flow_cluster_detector=isinstance(
                                box_predictor, FlowClusterDetector
                            ),
                        )

                        if keep:
                            box_sequence_sensor_for_track_id = (
                                Shape.from_list_of_shapes(
                                    [
                                        boxes_sensor_Ts_box[start_time_idx + time_step][
                                            box_idx
                                        ]
                                        for time_step, box_idx in enumerate(
                                            box_indices_for_track_id
                                        )
                                    ]
                                )
                            )
                            keep_these_track_ids_timestamps_boxes_extra_attrs[
                                "sensor_raw"
                            ][
                                (int(track_id), int(start_time_idx))
                            ] = box_sequence_sensor_for_track_id.clone()
                            keep_these_track_ids_timestamps_boxes_extra_attrs[
                                "world_raw"
                            ][
                                (int(track_id), int(start_time_idx))
                            ] = box_sequence_world_for_track_id.clone()

                            box_sequence_sensor_for_track_id = perform_local_box_refinement(
                                cfg,
                                box_predictor,
                                point_clouds_sensor_cosy=point_clouds_sensor_cosy,
                                box_sequence_in_sensor_cosy_for_specific_track_id=box_sequence_sensor_for_track_id,
                                track_age=track_age,
                                start_time_idx=start_time_idx,
                            )

                            box_sequence_world_for_track_id = update_world_boxes_from_sensor_boxes(
                                box_sequence_sensor=box_sequence_sensor_for_track_id,
                                box_sequence_world=box_sequence_world_for_track_id,
                                w_T_sensor_ti=w_T_sensor_poses_ti[
                                    start_time_idx : start_time_idx + track_age
                                ],
                            )

                            # smooth confidence
                            median_box_confs = torch.median(
                                box_sequence_world_for_track_id.probs, dim=0
                            ).values
                            box_sequence_world_for_track_id.probs = (
                                median_box_confs
                                * torch.ones_like(box_sequence_world_for_track_id.probs)
                            )
                            min_dist_covered_by_track_for_smoothing = (
                                cfg.data.tracking_cfg.setdefault(
                                    "min_dist_for_track_smoothing", 5.0
                                )
                            )
                            extra_attributes_for_this_box = [
                                extra_attrs_box[start_time_idx + time_step][box_idx]
                                for time_step, box_idx in enumerate(
                                    box_indices_for_track_id
                                )
                            ]
                            if (
                                dist_covered_by_this_track_m
                                > min_dist_covered_by_track_for_smoothing
                                and tracker_model_name == "flow_tracker"
                                and cfg.data.tracking_cfg.flow_tracker.use_track_smoothing
                                and track_age
                                >= MIN_TRACK_LEN_FOR_SMOOTHING  # min track age for smoothing
                            ):
                                box_data_for_smoothing.append(
                                    BoxDataForSmoothing(
                                        track_id,
                                        track_age,
                                        start_time_idx,
                                        box_sequence_world_for_track_id,
                                        box_sequence_sensor_for_track_id,
                                        extra_attributes_for_this_box,
                                    )
                                )
                            else:
                                box_sequence_world_for_track_id.velo = (
                                    torch.ones_like(
                                        box_sequence_world_for_track_id.probs
                                    )
                                    * dist_covered_by_this_track_m
                                    / (track_age * time_between_frames_s)
                                )

                                update_db_with_this_box_stuff(
                                    keep_these_track_ids_timestamps_boxes_extra_attrs,
                                    track_id,
                                    start_time_idx,
                                    box_sequence_world_for_track_id,
                                    box_sequence_sensor_for_track_id,
                                    extra_attributes_for_this_box,
                                    w_T_sensor_poses_ti,
                                    track_age,
                                )
                if len(box_data_for_smoothing) > 0:
                    # print("smoothing ", len(box_data_for_smoothing), " tracks")
                    (
                        batched_observed_pos_m,
                        batched_observed_yaw_angle_rad,
                        batched_vehicle_length_m,
                        batched_valid_mask,
                    ) = batch_box_data_for_batched_smoothing(
                        [
                            bbox.box_sequence_world_for_track_id
                            for bbox in box_data_for_smoothing
                        ],
                        torch.device("cpu"),
                    )
                    if batched_valid_mask.shape[1] <= 4:
                        print("TRACKS ARE TOO SHORT - not optmizing")
                        batched_padded_smooth_pos = batched_observed_pos_m
                        batched_padded_smooth_yaw = batched_observed_yaw_angle_rad
                        batched_padded_smooth_velo = batched_displacement_from_pos(
                            batched_observed_pos_m
                        )[..., None]

                    else:
                        track_smoothing_method = (
                            cfg.data.tracking_cfg.flow_tracker.setdefault(
                                "track_smoothing_method", "jerk"
                            )
                        )
                        if track_smoothing_method == "bike_model":
                            (
                                batched_padded_smooth_pos,
                                batched_padded_smooth_yaw,
                                batched_padded_smooth_velo,
                            ) = smooth_track_bike_model(
                                batched_observed_pos_m=batched_observed_pos_m,
                                batched_observed_yaw_angle_rad=batched_observed_yaw_angle_rad,
                                batched_vehicle_length_m=batched_vehicle_length_m,
                                batched_valid_mask=batched_valid_mask,
                                time_between_frames_s=time_between_frames_s,
                                verbose=False,
                                return_losses=False,
                            )
                        elif track_smoothing_method == "jerk":
                            (
                                batched_padded_smooth_pos,
                                batched_padded_smooth_yaw,
                                batched_padded_smooth_velo,
                            ) = smooth_track_jerk(
                                batched_observed_pos_m=batched_observed_pos_m,
                                batched_observed_yaw_angle_rad=batched_observed_yaw_angle_rad,
                                batched_valid_mask=batched_valid_mask,
                                time_between_frames_s=time_between_frames_s,
                                verbose=False,
                                return_losses=False,
                            )
                        elif track_smoothing_method == "none":
                            batched_padded_smooth_pos = batched_observed_pos_m
                            batched_padded_smooth_yaw = batched_observed_yaw_angle_rad
                            batched_padded_smooth_velo = batched_displacement_from_pos(
                                batched_observed_pos_m
                            )[..., None]
                        else:
                            raise NotImplementedError(track_smoothing_method)

                    batched_smooth_pos = split_batched_padded_tensor_into_list(
                        batched_padded_smooth_pos.detach().cpu(),
                        batched_valid_mask.detach().cpu(),
                    )
                    batched_smooth_yaw = split_batched_padded_tensor_into_list(
                        batched_padded_smooth_yaw.detach().cpu(),
                        batched_valid_mask.detach().cpu(),
                    )
                    batched_smooth_velo = split_batched_padded_tensor_into_list(
                        batched_padded_smooth_velo.detach().cpu(),
                        batched_valid_mask.detach().cpu(),
                    )

                    for i, bbox in enumerate(box_data_for_smoothing):
                        bbox.box_sequence_world_for_track_id.pos = batched_smooth_pos[i]
                        bbox.box_sequence_world_for_track_id.rot = batched_smooth_yaw[i]
                        bbox.box_sequence_world_for_track_id.velo = batched_smooth_velo[
                            i
                        ]
                        update_db_with_this_box_stuff(
                            keep_these_track_ids_timestamps_boxes_extra_attrs,
                            bbox.track_id,
                            bbox.start_time_idx,
                            bbox.box_sequence_world_for_track_id,
                            bbox.box_sequence_sensor_for_track_id,
                            bbox.extra_attributes_for_this_box,
                            w_T_sensor_poses_ti,
                            bbox.track_age,
                        )

                if trigger_img_logging:
                    sequence_id = get_sequence_id(cfg, meta)[0]
                    img_grid_size = torch.tensor(
                        (1024, 1024),
                        dtype=torch.long,
                    )

                    pcl_accum = accumulate_pcl(point_clouds_sensor_cosy, odoms_t0_t1)
                    min_pcl_extent = pcl_accum.min(dim=0)[0][:2]  # only need x and y
                    max_pcl_extent = pcl_accum.max(dim=0)[0][:2]  # only need x and y
                    torch_bev_extent_m = torch.concat(
                        [min_pcl_extent, max_pcl_extent], dim=0
                    )
                    assert (
                        pcl_accum.shape[-1] == 4
                    ), pcl_accum.shape  # need intensity attribute
                    _, pixel_occup_mask = create_topdown_f32_pcl_image_variable_extent(
                        pcl_accum,
                        pcl_accum[:, -1],
                        min_pcl_extent,
                        max_pcl_extent,
                        img_grid_size,
                    )
                    pcl_img_bw_f32 = pixel_occup_mask[..., None] * torch.ones(
                        (3,), dtype=torch.float32, device=pixel_occup_mask.device
                    )
                    pcl_img_bw_uint8 = (
                        (255 * pcl_img_bw_f32.clone()).cpu().numpy().astype(np.uint8)
                    )

                    writer.add_text(
                        writer_prefix + "/tracking/sequence_id",
                        sequence_id,
                        global_step=global_step + num_successfull_tracks,
                    )
                    writer.add_image(
                        writer_prefix + "/tracking/accum_pcl_old",
                        pcl_img_bw_uint8,
                        global_step=global_step + num_successfull_tracks,
                        dataformats="HWC",
                    )

                    pcl_img_all_tracks_f32 = (
                        pcl_img_bw_f32.clone().detach().cpu().numpy()[None, ...]
                    )
                    pcl_img_all_tracks_f32, colors = draw_colored_tracks_onto_image(
                        simple_tracker, torch_bev_extent_m, pcl_img_all_tracks_f32
                    )

                    writer.add_image(
                        writer_prefix + "/tracking/all_box_detections_raw",
                        (255 * pcl_img_all_tracks_f32[0, ...]).astype(np.uint8),
                        global_step=global_step + num_successfull_tracks,
                        dataformats="HWC",
                    )

                    pcl_img_gt_tracks_f32, _ = draw_colored_tracks_onto_image(
                        gt_tracker,
                        torch_bev_extent_m,
                        pcl_img_all_tracks_f32=pcl_img_bw_f32.clone()
                        .detach()
                        .cpu()
                        .numpy()[None, ...],
                    )

                    writer.add_image(
                        writer_prefix + "/tracking/all_gt_boxes_tracks",
                        (255 * pcl_img_gt_tracks_f32[0, ...]).astype(np.uint8),
                        global_step=global_step + num_successfull_tracks,
                        dataformats="HWC",
                    )

                    pcl_img_selected_tracks_f32 = (
                        pcl_img_bw_f32.clone().detach().cpu().numpy()[None, ...]
                    )
                    for (
                        (track_id, _),
                        track_boxes_world,
                    ) in keep_these_track_ids_timestamps_boxes_extra_attrs[
                        "world_raw"
                    ].items():
                        batched_box_colors = colors[track_id, :3][None, None, ...]
                        num_boxes = track_boxes_world.pos.shape[0]
                        batched_box_colors = np.tile(
                            batched_box_colors, (1, num_boxes, 1)
                        )
                        pcl_img_selected_tracks_f32 = draw_box_onto_image(
                            track_boxes_world[None],
                            pcl_img_selected_tracks_f32,
                            bev_range_m=torch_bev_extent_m,
                            color=batched_box_colors,
                        )
                    writer.add_image(
                        writer_prefix + "/tracking/selected_box_tracks_raw",
                        (255 * pcl_img_selected_tracks_f32[0, ...]).astype(np.uint8),
                        global_step=global_step + num_successfull_tracks,
                        dataformats="HWC",
                    )

                    confidence_text_target_canvas = (
                        255.0 * pcl_img_selected_tracks_f32[0, ...]
                    ).astype(np.uint8)
                    target_canvas_size = torch.tensor(
                        confidence_text_target_canvas.shape[:-1]
                    )
                    for (
                        track_boxes_world
                    ) in keep_these_track_ids_timestamps_boxes_extra_attrs[
                        "world_raw"
                    ].values():
                        plot_these_scalars = np.squeeze(
                            track_boxes_world.probs, axis=-1
                        )
                        text_pos = project_2d_pcl_to_rowcol_nonsquare_bev_range(
                            pcl_2d=track_boxes_world.pos[:, :2],
                            coords_min=torch_bev_extent_m[:2],
                            coords_max=torch_bev_extent_m[2:],
                            img_grid_size=target_canvas_size,
                        )
                        first_text_pos = text_pos[0]
                        last_text_pos = text_pos[-1]
                        min_pixel_dist_per_text = 5
                        downsample_plotted_confs = torch.linalg.norm(
                            first_text_pos - last_text_pos
                        ) < (min_pixel_dist_per_text * len(text_pos))
                        if downsample_plotted_confs:
                            min_conf = plot_these_scalars.min()
                            max_conf = plot_these_scalars.max()
                            median_conf = plot_these_scalars.median()
                            plot_these_texts = [
                                "Min: %.2f" % min_conf,
                                "Max: %.2f" % max_conf,
                                "Median: %.2f" % median_conf,
                            ]
                            ten_pixel_offset = torch.tensor([10.0, 0.0])
                            middle_pos = text_pos[len(text_pos) // 2]
                            text_pos = torch.stack(
                                [
                                    middle_pos - ten_pixel_offset,
                                    middle_pos,
                                    middle_pos + ten_pixel_offset,
                                ]
                            )
                        else:
                            plot_these_texts = [
                                "%.2f" % scalar for scalar in plot_these_scalars
                            ]

                        confidence_text_target_canvas = plot_text_on_canvas_at_position(
                            text_pos=text_pos.numpy(),
                            texts=plot_these_texts,
                            target_canvas_channels_last_uint8=confidence_text_target_canvas,
                        )

                    writer.add_image(
                        writer_prefix
                        + "/tracking/selected_box_tracks_raw_with_confidence",
                        confidence_text_target_canvas,
                        global_step=global_step + num_successfull_tracks,
                        dataformats="HWC",
                    )

                    pcl_img_refined_tracks_f32 = (
                        pcl_img_bw_f32.clone().detach().cpu().numpy()[None, ...]
                    )

                    for (
                        (track_id, _),
                        track_boxes_world,
                    ) in keep_these_track_ids_timestamps_boxes_extra_attrs[
                        "world_refined"
                    ].items():
                        batched_box_colors = colors[track_id, :3][None, None, ...]
                        num_boxes = track_boxes_world.pos.shape[0]
                        batched_box_colors = np.tile(
                            batched_box_colors, (1, num_boxes, 1)
                        )
                        pcl_img_refined_tracks_f32 = draw_box_onto_image(
                            track_boxes_world[None],
                            pcl_img_refined_tracks_f32,
                            bev_range_m=torch_bev_extent_m,
                            color=batched_box_colors,
                        )

                    writer.add_image(
                        writer_prefix + "/tracking/selected_box_tracks_refined",
                        (255 * pcl_img_refined_tracks_f32[0, ...]).astype(np.uint8),
                        global_step=global_step + num_successfull_tracks,
                        dataformats="HWC",
                    )

                if export_raw_tracked_detections_to:
                    per_sample_id_export_boxes = defaultdict(defaultdict(list).copy)
                for (
                    track_id,
                    start_time_idx,
                ), boxes_sensor_ti in keep_these_track_ids_timestamps_boxes_extra_attrs[
                    "sensor_refined"
                ].items():
                    boxes_world_ti = keep_these_track_ids_timestamps_boxes_extra_attrs[
                        "world_refined"
                    ][(track_id, start_time_idx)]
                    track_len = boxes_sensor_ti.shape[0]
                    assert torch.allclose(
                        boxes_world_ti.dims, boxes_sensor_ti.dims
                    ), "error: dims change with coordinate system"
                    assert boxes_world_ti.shape == boxes_sensor_ti.shape, (
                        track_id,
                        boxes_world_ti.shape,
                        boxes_sensor_ti.shape,
                    )
                    dist_covered_by_track_m = np.linalg.norm(
                        boxes_world_ti.pos[-1] - boxes_world_ti.pos[0]
                    )
                    # short time tracks: uncertain, sample less often
                    # long dist tracks: very certain
                    assert track_len >= tracking_cfg.min_track_age
                    num_samples_to_keep_from_this_track = (
                        track_len // tracking_cfg.min_track_age
                    ) * int(dist_covered_by_track_m)
                    max_saves_for_this_track = min(10, track_len)

                    num_samples_to_keep_from_this_track = min(
                        max(1, num_samples_to_keep_from_this_track),
                        max_saves_for_this_track,
                    )
                    # num_skip = len(point_clouds) - track_len
                    pcl_time_idxs = np.random.choice(
                        np.arange(
                            start=int(start_time_idx),
                            stop=int(track_len + start_time_idx),
                            step=1,
                        ),
                        size=num_samples_to_keep_from_this_track,
                        replace=False,
                    )
                    unique_track_id = max_track_id
                    max_track_id += 1
                    for pcl_time_idx in pcl_time_idxs:
                        pcl_at_t = point_clouds_sensor_cosy[pcl_time_idx]
                        row_idxs_at_t = point_cloud_row_idxs[pcl_time_idx]
                        # print(pcl_at_t[:3])
                        # box sequence starts "later" than point cloud sequence - shift index accordingly:
                        box_at_t = boxes_sensor_ti[int(pcl_time_idx - start_time_idx)]

                        sensor_T_box = box_at_t[None].get_poses()[0]
                        box_T_sensor = torch.linalg.inv(sensor_T_box)
                        pcl_at_t_homog = homogenize_pcl(pcl_at_t[:, :3])
                        pcl_box = torch.cat(
                            [
                                torch.einsum(
                                    "ij,nj->ni",
                                    box_T_sensor,
                                    pcl_at_t_homog.double(),
                                )[:, :3].float(),
                                pcl_at_t[:, [-1]],
                            ],
                            dim=-1,
                        )
                        point_is_in_box = torch.all(
                            torch.abs(pcl_box[:, 0:3]) <= 1.1 * 0.5 * box_at_t.dims,
                            dim=-1,
                        )
                        num_points_in_box = torch.count_nonzero(point_is_in_box)
                        if num_points_in_box == 0:
                            # does not make sense to store empty boxes to do patch augmentation
                            continue

                        pcl_box = (
                            pcl_box[point_is_in_box].cpu().numpy().astype(np.float32)
                        )
                        rows_box = row_idxs_at_t[point_is_in_box].cpu().numpy()
                        box_points_snippets_db["pcl_in_box_cosy"].append(pcl_box)
                        box_points_snippets_db["boxes"].append(box_at_t)
                        box_points_snippets_db["box_T_sensor"].append(
                            box_T_sensor.numpy()
                        )
                        box_points_snippets_db["lidar_rows"].append(rows_box)
                        box_points_snippets_db["unique_track_id"].append(
                            unique_track_id
                        )
                    num_successfull_tracks += 1

                    if export_raw_tracked_detections_to:
                        tracked_box_attrs = list_of_dicts_to_dict_of_lists(
                            keep_these_track_ids_timestamps_boxes_extra_attrs[
                                "extra_attributes"
                            ][(track_id, start_time_idx)]
                        )
                        for track_timestamp_idx, file_name in enumerate(
                            tracked_box_attrs["sample_id"]
                        ):
                            tracked_box_is_in_annotated_fov = tracked_box_attrs[
                                "is_in_annotated_fov"
                            ][track_timestamp_idx]
                            save_tracked_box = (
                                tracked_box_is_in_annotated_fov
                                if export_detections_only_in_annotated_fov
                                else True
                            )
                            if save_tracked_box:
                                tracked_box_sample_id = tracked_box_attrs["sample_id"][
                                    track_timestamp_idx
                                ]
                                assert file_name == tracked_box_sample_id, (
                                    file_name,
                                    tracked_box_sample_id,
                                )
                                carryover_box = (
                                    keep_these_track_ids_timestamps_boxes_extra_attrs[
                                        "sensor_refined"
                                    ][(track_id, start_time_idx)][track_timestamp_idx]
                                ).numpy()
                                per_sample_id_export_boxes[tracked_box_sample_id][
                                    "sensor_refined"
                                ].append(carryover_box)
                                per_sample_id_export_boxes[tracked_box_sample_id][
                                    "track_id"
                                ].append(track_id)

                if export_raw_tracked_detections_to:
                    for (
                        export_sample_id,
                        tracked_box_attrs_for_export,
                    ) in per_sample_id_export_boxes.items():
                        num_export_tracked_boxes = len(
                            tracked_box_attrs_for_export["sensor_refined"]
                        )
                        max_confidence_in_sample = float("-inf")
                        if num_export_tracked_boxes > 0:
                            tmp_boxes = Shape.from_list_of_npy_shapes(
                                tracked_box_attrs_for_export["sensor_refined"]
                            )
                            tracked_boxes_db[export_sample_id] = {
                                "lidar_T_box": tmp_boxes.get_poses(),
                                "raw_box": tmp_boxes.__dict__,
                                "track_id": np.array(
                                    tracked_box_attrs_for_export["track_id"]
                                ),
                            }
                            max_confidence_in_sample = float(tmp_boxes.probs.max())
                        else:
                            print(
                                f"No tracked boxes found in {export_sample_id} - skipping"
                            )
                        assert (
                            export_sample_id not in tracked_boxes_conf_stats
                        ), f"overwriting occuring for sample: {export_sample_id}"
                        tracked_boxes_conf_stats[export_sample_id] = {
                            "max_confidence": max_confidence_in_sample,
                            "num_boxes": num_export_tracked_boxes,
                        }
                if dump_sequences_for_visu:
                    seq_target_dir = (
                        Path(export_raw_tracked_detections_to)
                        / "blender_visu_export"
                        / sequence_id
                    )
                    seq_target_dir.mkdir(exist_ok=True, parents=True)
                    print(f"Dumping exported sequence for blender to {seq_target_dir}")
                    for export_sample_id, export_payload in visu_boxes_db.items():
                        if export_sample_id in tracked_boxes_db:
                            export_payload["tracked"] = {
                                "boxes": tracked_boxes_db[export_sample_id]["raw_box"]
                            }
                        else:
                            export_payload["tracked"] = {
                                "boxes": Shape.createEmpty().numpy().__dict__
                            }
                        np.savez_compressed(
                            seq_target_dir / export_sample_id.replace("/", "_"),
                            export_payload,
                            allow_pickle=True,
                        )

                if trigger_gif_logging or trigger_img_logging:
                    sequence_id = get_sequence_id(cfg, meta)[0]
                if trigger_gif_logging:
                    confidence_threshold_box_gif_summary = cfg.logging.setdefault(
                        "confidence_threshold_box_gif_summary",
                        cfg.optimization.rounds.confidence_threshold_mined_boxes,
                    )
                    pcl_box_gif_summary.log_box_gif(
                        writer_prefix=writer_prefix + "/" + sequence_id + "_tracked",
                        global_step=global_step,
                        box_db=tracked_boxes_db,
                        confidence_threshold=confidence_threshold_box_gif_summary,
                    )
                    pcl_box_gif_summary.log_box_gif(
                        writer_prefix=writer_prefix + "/" + sequence_id + "_raw",
                        global_step=global_step,
                        box_db=raw_boxes_db,
                        confidence_threshold=confidence_threshold_box_gif_summary,
                    )
                    if len(gt_boxes_db) > 0 and log_freq > 5:
                        pcl_box_gif_summary.log_box_gif(
                            writer_prefix=writer_prefix + "/" + sequence_id + "_gt",
                            global_step=global_step,
                            box_db=gt_boxes_db,
                            confidence_threshold=confidence_threshold_box_gif_summary,
                        )
                if trigger_img_logging:
                    img_grid_size = torch.tensor(
                        (1024, 1024),
                        dtype=torch.long,
                    )
                    if num_tracked_sequences == 0:  # found_it:
                        # if found_it:
                        min_pcl_sensor_extent = -torch.tensor(cfg.data.bev_range_m) / 2
                        max_pcl_sensor_extent = torch.tensor(cfg.data.bev_range_m) / 2
                        assert len(sample_ids_in_seq) == len(
                            point_clouds_sensor_cosy
                        ), (
                            len(sample_ids_in_seq),
                            len(point_clouds_sensor_cosy),
                        )
                        for timestep, (sample_id, pcl_sensor_cosy) in enumerate(
                            zip(sample_ids_in_seq, point_clouds_sensor_cosy)
                        ):
                            if sample_id in raw_boxes_db:
                                raw_boxes = Shape(
                                    **raw_boxes_db[sample_id]["raw_box"]
                                ).to_tensor()
                            else:
                                raw_boxes = Shape.createEmpty().to_tensor()
                            raw_boxes = raw_boxes[None]
                            if sample_id in tracked_boxes_db:
                                refined_boxes = Shape(
                                    **tracked_boxes_db[sample_id]["raw_box"]
                                ).to_tensor()
                            else:
                                refined_boxes = Shape.createEmpty().to_tensor()
                            refined_boxes = refined_boxes[None]
                            (
                                _,
                                bev_occup_mask,
                            ) = create_topdown_f32_pcl_image_variable_extent(
                                pcl_sensor_cosy,
                                pcl_sensor_cosy[:, -1],
                                min_pcl_sensor_extent,
                                max_pcl_sensor_extent,
                                img_grid_size,
                            )
                            box_img = draw_box_image(
                                cfg=cfg,
                                gt_boxes=raw_boxes,
                                canvas_f32=bev_occup_mask[None, None, ...].float(),
                                gt_background_boxes=None,
                                pred_boxes=refined_boxes,
                            )

                            writer.add_image(
                                writer_prefix
                                + f"/tracking/raw_vs_refined_sensor/{sequence_id}",
                                box_img[0],
                                global_step=global_step
                                + num_successfull_tracks
                                + timestep,
                                dataformats="HWC",
                            )

            else:
                assert tracker_model_name == "None", tracker_model_name

                boxes_sensor_Ts_box = (
                    simple_tracker.get_boxes_in_sensor_coordinates_at_each_timestamp()
                )

                for pcl_time_idx in range(len(point_clouds_sensor_cosy)):
                    export_sample_id = sample_ids_in_seq[pcl_time_idx]
                    boxes_at_t = boxes_sensor_Ts_box[pcl_time_idx]
                    max_confidence_in_sample = float("-inf")
                    num_detected_boxes_in_sample = np.count_nonzero(boxes_at_t.valid)
                    if num_detected_boxes_in_sample > 0:
                        np_boxes_at_t = boxes_at_t.clone().numpy()
                        tracked_boxes_db[export_sample_id] = {
                            "lidar_T_box": np_boxes_at_t.get_poses(),
                            "raw_box": np_boxes_at_t.__dict__,
                            "track_id": np.array(
                                simple_tracker.track_ids[pcl_time_idx]
                            ),
                        }
                        max_confidence_in_sample = float(np_boxes_at_t.probs.max())
                    else:
                        print(
                            f"No tracked boxes found in {export_sample_id} - skipping"
                        )
                    assert (
                        export_sample_id not in tracked_boxes_conf_stats
                    ), f"overwriting occuring for sample: {export_sample_id}"
                    tracked_boxes_conf_stats[export_sample_id] = {
                        "max_confidence": max_confidence_in_sample,
                        "num_boxes": num_detected_boxes_in_sample,
                    }

                    max_num_box_snippets_to_export = 3
                    if num_detected_boxes_in_sample > 0:
                        num_cutouts_to_export = min(
                            max_num_box_snippets_to_export, num_detected_boxes_in_sample
                        )
                        if num_cutouts_to_export >= num_detected_boxes_in_sample:
                            box_export_idxs = np.arange(num_detected_boxes_in_sample)
                        else:
                            select_box_probs = (
                                np.squeeze(boxes_at_t.probs.cpu().numpy()) + 1e-6
                            )
                            select_box_probs /= select_box_probs.sum()
                            box_export_idxs = np.random.choice(
                                np.arange(num_detected_boxes_in_sample),
                                size=num_cutouts_to_export,
                                p=select_box_probs,
                                replace=False,
                            )
                        pcl_at_t = point_clouds_sensor_cosy[pcl_time_idx]
                        row_idxs_at_t = point_cloud_row_idxs[pcl_time_idx]
                        pcl_at_t_homog = homogenize_pcl(pcl_at_t[:, :3])
                        # print(pcl_at_t[:3])
                        # box sequence starts "later" than point cloud sequence - shift index accordingly:
                        for box_idx in box_export_idxs:
                            box_at_t = boxes_at_t[box_idx]

                            sensor_T_box = box_at_t[None].get_poses()[0]
                            box_T_sensor = torch.linalg.inv(sensor_T_box)
                            pcl_box = torch.cat(
                                [
                                    torch.einsum(
                                        "ij,nj->ni",
                                        box_T_sensor,
                                        pcl_at_t_homog.double(),
                                    )[:, :3].float(),
                                    pcl_at_t[:, [-1]],
                                ],
                                dim=-1,
                            )
                            point_is_in_box = torch.all(
                                torch.abs(pcl_box[:, 0:3]) <= 1.1 * 0.5 * box_at_t.dims,
                                dim=-1,
                            )
                            num_points_in_box = torch.count_nonzero(point_is_in_box)
                            if num_points_in_box == 0:
                                # does not make sense to store empty boxes to do patch augmentation
                                continue
                            pcl_box = (
                                pcl_box[point_is_in_box]
                                .cpu()
                                .numpy()
                                .astype(np.float32)
                            )
                            rows_box = row_idxs_at_t[point_is_in_box].cpu().numpy()
                            box_points_snippets_db["pcl_in_box_cosy"].append(pcl_box)
                            box_points_snippets_db["boxes"].append(box_at_t.clone())
                            box_points_snippets_db["box_T_sensor"].append(
                                box_T_sensor.numpy()
                            )
                            box_points_snippets_db["lidar_rows"].append(rows_box)
                            box_points_snippets_db["unique_track_id"].append(
                                int(
                                    simple_tracker.track_ids[pcl_time_idx][
                                        box_idx
                                    ].numpy()
                                )
                            )

            curr_db_size_mb = estimate_augm_db_size_mb(box_points_snippets_db)
            if curr_db_size_mb > max_augm_db_size_mb:
                box_points_snippets_db = drop_boxes_from_augmentation_db(
                    box_points_snippets_db, max_augm_db_size_mb
                )

            pbar.update()
            num_tracked_sequences += 1

    if time.time() >= timeout_at:
        print(f"Triggered timeout after {timeout_s} seconds!")

    with torch.no_grad():
        mined_objects_target_paths = {}
        if export_raw_tracked_detections_to:
            save_mined_box_db(
                tracking_cfg,
                export_raw_tracked_detections_to,
                tracked_boxes_conf_stats,
                tracked_boxes_db,
                mined_objects_target_paths,
            )
            print(
                f"Saving data from {num_successfull_tracks} sequences to {export_raw_tracked_detections_to}"
            )
            save_name, _ = save_augmentation_database(
                box_points_snippets_db,
                export_raw_tracked_detections_to,
                global_step,
            )
    print(f"{datetime.now()} finished tracking at step {global_step}.")
    return save_name, mined_objects_target_paths


def save_mined_box_db(
    tracking_cfg: Dict[str, Any],
    export_raw_tracked_detections_to: Path,
    tracked_boxes_conf_stats: Dict[str, float],
    tracked_boxes_db: Shape,
    mined_objects_target_paths: Dict[str, Path],
):
    """
    save the mined pseudo ground truth to a directory,
    path to result is returned in mined_objects_target_paths as dict entries ["tracked", "raw"]
    """
    Path(export_raw_tracked_detections_to).mkdir(exist_ok=True, parents=True)
    save_config(
        cfg=tracking_cfg,
        path=Path(export_raw_tracked_detections_to) / "tracking_cfg.yaml",
    )

    for stats_fname, box_stats in {
        "tracked_box_stats.yaml": tracked_boxes_conf_stats,
    }.items():
        with open(
            Path(export_raw_tracked_detections_to).joinpath(stats_fname), "w"
        ) as outfile:
            yaml.dump(box_stats, outfile)

    for db_identifier, box_db in {
        "tracked": tracked_boxes_db,
    }.items():
        db_target_pth = Path(export_raw_tracked_detections_to) / db_identifier
        if db_target_pth.with_suffix(".npz").exists():
            save_str = "Overwrote"
        else:
            save_str = "Saving"
        np.savez_compressed(db_target_pth, box_db)
        mined_objects_target_paths[db_identifier] = db_target_pth.with_suffix(".npz")
        print(f"{save_str} box db with {len(box_db)} entries to {db_target_pth}!")


def update_db_with_this_box_stuff(
    keep_these_track_ids_timestamps_boxes_extra_attrs: Dict[
        str, Dict[Tuple[int, int], Shape]
    ],
    track_id: int,
    start_time_idx: int,
    box_sequence_world_for_track_id: Shape,
    box_sequence_sensor_for_track_id: Shape,
    extra_attributes_for_this_box: List[Dict[str, Any]],
    w_T_sensor_poses_ti: List[torch.FloatTensor],
    track_age: int,
):
    box_sequence_sensor_for_track_id = update_sensor_boxes_from_world_boxes(
        box_sequence_world=box_sequence_world_for_track_id,
        box_sequence_sensor=box_sequence_sensor_for_track_id,
        w_T_sensor_ti=w_T_sensor_poses_ti[start_time_idx : start_time_idx + track_age],
    )
    keep_these_track_ids_timestamps_boxes_extra_attrs["world_refined"][
        (
            int(track_id),
            int(start_time_idx),
        )  # these are tensors...
    ] = box_sequence_world_for_track_id

    keep_these_track_ids_timestamps_boxes_extra_attrs["sensor_refined"][
        (
            int(track_id),
            int(start_time_idx),
        )  # these are tensors...
    ] = box_sequence_sensor_for_track_id

    keep_these_track_ids_timestamps_boxes_extra_attrs["extra_attributes"][
        (
            int(track_id),
            int(start_time_idx),
        )  # these are tensors...
    ] = extra_attributes_for_this_box


def perform_local_box_refinement(
    cfg: Dict[str, float],
    box_predictor: BoxLearner,
    point_clouds_sensor_cosy: List[torch.FloatTensor],
    box_sequence_in_sensor_cosy_for_specific_track_id: Shape,
    track_age: int,
    start_time_idx: int,
):
    # smooth dims
    box_dims_quantile = cfg.data.tracking_cfg.setdefault(
        "box_refinement_dims_quantile", 0.95
    )
    if isinstance(box_predictor, (FlowClusterDetector,)):
        box_dims_quantile = 0.95
    else:
        box_dims_quantile = 0.6

    refined_box_dims = torch.quantile(
        box_sequence_in_sensor_cosy_for_specific_track_id.dims,
        q=box_dims_quantile,
        dim=0,
    )

    if (
        cfg.data.tracking_cfg.fit_box_to_points.fit_rot
        or cfg.data.tracking_cfg.fit_box_to_points.fit_pos
    ):
        assert (
            box_sequence_in_sensor_cosy_for_specific_track_id.shape[0] == track_age
        ), (
            box_sequence_in_sensor_cosy_for_specific_track_id.shape,
            track_age,
        )
        for track_time_idx in range(track_age):
            track_time_idx_global = start_time_idx + track_time_idx
            pcl_at_t = point_clouds_sensor_cosy[track_time_idx_global]
            sensor_box_at_t = box_sequence_in_sensor_cosy_for_specific_track_id[
                track_time_idx
            ]

            sensor_T_box = sensor_box_at_t[None].get_poses()[0]
            box_T_sensor = torch.linalg.inv(sensor_T_box)
            pcl_at_t_homog = homogenize_pcl(pcl_at_t[:, :3]).double()
            pcl_box_3d = torch.einsum("ij,nj->ni", box_T_sensor, pcl_at_t_homog)[:, :3]
            point_is_in_box = torch.all(
                torch.abs(pcl_box_3d[:, 0:2].float())
                < (
                    0.5  # need only half size here!
                    * cfg.data.tracking_cfg.fit_box_to_points.fitting_dims_bloat_factor
                    * sensor_box_at_t.dims[:2]
                ),
                dim=-1,
            )
            if torch.count_nonzero(point_is_in_box) > 0:
                pcl_sensor_inside_box = pcl_at_t_homog[:, :3][point_is_in_box].numpy()
                (
                    refined_box_center_sensor,
                    _,
                    _,
                    refined_box_yaw_sensor,
                ) = fit_2d_box_modest(
                    pcl_sensor_inside_box,
                    fit_method="closeness_to_edge",
                )
                if cfg.data.tracking_cfg.fit_box_to_points.fit_rot:
                    delta_rot_sensor = refined_box_yaw_sensor - float(
                        sensor_box_at_t.rot
                    )
                    box_sequence_in_sensor_cosy_for_specific_track_id.rot[
                        track_time_idx
                    ] += delta_rot_sensor
                if cfg.data.tracking_cfg.fit_box_to_points.fit_pos:
                    box_sequence_in_sensor_cosy_for_specific_track_id.pos[
                        track_time_idx
                    ] = torch.cat(
                        [
                            torch.from_numpy(refined_box_center_sensor),
                            sensor_box_at_t.pos[[2]],
                        ]
                    )
                # w_T_sensor = w_T_sensor_poses_ti[track_time_idx_global]
                # refined_box_center_world = torch.einsum(
                #     "ij,j->i",
                #     w_T_sensor,
                #     refined_box_center_sensor_homog,
                # )

                # box_sequence_world_for_track_id.pos[
                #     track_time_idx
                # ] = refined_box_center_world[:3]

                # delta_pos_sensor_2d = (
                #    torch.from_numpy(
                #        refined_box_center_sensor,
                #    )
                #    - sensor_box_at_t.pos[:2]
                # )
                # delta_pos_sensor_3d = torch.cat(
                #    (delta_pos_sensor_2d, torch.tensor((0.0,))), dim=-1
                # )
                # box_sequence_in_sensor_cosy_for_specific_track_id.pos[
                #    track_time_idx
                # ] += delta_pos_sensor_3d
                # w_T_sensor_at_t = w_T_sensor_poses_ti[
                #    start_time_idx + track_time_idx
                # ]
                # box_sequence_world_for_track_id.pos[
                #    track_time_idx
                # ] = torch.einsum(
                #    "ij,j->i",
                #    w_T_sensor_at_t,
                #    torch.cat(
                #        [
                #            box_sequence_in_sensor_cosy_for_specific_track_id.pos[
                #                track_time_idx
                #            ],
                #            torch.tensor((1.0,)),
                #        ],
                #        dim=-1,
                #    ).double(),
                # )[
                #    :3
                # ].float()

    box_sequence_in_sensor_cosy_for_specific_track_id = (
        set_box_size_keep_closest_point_constant(
            box_sequence_in_sensor_cosy_for_specific_track_id, refined_box_dims
        )
    )
    return box_sequence_in_sensor_cosy_for_specific_track_id


def draw_colored_tracks_onto_image(
    simple_tracker: Union[NotATracker, FlowBasedBoxTracker],
    torch_bev_extent_m: torch.FloatTensor,
    pcl_img_all_tracks_f32: np.ndarray,
):
    _, max_track_id = simple_tracker.get_min_max_track_id()
    num_categories = 1 + max_track_id.detach().cpu().numpy()
    categories = list(range(num_categories))
    np.random.shuffle(categories)
    colors = np.stack(
        [gist_rainbow(float(i) / num_categories) for i in categories]
    ).astype(pcl_img_all_tracks_f32.dtype)

    global_boxes = simple_tracker.get_boxes_in_world_coordinates()
    for time_idx, boxes_at_t in enumerate(global_boxes):
        track_ids_at_t = simple_tracker.track_ids[time_idx].detach().cpu().numpy()
        if torch.count_nonzero(boxes_at_t.valid) == 0:
            continue
        batched_box_colors = colors[track_ids_at_t][None, :, :3]
        batched_boxes_at_t = (
            Shape.from_list_of_shapes((boxes_at_t,)).to(torch.float64).cpu()
        )
        pcl_img_all_tracks_f32 = draw_box_onto_image(
            batched_boxes_at_t,
            pcl_img_all_tracks_f32,
            bev_range_m=torch_bev_extent_m,
            color=batched_box_colors,
        )

    return pcl_img_all_tracks_f32, colors


def propagate_boxes_forward_using_flow(
    pred_boxes: Shape,
    point_cloud_ta: torch.FloatTensor,
    valid_mask_ta: torch.BoolTensor,
    pointwise_flow_ta_tb: torch.FloatTensor,
    odom_t0_t1: torch.DoubleTensor,
    device: str,
):
    point_is_in_box = pred_boxes.get_points_in_box_bool_mask(point_cloud_ta)
    mean_flow_per_box = (
        # dims: [batch, num_points, boxes, flow_dims(3)]
        pointwise_flow_ta_tb[:, :, None, :]  # broadcast flow across all boxes
        * valid_mask_ta[
            :, :, None, None
        ].float()  # mask out invalid flow/points, broadcast across flow dims
        * point_is_in_box[:, :, :, None].float()  # broadcast across flow dims
    ).sum(dim=1) / torch.clip(point_is_in_box.sum(dim=1), min=1.0)[:, :, None]

    fg_kabsch_trafos = torch.eye(4, dtype=torch.float64, device=device)[
        None, None, ...
    ].repeat(pred_boxes.shape[0], pred_boxes.shape[1], 1, 1)
    fg_kabsch_trafos[:, :, :3, 3] = mean_flow_per_box.double()

    # this is the odometry that fits to the kabsch trafo
    bg_kabsch_trafo = torch.linalg.inv(odom_t0_t1)[None, None, ...].to(device)
    bt0_deltaT_bt1 = extract_box_motion_transform_without_sensor_odometry(
        pred_boxes, fg_kabsch_trafos, bg_kabsch_trafo
    )
    st0_T_bt0 = pred_boxes.get_poses()

    st0_T_dyn_motion_warped_bt1 = (st0_T_bt0 @ bt0_deltaT_bt1)[0].detach().cpu()

    # redo the bg_kabsch_trafo computation here, since on NuscenesDataset this might have been extrapolated to _tx
    bg_kabsch_trafo = torch.linalg.inv(odom_t0_t1)[None, None, ...].to(device)

    st1_T_bt1 = fg_kabsch_trafos @ st0_T_bt0

    return (
        fg_kabsch_trafos,
        odom_t0_t1,
        bg_kabsch_trafo,
        st0_T_dyn_motion_warped_bt1,
        st1_T_bt1,
    )


def decide_keep_or_drop_box(
    *,
    tracking_cfg: Dict[str, float],
    box_sequence_world_for_specific_track_id: Shape,
    min_track_obj_speed_mps: float,
    track_id: Union[int, torch.IntTensor],
    time_between_frames_s: Union[float, np.ndarray],
    verbose: bool,
    is_flow_cluster_detector: bool,
):
    box_poses_world = box_sequence_world_for_specific_track_id.get_poses()
    track_coors, track_rots = torch_decompose_matrix(box_poses_world)
    global_tracklet_coors = track_coors[:, 0:2].cpu().numpy()
    general_motion_vector_m = global_tracklet_coors[-1] - global_tracklet_coors[0]
    total_dist_covered_m = np.linalg.norm(general_motion_vector_m)
    # dist_between_track_coors_m_2d = np.sqrt(
    #     np.diff(global_tracklet_coors, axis=0) ** 2
    # ).sum(axis=-1)
    # dist_between_track_coors_m_2d = np.append(
    #     dist_between_track_coors_m_2d, dist_between_track_coors_m_2d[-1]
    # )

    track_rots = np.squeeze(track_rots.cpu().numpy())

    seq_len = len(global_tracklet_coors)
    if min_track_obj_speed_mps > 0.0:
        total_track_length_sec = seq_len * time_between_frames_s
        obj_speed_mps = total_dist_covered_m / total_track_length_sec
        obj_is_fast_enough = obj_speed_mps >= min_track_obj_speed_mps
    else:
        obj_is_fast_enough = True

    keep = obj_is_fast_enough

    if keep and is_flow_cluster_detector:
        track_is_long_enough = (
            total_dist_covered_m
            >= tracking_cfg.flow_cluster_detector_min_travel_dist_filter_m
        )
        keep = keep and track_is_long_enough

    if verbose:
        keep_or_drop = "Keep" if keep else "Drop"
        print(f"{keep_or_drop}: Track ID: {track_id}")

    return keep, total_dist_covered_m


def get_clean_train_dataset_single_batch(cfg, for_tracking=False, need_flow=True):
    """
    no augmentation, no shuffling, just pure train data
    """
    if cfg.data.source == "nuscenes":
        train_dataset = NuscenesDataset(
            mode="train",
            cfg=cfg,
            use_geom_augmentation=False,
            use_skip_frames="never",
            pure_inference_mode=False,
            for_tracking=for_tracking,
            shuffle=False,
            need_flow=need_flow,
        )
    if cfg.data.source == "av2":
        train_dataset = AV2Dataset(
            mode="train",
            cfg=cfg,
            use_geom_augmentation=False,
            use_skip_frames="never",
            pure_inference_mode=False,
            for_tracking=for_tracking,
            shuffle=False,
            need_flow=need_flow,
        )
    if cfg.data.source == "waymo":
        train_dataset = WaymoDataset(
            mode="train",
            cfg=cfg,
            use_geom_augmentation=False,
            use_skip_frames="never",
            pure_inference_mode=False,
            for_tracking=for_tracking,
            shuffle=False,
            need_flow=need_flow,
        )

    elif cfg.data.source == "kitti":
        if cfg.data.train_on_box_source == "mined":
            train_dataset = KittiRawDataset(
                mode="train",
                cfg=cfg,
                use_geom_augmentation=False,
                use_skip_frames="never",
                pure_inference_mode=False,
                training_target="object",  # trigger slim flow loading
                for_tracking=for_tracking,
                shuffle=False,
                need_flow=need_flow,
            )
        elif cfg.data.train_on_box_source == "gt":
            train_dataset = KittiObjectDataset(
                mode="train",
                cfg=cfg,
                use_geom_augmentation=False,
                use_skip_frames="never",
                pure_inference_mode=True,
                shuffle=False,
            )
        else:
            raise NotImplementedError(cfg.data.train_on_box_source)

    return train_dataset


# def get_tracked_boxes_from_dataset(od_network, cfg):
#     dataset, dataloader = get_clean_train_dataset_single_batch(cfg)
#     map_sequence_to_file_indices = {}
#
#     indices = [0, 1, 2]  # select your indices here as a list
#     subset = torch.utils.data.Subset(dataset, indices)


if __name__ == "__main__":
    main()
