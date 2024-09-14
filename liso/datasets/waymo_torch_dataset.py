import pickle
import time
from collections import abc
from copy import deepcopy
from inspect import getsourcefile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from liso.datasets.torch_dataset_commons import (
    LidarDataset,
    LidarSample,
    get_weighted_random_sampler_dropping_samples_without_boxes,
    lidar_dataset_collate_fn,
    recursive_npy_dict_to_torch,
    worker_init_fn,
)
from liso.jcp.jcp import JPCGroundRemove
from liso.kabsch.shape_utils import Shape
from liso.utils.torch_transformation import homogenize_pcl
from tqdm import tqdm

WAYMO_MOVABLE_CLASSES = ("Vehicle", "Pedestrian", "Cyclist")
WAYMO_NON_MOVABLE_CLASSES = ("unknown", "Sign")
WAYMO_CLASSES = WAYMO_MOVABLE_CLASSES + WAYMO_NON_MOVABLE_CLASSES

WAYMO_MOVABLE_CLASS_FREQS = {
    "Cyclist": 13340,
    "Pedestrian": 542324,
    "Vehicle": 1245056,
}

vehicle_Twaymo_lidar = np.array(
    [
        [
            1.0,
            0.0,
            0.0,
            1.751,  # t_x, dirty hack, reconstructed from image of car
        ],
        [
            0.0,
            1.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            1.0,
            1.765,  # t_z
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    ]
)


class WaymoDataset(LidarDataset):
    class_names = WAYMO_CLASSES
    movable_class_names = WAYMO_MOVABLE_CLASSES
    class_name_to_idx_mapping = {k: i for i, k in enumerate(class_names)}
    idx_to_class_name_mapping = dict(enumerate(class_names))
    movable_class_frequencies = np.array(
        [WAYMO_MOVABLE_CLASS_FREQS[k] for k in WAYMO_MOVABLE_CLASS_FREQS]
    ) / sum(WAYMO_MOVABLE_CLASS_FREQS.values())

    def __init__(
        self,
        cfg,
        shuffle: bool,
        use_geom_augmentation: bool,
        use_skip_frames: str,
        mode="train",
        size=None,
        verbose=False,
        pure_inference_mode=False,
        get_only_these_specific_samples=None,
        path_to_augmentation_db=None,
        path_to_mined_boxes_db=None,
        for_tracking=False,
        need_flow=True,
    ) -> None:
        super().__init__(
            cfg,
            mode=mode,
            shuffle=shuffle,
            use_geom_augmentation=use_geom_augmentation,
            use_skip_frames=use_skip_frames,
            path_to_augmentation_db=path_to_augmentation_db,
            path_to_mined_boxes_db=path_to_mined_boxes_db,
            for_tracking=for_tracking,
            need_flow=need_flow,
        )
        self.verbose = verbose
        self.pure_inference_mode = pure_inference_mode
        self.dataset_root = Path(cfg.data.paths.waymo.local)
        self.groundsegmentation_root = Path(
            cfg.data.paths.waymo.ground_segmentation.local
        )
        self.flow_gt_root = Path(cfg.data.paths.waymo.flow_gt.local)
        kiss_icp_root = Path(cfg.data.paths.waymo.poses_kiss_icp_kitti_lidar.local)
        self.kiss_icp_poses_path_wo_ext = kiss_icp_root.joinpath(self.mode)
        if self.cfg.data.odom_source == "kiss_icp":
            self.initialize_loader_saver_if_necessary()

            self.kiss_icp_poses = self.loader_saver_helper.load_sample(
                self.kiss_icp_poses_path_wo_ext.with_suffix(".npy"),
                np.load,
                allow_pickle=True,
            ).item()
            self.loader_saver_helper = (
                None  # we need one separate loader saver for each process
            )

        self.processed_data_path = "waymo_processed_data_v0_5_0"

        success = False  # mount is so unstable we need to try-catch retry
        for _ in range(20):
            try:
                label_infos_pkls = sorted(
                    self.dataset_root.joinpath(self.processed_data_path).rglob("*.pkl")
                )
                success = True
            except Exception as e:
                print(e)
                time.sleep(10)

            if success:
                break

        print("found {0} pkl files".format(len(label_infos_pkls)))

        split_file = self.dataset_root.joinpath("ImageSets", f"{mode}.txt")
        with open(split_file, "r") as f:
            sequences_in_split = [Path(el).stem for el in f.read().splitlines()]
        self.label_infos = []
        for label_info_pkl in label_infos_pkls:
            if label_info_pkl.stem in sequences_in_split:
                with open(
                    label_info_pkl,
                    "rb",
                ) as f:
                    self.label_infos.append(pickle.load(f))

        print("Found {0} sequences in split".format(len(self.label_infos)))
        if mode == "train" and cfg.data.setdefault("waymo_downsample_factor", 5) != 1:
            self.label_infos = self.label_infos[:: cfg.data.waymo_downsample_factor]
        downsample_dataset_keep_ratio = self.cfg.data.setdefault(
            "downsample_dataset_keep_ratio", 1.0
        )
        if self.mode == "train" and self.cfg.data.downsample_dataset_keep_ratio != 1.0:
            self.dataset_sequence_is_messed_up = True
            if cfg.data.waymo_downsample_factor != 1:
                print(
                    "not recommended to downsample sequences AND random downsample",
                    cfg.data.waymo_downsample_factor,
                )
            assert downsample_dataset_keep_ratio < 1.0, downsample_dataset_keep_ratio
            print(
                f"Downsampling dataset by {downsample_dataset_keep_ratio} from {sum(len(li) for li in self.label_infos)} to ..."
            )
            num_seq = len(self.label_infos)
            # chosen_seq_frame_idxs = []
            # for seq_idx in range(num_seq):
            #     num_samples_in_seq = len(self.label_infos[seq_idx])
            #     for sample_idx in range(
            #         num_samples_in_seq - 1
            #     ):  # -1: skips the last sample, which messes up flow computation
            #         chosen_seq_frame_idxs.append((seq_idx, sample_idx))
            all_seq_frame_idxs = np.load(
                Path(getsourcefile(lambda: 0)).parent.parent
                / "assets"
                / "waymo_shuffled_sequence_sample_pairs.npz",
                allow_pickle=True,
            )["arr_0"]
            # chosen_sample_idxs = np.random.choice(
            #     np.arange(len(chosen_seq_frame_idxs)),
            #     size=int(downsample_dataset_keep_ratio * len(chosen_seq_frame_idxs)),
            #     replace=False,
            # ).tolist()
            # chosen_seq_frame_idxs = [
            #     chosen_seq_frame_idxs[idx] for idx in chosen_sample_idxs
            # ]
            all_seq_frame_idxs_that_we_still_have_mask = all_seq_frame_idxs[:, 0] < len(
                self.label_infos
            )
            all_seq_frame_idxs = all_seq_frame_idxs[
                all_seq_frame_idxs_that_we_still_have_mask
            ].tolist()
            chosen_seq_frame_idxs = all_seq_frame_idxs[
                : int(downsample_dataset_keep_ratio * len(all_seq_frame_idxs))
            ]
            keep_label_infos = [[] for _ in range(num_seq)]
            for seq_idx, sample_idx in chosen_seq_frame_idxs:
                keep_label_infos[seq_idx].append(self.label_infos[seq_idx][sample_idx])
            self.label_infos = list(filter(lambda x: len(x) > 0, keep_label_infos))
            print(f"...{sum(len(li) for li in self.label_infos)} samples.")
            print("preloading databses!")
            self.initialize_dbs_if_necessary()

        print("Keeping {0} sequences".format(len(self.label_infos)))

        if get_only_these_specific_samples:
            assert self.mode == "val", self.mode
            keep_label_infos = []
            for label_info in self.label_infos:
                keep_frame_infos = []
                for frame_info in label_info:
                    sample = frame_info["frame_id"]
                    for special_sample in get_only_these_specific_samples:
                        if sample in special_sample:
                            keep_frame_infos.append(frame_info)
                            break
                if len(keep_frame_infos) > 0:
                    keep_label_infos.append(keep_frame_infos)

            self.label_infos = keep_label_infos
            print(f"Warning: Only requested {len(self.label_infos)} samples")

        # self.label_infos = [
        #     self.label_infos[0],
        # ]
        self.label_infos_lens = [len(el) for el in self.label_infos]

        self.cumulative_label_infos_lens = np.cumsum(
            [
                0,
            ]
            + self.label_infos_lens
        )
        assert self.mode in (
            "train",
            "val",
        ), self.mode
        assert not (
            (self.mode == "val" or self.mode == "test") and self.use_geom_augmentation
        ), "we don't want to augment validation data"
        if self.cfg.data.flow_source != "gt":
            pred_flow_path = Path(
                self.cfg.data.paths.waymo.slim_flow[self.cfg.data.flow_source]["local"]
            )
            self.pred_flow_path = pred_flow_path
            print(f"Loading flow seperately from source {pred_flow_path}")
        else:
            self.pred_flow_path = None

        # self.box_augm_db = None
        # if (
        #     self.cfg.data.augmentation.boxes.active
        #     and path_to_augmentation_db is not None
        # ):
        #     self.box_augm_db = np.load(
        #         path_to_augmentation_db, allow_pickle=True
        #     ).item()
        #     self.box_augm_db["boxes"] = Shape(**self.box_augm_db["boxes"]).to_tensor()
        #     self.box_augm_db["box_T_sensor"] = torch.from_numpy(
        #         self.box_augm_db["box_T_sensor"]
        #     )
        #     self.box_augm_cfg = self.cfg.data.augmentation.boxes

    def map_flat_index_to_meta_data(self, index):
        segment_idx = np.max(np.where(self.cumulative_label_infos_lens <= index))
        frame_idx = index - self.cumulative_label_infos_lens[segment_idx]
        src_key = "t0"
        target_key = "t1"
        num_skip_frames = {"t1": 1}[target_key]
        if (frame_idx + num_skip_frames) >= self.label_infos_lens[segment_idx] - 1:
            # not enough follow up samples, just reuse last sample pair
            frame_idx -= 1

        label_infos_t0 = self.label_infos[segment_idx][frame_idx]
        _, fname = self.get_pcl_path_sample_id(label_infos_t0)
        return (
            segment_idx,
            frame_idx,
            src_key,
            target_key,
            num_skip_frames,
            label_infos_t0,
            fname,
        )

    def __len__(self) -> int:
        return sum(self.label_infos_lens)

    def object_is_movable(self, obj) -> bool:
        return obj in WAYMO_MOVABLE_CLASSES

    def generate_save_kiss_icp_poses_groundseg(self) -> bool:
        all_poses_w_T_lidar = {}
        try:
            all_poses_w_T_lidar = np.load(
                self.kiss_icp_poses_path_wo_ext.with_suffix(".npy"),
                allow_pickle=True,
            ).item()
        except FileNotFoundError as e:
            print(e)
            print("Recomputing full KISS odometry from scratch")
        from kiss_icp.config import KISSConfig
        from kiss_icp.kiss_icp import KissICP

        kiss_config = KISSConfig()
        kiss_config.mapping.voxel_size = 0.01 * kiss_config.data.max_range
        for sequence in tqdm(self.label_infos, disable=False):
            scene_name = sequence[0]["frame_id"][:-4]

            if scene_name in all_poses_w_T_lidar:
                continue
            pcl_filenames = []
            odometry = KissICP(config=kiss_config)
            for frame_idx in range(len(sequence)):
                label_infos_t0 = sequence[frame_idx]
                pcl_t0, fname_t0 = self.load_waymo_pcl_into_lidar_frame(label_infos_t0)

                self.get_is_ground_mask(pcl_t0, fname_t0)

                pcl_filenames.append(str(fname_t0))

                odometry.register_frame(
                    pcl_t0[:, :3].astype(np.float64),
                    np.zeros(pcl_t0.shape[0]).astype(np.float64),
                )
            all_poses_w_T_lidar[scene_name] = dict(zip(pcl_filenames, odometry.poses))
            np.save(
                self.kiss_icp_poses_path_wo_ext, all_poses_w_T_lidar, allow_pickle=True
            )
        return all_poses_w_T_lidar

    def get_samples_for_sequence(
        self,
        sequence_idx: int,
        start_idx_in_sequence: int,
        sequence_length: int = None,
    ) -> List[LidarSample]:
        assert not self.shuffle, "does not make sense when shuffled"
        assert not self.dataset_sequence_is_messed_up
        if sequence_length is None:
            sequence_length = self.label_infos_lens[sequence_idx]
        global_start_idx_of_sequence = int(np.sum(self.label_infos_lens[:sequence_idx]))
        global_start_idx = global_start_idx_of_sequence + start_idx_in_sequence
        # global_end_idx = global_start_idx + sequence_length
        chosen_samples = [
            LidarSample(
                idx=global_start_idx + idx,
                sample_name=str(
                    self.get_pcl_path_sample_id(
                        self.label_infos[sequence_idx][start_idx_in_sequence + idx]
                    )[1]
                ),
                timestamp=0,
                full_path="",
            )
            for idx in range(sequence_length)
        ]

        return chosen_samples

    def get_scene_index_for_scene_name(
        self,
        scene_str,
    ) -> int:
        for i, scene in enumerate(self.label_infos):
            if scene[0]["frame_id"].split("_")[0] == scene_str:
                return i
        return None

    def get_consecutive_sample_idxs_for_sequence(
        self,
        sequence_idx: int,
        drop_last_sample: bool = True,
    ) -> List[LidarSample]:
        assert not self.shuffle, "does not make sense when shuffled"
        assert not self.dataset_sequence_is_messed_up
        if sequence_idx >= len(self.label_infos_lens):
            print("Ran out of sequences!")
            return None
        samples_in_sequence = self.get_samples_for_sequence(
            sequence_idx=sequence_idx,
            start_idx_in_sequence=0,
            sequence_length=self.label_infos_lens[sequence_idx],
        )
        if drop_last_sample:
            samples_in_sequence = samples_in_sequence[:-2]
        return samples_in_sequence

    def extract_boxes_for_timestamp(
        self,
        sample_content: Dict[str, np.ndarray],
        src_key: str,
        target_key: str,
    ) -> Shape:
        objects_key = f"objects_{src_key}"
        if objects_key in sample_content["gt"]:
            objects = sample_content["gt"][objects_key]
            class_names = sample_content.get(f"class_names_{src_key}", None)
        else:
            objects = sample_content["gt"][f"boxes_{src_key}"]
            class_names = [
                "Vehicle",
            ] * objects.shape[0]

        # mapped_class_names = None
        # if class_names is not None:
        #     mapped_class_names = np.ones_like(class_names)
        #     for k in MAP_WAYMO_TO_NUSC_CLASSES:
        #         mapped_class_names[class_names == k] = MAP_WAYMO_TO_NUSC_CLASSES[k]
        # return objects, mapped_class_names

        return objects, class_names

    def load_waymo_pcl_into_lidar_frame(self, label_infos) -> np.ndarray:
        pcl_path, sample_id = self.get_pcl_path_sample_id(label_infos)
        pcl = self.loader_saver_helper.load_sample(pcl_path, np.load)[:, :4][
            : label_infos["num_points_of_each_lidar"][0], :
        ]  # drop everything but x,y,z,intensity
        pcl_veh_homog = homogenize_pcl(pcl[:, :3])
        pcl_lidar = np.einsum(
            "ij,nj->ni", np.linalg.inv(vehicle_Twaymo_lidar), pcl_veh_homog
        )[:, :3]
        pcl[:, :3] = pcl_lidar
        return pcl, sample_id

    def get_pcl_path_sample_id(self, label_infos) -> (Path, Path):
        pcl_path = Path(self.dataset_root).joinpath(
            self.processed_data_path,
            label_infos["point_cloud"]["lidar_sequence"],
            "{0:04d}.npy".format(label_infos["point_cloud"]["sample_idx"]),
        )
        sample_id = Path(*pcl_path.parts[-2:]).with_suffix("")
        return pcl_path, sample_id

    def __getitem__(self, index):
        self.initialize_loader_saver_if_necessary()

        self.initialize_dbs_if_necessary()
        # if self.mode != "train":
        #     index = 16604
        (
            segment_idx,
            frame_idx,
            src_key,
            target_key,
            num_skip_frames,
            label_infos_t0,
            _,
        ) = self.map_flat_index_to_meta_data(index)
        src_trgt_time_delta_s = 0.1
        # if we are training SLIM, not from fused train data
        # we may decide to skip some data
        # (
        #     src_key,
        #     target_key,
        #     _,
        #     src_trgt_time_delta_s,
        # ) = self.select_time_keys()

        if (frame_idx + num_skip_frames) >= self.label_infos_lens[segment_idx] - 1:
            # not enough follow up samples, just reuse last sample pair
            frame_idx -= 1
            src_key = "t0"
            target_key = "t1"
            src_trgt_time_delta_s = 0.1
            num_skip_frames = 1

        _, fname_t0 = self.get_pcl_path_sample_id(label_infos_t0)
        (
            pcl_t0,
            fname_must_be_same_as_fname_t0,
        ) = self.load_waymo_pcl_into_lidar_frame(label_infos_t0)
        assert fname_must_be_same_as_fname_t0 == fname_t0, (
            fname_must_be_same_as_fname_t0,
            fname_t0,
        )

        label_infos_t1 = self.label_infos[segment_idx][frame_idx + num_skip_frames]
        time_delta_s = (
            label_infos_t1["metadata"]["timestamp_micros"]
            - label_infos_t0["metadata"]["timestamp_micros"]
        ) / 1e6
        if np.abs(src_trgt_time_delta_s - time_delta_s) > 0.02:
            # make sure that we don't have a time offset of more than 20 ms from expected
            print(
                "large time delta encountered",
                segment_idx,
                frame_idx,
                fname_t0,
                label_infos_t1["metadata"]["timestamp_micros"],
                label_infos_t0["metadata"]["timestamp_micros"],
                src_trgt_time_delta_s,
                time_delta_s,
            )
        src_trgt_time_delta_s = time_delta_s

        odom_t0_t1, odom_t1_t0 = self.get_odometry_from_label_infos(
            label_infos_t0, label_infos_t1
        )
        (
            boxes_t0,
            obj_names_t0,
            obj_ids_t0,
        ) = self.get_nonempty_flow_relevant_boxes_into_lidar_frame(label_infos_t0)
        (
            boxes_t1,
            obj_names_t1,
            obj_ids_t1,
        ) = self.get_nonempty_flow_relevant_boxes_into_lidar_frame(label_infos_t1)
        both_frames_have_boxes = np.size(obj_ids_t0) != 0 and np.size(obj_ids_t1) != 0
        if both_frames_have_boxes:
            indices_into_t0, indices_into_t1 = np.where(
                obj_ids_t0[:, None] == obj_ids_t1[None, :]
            )
            boxes_reordered_t0 = boxes_t0[indices_into_t0]
            boxes_reordered_t1 = boxes_t1[indices_into_t1]
            assert (
                obj_names_t0[indices_into_t0] == obj_names_t1[indices_into_t1]
            ).all(), "object changed class - logical error!!!"
        else:
            boxes_reordered_t0 = Shape.createEmpty()
            boxes_reordered_t1 = Shape.createEmpty()

        sensor_Tt0_box = boxes_reordered_t0.get_poses()
        sensor_Tt1_box = boxes_reordered_t1.get_poses()

        pcl_t1, fname_t1 = self.load_waymo_pcl_into_lidar_frame(label_infos_t1)

        box_velos_reordered_t0_t1 = np.linalg.norm(
            self.get_object_velocity_in_obj_coords(
                odom_t0_t1, sensor_Tt0_box, sensor_Tt1_box
            ),
            axis=-1,
            keepdims=True,
        )
        box_velos_reordered_t1_t0 = np.linalg.norm(
            self.get_object_velocity_in_obj_coords(
                odom_t1_t0, sensor_Tt1_box, sensor_Tt0_box
            ),
            axis=-1,
            keepdims=True,
        )
        boxes_t0.velo = np.linalg.norm(boxes_t0.velo, axis=-1, keepdims=True)
        boxes_t1.velo = np.linalg.norm(boxes_t1.velo, axis=-1, keepdims=True)
        if both_frames_have_boxes:
            boxes_t0.velo[indices_into_t0] = box_velos_reordered_t0_t1
            boxes_t1.velo[indices_into_t1] = box_velos_reordered_t1_t0

        sample_content = {}

        sample_content[f"pcl_{src_key}"] = pcl_t0.astype(np.float32)

        is_ground_t0 = self.get_is_ground_mask(pcl_t0, fname_t0)
        is_ground_t1 = self.get_is_ground_mask(pcl_t1, fname_t1)

        sample_content[f"is_ground_{src_key}"] = is_ground_t0
        sample_content[f"is_ground_{target_key}"] = is_ground_t1
        sample_content[f"pcl_{target_key}"] = pcl_t1.astype(np.float32)
        sample_content[f"odom_{src_key}_{target_key}"] = odom_t0_t1
        sample_content[f"odom_{target_key}_{src_key}"] = odom_t1_t0
        if self.need_flow:
            (
                lidar_flow_t0_t1,
                lidar_flow_t1_t0,
            ) = self.load_precomputed_waymo_gt_flow(
                pcl_t0=pcl_t0,
                pcl_t1=pcl_t1,
                fname_t0=fname_t0,
                fname_t1=fname_t1,
                odom_t0_t1=odom_t0_t1,
                odom_t1_t0=odom_t1_t0,
                boxes_reordered_t0=boxes_reordered_t0,
                boxes_reordered_t1=boxes_reordered_t1,
                sensor_Tt0_box=sensor_Tt0_box,
                sensor_Tt1_box=sensor_Tt1_box,
            )
            sample_content[f"flow_{src_key}_{target_key}"] = lidar_flow_t0_t1.astype(
                np.float32
            )
            sample_content[f"flow_{target_key}_{src_key}"] = lidar_flow_t1_t0.astype(
                np.float32
            )

            if self.cfg.data.flow_source != "gt":
                self.load_add_flow_to_sample_content(
                    fname_t0,
                    sample_content,
                    src_key,
                    target_key,
                    use_path_stem_only=False,
                )
        sample_content[f"objects_{src_key}"] = boxes_t0
        sample_content[f"class_names_{src_key}"] = obj_names_t0
        sample_content[f"objects_{target_key}"] = boxes_t1
        sample_content[f"class_names_{target_key}"] = obj_names_t1
        sample_content = self.move_keys_to_subdict(sample_content)
        if self.cfg.data.odom_source == "kiss_icp":
            sample_content["kiss_icp"] = {}
            scene_name = label_infos_t0["frame_id"][:-4]
            w_Tt0_lidar = self.kiss_icp_poses[scene_name][fname_t0.as_posix()]
            w_Tt1_lidar = self.kiss_icp_poses[scene_name][fname_t1.as_posix()]
            kiss_odom_t0_t1 = np.linalg.inv(w_Tt0_lidar) @ w_Tt1_lidar
            sample_content["kiss_icp"][f"odom_{src_key}_{target_key}"] = kiss_odom_t0_t1
            sample_content["kiss_icp"][f"odom_{target_key}_{src_key}"] = np.linalg.inv(
                kiss_odom_t0_t1
            )

        add_lidar_rows_to_waymo_sample(
            sample_content, time_keys=(src_key,), pcl_key="pcl_"
        )
        if not self.pure_inference_mode:  # we only have slim flow for train dataset
            if self.mined_boxes_db is not None:
                # assert src_key == "t0", src_key
                # num_skip_files = int(target_key[-1])
                # assert num_skip_files in (1, 2), {target_key: num_skip_files}
                # target_fname =

                self.load_add_mined_boxes_to_sample_content(
                    fname_t0.as_posix(),
                    sample_content,
                )
        if (
            not self.pure_inference_mode and self.mode == "train"
        ):  # we only have slim flow for train dataset
            if self.use_geom_augmentation and self.cfg.data.augmentation.active:
                self.augment_sample_content(
                    sample_content,
                    src_key,
                    target_key,
                    "waymo",
                )
        sample_content["gt"]["objects_t0"].assert_attr_shapes_compatible()

        sample_data_ta = self.assemble_sample_data(
            deepcopy(sample_content), src_key, target_key, src_trgt_time_delta_s
        )
        if self.need_reverse_time_sample_data:
            sample_data_tb = self.assemble_sample_data(
                sample_content, target_key, src_key, src_trgt_time_delta_s
            )
        else:
            sample_data_tb = {"gt": {}}
        for sd in (sample_data_ta, sample_data_tb):
            # cannot convert to tensor
            sd.pop("class_names_ta", None)
            sd.pop("class_names_tb", None)
            sd["gt"].pop("objects_ta", None)
            sd["gt"].pop("objects_tb", None)

        if (
            self.pure_inference_mode
            or not self.use_geom_augmentation
            or not self.cfg.data.augmentation.active
        ):
            # need ta instead of t0 here because of remapping
            if "kitti_lid_ta_T_tx_kitti_lid" in sample_data_ta:
                sample_data_ta["gt"]["odom_ta_tx"] = sample_data_ta.pop(
                    "kitti_lid_ta_T_tx_kitti_lid"
                )
            else:
                sample_data_ta["gt"]["odom_ta_tx"] = np.nan * np.eye(5)

            if "pcl_tx" in sample_content:
                sample_data_tx = self.assemble_sample_data(
                    deepcopy(sample_content),
                    "tx",
                    "ta",
                    remap_src_target_keys_to_ta_tb=False,
                    src_trgt_time_delta_s=0.5,
                )
                sample_data_ta["pcl_tx"] = sample_data_tx["pcl_tx"]
                sample_data_ta["pcl_full_tx"] = sample_data_tx["pcl_full_tx"]
                sample_data_tb.pop("pcl_tx", None)  # otherwise collate will fail
        else:
            # don't need the _tx samples, delete if there
            sample_data_ta.pop("pcl_tx", None)
            sample_data_tb.pop("pcl_tx", None)

        if self.verbose:
            print("Loaded sample: {0}".format(fname_t0))

        if "pcl_tx" in sample_data_ta:
            assert isinstance(sample_data_ta["pcl_tx"], abc.Mapping)
        if "pcl_tx" in sample_data_tb:
            assert isinstance(sample_data_ta["pcl_tx"], abc.Mapping)

        sample_data_ta = recursive_npy_dict_to_torch(sample_data_ta)
        if self.cfg.loss.supervised.centermaps.active:
            sample_data_ta["gt"].update(
                self.get_motion_based_centermaps(sample_data_ta)
            )

        sample_data_tb = recursive_npy_dict_to_torch(sample_data_tb)
        if (
            self.need_reverse_time_sample_data
            and self.cfg.loss.supervised.centermaps.active
        ):
            sample_data_tb["gt"].update(
                self.get_motion_based_centermaps(sample_data_tb)
            )
        if self.mode == "train":
            augm_sample_ta = self.create_augmented_sample_from_flow_cluster_detector_and_box_snippet_db(
                src_trgt_time_delta_s,
                sample_data_ta,
            )
        else:
            augm_sample_ta = {}
        meta = {}
        meta["sample_id"] = str(fname_t0)
        return (
            sample_data_ta,
            sample_data_tb,
            augm_sample_ta,
            meta,
        )

    def get_odometry_from_label_infos(self, label_infos_t0, label_infos_t1):
        w_T_t0 = (
            label_infos_t0["pose"].astype(np.float64) @ vehicle_Twaymo_lidar
        )  # transform to LiDAR
        w_T_t1 = (
            label_infos_t1["pose"].astype(np.float64) @ vehicle_Twaymo_lidar
        )  # transform to LiDAR
        odom_t0_t1 = np.linalg.inv(w_T_t0) @ w_T_t1
        odom_t1_t0 = np.linalg.inv(odom_t0_t1)
        return odom_t0_t1, odom_t1_t0

    def load_precomputed_waymo_gt_flow(
        self,
        *,
        pcl_t0: np.ndarray,
        pcl_t1: np.ndarray,
        fname_t0: Path,
        fname_t1: Path,
        odom_t0_t1: np.ndarray,
        odom_t1_t0: np.ndarray,
        boxes_reordered_t0: Shape,
        boxes_reordered_t1: Shape,
        sensor_Tt0_box: np.ndarray,
        sensor_Tt1_box: np.ndarray,
    ):
        try:
            gt_flow_filename = self.flow_gt_root.joinpath(
                fname_t0.parent, fname_t0.stem + "_" + fname_t1.stem
            ).with_suffix(".npy")
            flow_gt = self.loader_saver_helper.load_sample(
                gt_flow_filename, np.load, allow_pickle=True
            ).item()
            lidar_flow_t0_t1 = flow_gt["flow_t0_t1"]
            assert lidar_flow_t0_t1.shape[0] == pcl_t0.shape[0], (
                lidar_flow_t0_t1.shape[0],
                pcl_t0.shape[0],
            )
            lidar_flow_t1_t0 = flow_gt["flow_t1_t0"]
            assert lidar_flow_t1_t0.shape[0] == pcl_t1.shape[0], (
                lidar_flow_t1_t0.shape[0],
                pcl_t1.shape[0],
            )

        except (IOError, pickle.UnpicklingError):
            gt_flow_filename.parent.mkdir(exist_ok=True, parents=True)

            homog_pcl_t0 = homogenize_pcl(pcl_t0[:, :3])
            lidar_flow_t0_t1 = self.get_flow_waymo(
                homog_pcl_t0,
                odom_t1_t0,
                boxes_reordered_t0,
                sensor_Tt0_box,
                sensor_Tt1_box,
            )

            lidar_flow_t1_t0 = self.get_flow_waymo(
                homogenize_pcl(pcl_t1[:, :3]),
                odom_t0_t1,
                boxes_reordered_t1,
                sensor_Tt1_box,
                sensor_Tt0_box,
            )
            flow_gt = {
                "flow_t0_t1": lidar_flow_t0_t1,
                "flow_t1_t0": lidar_flow_t1_t0,
            }
            np.save(gt_flow_filename, flow_gt)
        return lidar_flow_t0_t1, lidar_flow_t1_t0

    def get_is_ground_mask(self, pcl, pcl_fname):
        try:
            groundseg_fname = self.groundsegmentation_root.joinpath(
                pcl_fname
            ).with_suffix(".npy")
            is_ground = self.loader_saver_helper.load_sample(groundseg_fname, np.load)
        except (FileNotFoundError, ValueError) as e:
            if isinstance(e, ValueError):
                print(e)
            is_ground = JPCGroundRemove(
                pcl=pcl[:, :3],
                range_img_width=2000,
                range_img_height=60,
                sensor_height=1.8,
                delta_R=2,
            )
            groundseg_fname.parent.mkdir(exist_ok=True, parents=True)
            # np.save(groundseg_fname, is_ground, allow_pickle=False)
            try:
                self.loader_saver_helper.save_sample(
                    np.save, groundseg_fname, is_ground, allow_pickle=False
                )
            except Exception as e:
                print(
                    f"Failed to save ground segmentation: {groundseg_fname}. I am ignoring this for now."
                )
                print(e)
        assert pcl.shape[0] == is_ground.shape[0], (pcl.shape, is_ground.shape)

        return is_ground

    def get_flow_waymo(
        self,
        homog_pcl_t0: np.ndarray,
        odom_t1_t0: np.ndarray,
        boxes_reordered_t0: Shape,
        sensor_Tt0_box: np.ndarray,
        sensor_Tt1_box: np.ndarray,
    ) -> np.ndarray:
        assert np.allclose(homog_pcl_t0[:, -1], 1.0), "need homog coords"
        pts_homog_in_box = np.einsum(
            "kij,nj->nki", np.linalg.inv(sensor_Tt0_box), homog_pcl_t0
        )
        assert np.all(np.isfinite(pts_homog_in_box))
        pt_is_in_box = np.all(
            np.abs(pts_homog_in_box[..., 0:3])
            < 0.5 * boxes_reordered_t0.dims[None, ...],
            axis=-1,
        )
        lidar_flow_t0_t1 = np.einsum(
            "ij,kj->ki",
            odom_t1_t0 - np.eye(4),
            homog_pcl_t0,
        )[:, :3]
        flow_trafos_t0_t1 = sensor_Tt1_box @ np.linalg.inv(sensor_Tt0_box)
        for box_idx in range(flow_trafos_t0_t1.shape[0]):
            pts_in_box_mask = pt_is_in_box[:, box_idx]
            # prev_flow = lidar_flow_t0_t1[pts_in_box_mask]
            dyn_flows_t0_t1 = np.einsum(
                "ij,nj->ni",
                flow_trafos_t0_t1[box_idx] - np.eye(4),
                homog_pcl_t0[pts_in_box_mask],
            )[:, :3]

            lidar_flow_t0_t1[pts_in_box_mask] = dyn_flows_t0_t1

            # mean_prev_flow = prev_flow.mean(axis=0)
            # mean_new_flow = dyn_flows_t0_t1.mean(axis=0)
            # print("foo", np.count_nonzero(pts_in_box_mask))

        return lidar_flow_t0_t1.astype(np.float32)

    def get_label_idxs_from_label_name(self, label_name: np.ndarray) -> np.ndarray:
        return np.array(
            [self.class_name_to_idx_mapping[ln] for ln in label_name],
        ).astype(np.int32)

    def get_nonempty_flow_relevant_boxes_into_lidar_frame(self, label_infos_t0):
        lidar_boxes_t0 = label_infos_t0["annos"]["gt_boxes_lidar"]
        label_names_t0 = label_infos_t0["annos"]["name"].astype(str)
        num_pts_in_boxes_t0 = label_infos_t0["annos"]["num_points_in_gt"]
        box_velo_mps = label_infos_t0["annos"]["speed_global"]
        if box_velo_mps.size == 0:
            box_velo_mps = np.empty((0, 3), dtype=box_velo_mps.dtype)
        difficulty = label_infos_t0["annos"]["difficulty"].astype(np.int32)
        obj_ids_t0 = label_infos_t0["annos"]["obj_ids"].astype(str)
        is_flow_anno_box_t0 = np.any(
            # this creates a warning when label_names_t0 = [] (empty list), but works as expected
            label_names_t0[:, None] == np.array(WAYMO_MOVABLE_CLASSES)[None, :],
            axis=-1,
        )
        has_points_in_box = num_pts_in_boxes_t0 > 0
        is_flow_anno_box_w_points_t0 = is_flow_anno_box_t0 & has_points_in_box
        lidar_boxes_t0 = lidar_boxes_t0[is_flow_anno_box_w_points_t0]
        label_names_t0 = label_names_t0[is_flow_anno_box_w_points_t0]
        num_pts_in_boxes_t0 = num_pts_in_boxes_t0[is_flow_anno_box_w_points_t0]
        obj_ids_t0 = obj_ids_t0[is_flow_anno_box_w_points_t0]
        # difficulty: 0: easy, 2: hard
        difficulty = difficulty[is_flow_anno_box_w_points_t0]
        class_ids = self.get_label_idxs_from_label_name(label_names_t0)
        box_velo_mps = np.linalg.norm(
            box_velo_mps[is_flow_anno_box_w_points_t0], axis=-1, keepdims=True
        )

        shapes = Shape(
            pos=lidar_boxes_t0[:, :3],
            dims=lidar_boxes_t0[:, 3:6],
            rot=lidar_boxes_t0[:, [6]],
            probs=np.ones_like(lidar_boxes_t0[:, [0]]),
            velo=box_velo_mps,  # TODO: why was box velo zeros?
            difficulty=difficulty[..., None],
            class_id=class_ids[..., None],
        )
        shapes.assert_attr_shapes_compatible()
        veh_T_obj = shapes.get_poses()
        li_T_obj = np.linalg.inv(vehicle_Twaymo_lidar) @ veh_T_obj
        pos_li = li_T_obj[:, :3, 3]
        assert np.array_equal(
            vehicle_Twaymo_lidar[:3, :3], np.eye(3)
        ), "rotation not supported, decompose matrix  li_T_obj to get it"
        shapes.pos = pos_li
        return shapes, label_names_t0, obj_ids_t0

    def get_has_valid_scene_flow_label(self, sample_content, src_key):
        return np.ones_like(sample_content[f"pcl_{src_key}"]["pcl"][:, 0], dtype=bool)


def add_lidar_rows_to_waymo_sample(
    sample_content, time_keys: Tuple[str], pcl_key="pcl_"
):
    for tk in time_keys:
        pcl_key = f"pcl_{tk}"
        if pcl_key in sample_content:
            pcl = sample_content[f"pcl_{tk}"]
            elevation_anges_rad = np.arctan2(
                pcl[:, 2], np.linalg.norm(pcl[:, :2], axis=-1)
            )

            bins = np.linspace(
                elevation_anges_rad.min(), elevation_anges_rad.max(), num=64
            )

            row_idxs = np.digitize(elevation_anges_rad, bins)

            sample_content[f"lidar_rows_{tk}"] = row_idxs.astype(np.uint8)


def get_waymo_train_dataset(
    cfg,
    use_geom_augmentation: bool,
    use_skip_frames: str,
    shuffle=True,
    size=None,
    verbose=False,
    get_only_these_specific_samples=None,
    path_to_augmentation_db: str = None,
    path_to_mined_boxes_db: str = None,
    need_flow_during_training=True,
):
    extra_loader_kwargs = {"shuffle": shuffle}
    train_dataset = WaymoDataset(
        shuffle=False,
        use_geom_augmentation=use_geom_augmentation,
        use_skip_frames=use_skip_frames,
        mode="train",
        cfg=cfg,
        size=size,
        verbose=verbose,
        get_only_these_specific_samples=get_only_these_specific_samples,
        path_to_augmentation_db=path_to_augmentation_db,
        path_to_mined_boxes_db=path_to_mined_boxes_db,
        need_flow=need_flow_during_training,
    )
    if path_to_mined_boxes_db is not None:
        sample_file_stems = []
        for flat_idx in range(len(train_dataset)):
            (
                _,  # segment_idx,
                _,  # frame_idx,
                _,  # src_key,
                _,  # target_key,
                _,  # num_skip_frames,
                _,  # label_infos_t0,
                sample_id,
            ) = train_dataset.map_flat_index_to_meta_data(flat_idx)
            sample_file_stems.append(str(sample_id))
        num_idx_checks = 20
        print(
            "Checking, that the indexes match with the sample ids - otherwise we drop the wrong samples"
        )
        check_idxs = np.random.choice(
            np.arange(len(train_dataset)), size=num_idx_checks, replace=False
        )
        for check_idx in check_idxs:
            _, _, _, meta = train_dataset.__getitem__(check_idx)
            assert meta["sample_id"] == sample_file_stems[check_idx], (
                meta["sample_id"] == sample_file_stems[check_idx]
            )
        weighted_random_sampler = (
            get_weighted_random_sampler_dropping_samples_without_boxes(
                path_to_mined_boxes_db,
                extra_loader_kwargs,
                train_dataset,
                sample_file_stems,
            )
        )

        extra_loader_kwargs["sampler"] = weighted_random_sampler
    # subset = torch.utils.data.Subset(
    #     train_dataset, np.arange(start=39110, stop=158081, step=1)
    # )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        pin_memory=True,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=lidar_dataset_collate_fn,
        worker_init_fn=worker_init_fn,
        **extra_loader_kwargs,
    )

    return train_loader, train_dataset


def get_waymo_val_dataset(
    cfg,
    size,
    use_skip_frames: str,
    batch_size=None,
    pure_inference_mode=False,
    shuffle=False,
    need_flow=True,
    get_only_these_specific_samples=None,
):
    prefetch_args = {}
    if batch_size is None:
        batch_size = cfg.data.batch_size
    val_dataset = WaymoDataset(
        use_geom_augmentation=False,
        use_skip_frames=use_skip_frames,
        size=size,
        mode="val",
        shuffle=False,  # cant do this for waymo yet
        cfg=cfg,
        pure_inference_mode=pure_inference_mode,
        need_flow=need_flow,
        get_only_these_specific_samples=get_only_these_specific_samples,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=lidar_dataset_collate_fn,
        shuffle=shuffle,
        worker_init_fn=lambda id: np.random.seed(id + cfg.data.num_workers),
        **prefetch_args,
    )
    return val_loader


def main():
    from config_helper.config import parse_config

    default_cfg_file = (
        Path(getsourcefile(lambda: 0)).resolve().parent.parent
        / Path("config")
        / Path("liso_config.yml")
    )

    print(f"Loading config {default_cfg_file}")

    cfg = parse_config(
        default_cfg_file,
    )

    for mode in ("train", "val"):
        ds = WaymoDataset(
            cfg,
            shuffle=False,
            mode=mode,
            use_geom_augmentation=False,
            use_skip_frames="never",
        )
        ds.generate_save_kiss_icp_poses_groundseg()


if __name__ == "__main__":
    main()
