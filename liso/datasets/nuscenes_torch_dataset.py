import itertools
import time
from collections import abc, defaultdict
from copy import deepcopy
from functools import cached_property
from glob import glob
from inspect import getsourcefile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from liso.datasets.torch_dataset_commons import (
    LidarDataset,
    LidarSample,
    get_weighted_random_sampler_dropping_samples_without_boxes,
    lidar_dataset_collate_fn,
    recursive_npy_dict_to_torch,
    worker_init_fn,
)
from liso.kabsch.shape_utils import UNKNOWN_CLASS_ID, Shape
from liso.transformations.transformations import decompose_matrix
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset

NUSC_MOVABLE_CLASSES = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
)
NUSC_NON_MOVABLE_CLASSES = (
    "traffic_cone",
    "barrier",
)

NUSC_CLASSES = NUSC_MOVABLE_CLASSES + NUSC_NON_MOVABLE_CLASSES

NUSC_MOVABLE_CLASS_FREQUENCIES = {
    "pedestrian": 210306,  # 208240 + 2066,
    "car": 493322,
    "motorcycle": 12617,
    "trailer": 24860,
    "truck": 88519,
    "bicycle": 11859,
    "bus": 14501 + 1820,
    "construction_vehicle": 14671,
}


class NuscenesDataset(LidarDataset):
    movable_class_names = NUSC_MOVABLE_CLASSES
    class_names = NUSC_CLASSES
    class_name_to_idx_mapping = {k: i for i, k in enumerate(class_names)}
    idx_to_class_name_mapping = dict(enumerate(class_names))
    movable_class_frequencies = np.array(
        [NUSC_MOVABLE_CLASS_FREQUENCIES[k] for k in movable_class_names]
    ) / sum(NUSC_MOVABLE_CLASS_FREQUENCIES.values())

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
            shuffle=shuffle,
            mode=mode,
            use_geom_augmentation=use_geom_augmentation,
            use_skip_frames=use_skip_frames,
            path_to_augmentation_db=path_to_augmentation_db,
            path_to_mined_boxes_db=path_to_mined_boxes_db,
            for_tracking=for_tracking,
            need_flow=need_flow,
        )

        self.verbose = verbose
        self.pure_inference_mode = pure_inference_mode
        dataset_root = Path(cfg.data.paths.nuscenes.local).joinpath(mode)
        sample_files = sorted(glob(str(Path(dataset_root).joinpath("*.npy"))))
        if pure_inference_mode:
            assert not self.use_geom_augmentation, self.use_geom_augmentation
            self.sample_files = sample_files
        else:
            if mode == "val":
                # TODO: GENERATE INVALID FOLLOWUP SAMPLES
                # from tqdm import tqdm
                # bad_samples = []
                # for fname in tqdm(sample_files):
                #    file_content = np.load(fname, allow_pickle=True).item()
                #    valid_file = True
                #    for necessary_attr in (
                #        "pcl_t0",
                #        "pcl_t1",
                #    ):
                #        if necessary_attr not in file_content:
                #            valid_file = False
                #            break
                #    if not valid_file:
                #        bad_samples.append(Path(fname).stem)
                # with open(
                #    "val_samples_with_insufficient_following_samples.yaml", "a"
                # ) as the_file:
                #    for item in bad_samples:
                #        the_file.write(f"- {item}\n")
                # print(len(bad_samples))

                with open(
                    Path(getsourcefile(lambda: 0)).parent.parent
                    / Path("config")
                    / Path("nusc_val_samples_with_insufficient_following_samples.yaml"),
                    "r",
                ) as stream:
                    val_samples_w_insufficient_followup_samples = yaml.safe_load(stream)

                num_samples_total = len(sample_files)
                num_broken_samples = len(val_samples_w_insufficient_followup_samples)
                # filter samples whcih cannot be used during non pure pass through pipeline
                valid_samples = list(
                    itertools.filterfalse(
                        lambda x: Path(x).stem
                        in val_samples_w_insufficient_followup_samples,
                        sample_files,
                    )
                )
                assert len(valid_samples) == num_samples_total - num_broken_samples
                print(
                    f"Dropped {num_broken_samples}/{num_samples_total} files from VAL set due to insufficient follow up samples. Remaining: {len(valid_samples)}"
                )
                sample_files = valid_samples

            if size is not None:
                if size < len(sample_files):
                    self.sample_files = np.random.choice(
                        sample_files, size=size, replace=False
                    ).tolist()
                else:
                    self.sample_files = sample_files
                    print(
                        f"Warning: Requested size={size} samples, but I only have {len(sample_files)}"
                    )
            else:
                self.sample_files = sample_files

        if get_only_these_specific_samples:
            filtered_samples = []
            for sample in self.sample_files:
                for special_sample in get_only_these_specific_samples:
                    if special_sample in sample:
                        filtered_samples.append(sample)
            self.sample_files = filtered_samples
            if size is not None:
                assert size == len(
                    filtered_samples
                ), "either stop requesting specific samples or request size=None"
            print(f"Warning: Only requested {len(self.sample_files)}")

        if self.cfg.data.flow_source != "gt":
            pred_flow_path = self.get_pred_flow_path()
            self.pred_flow_path = pred_flow_path
            print(f"Loading flow seperately from source {self.pred_flow_path}")
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

        if self.shuffle:
            assert self.mode != "train", self.mode
            np.random.shuffle(self.sample_files)
        # see thread https://github.com/pytorch/pytorch/issues/13246#issuecomment-715050814

        downsample_dataset_keep_ratio = self.cfg.data.setdefault(
            "downsample_dataset_keep_ratio", 1.0
        )
        if self.mode == "train" and self.cfg.data.downsample_dataset_keep_ratio != 1.0:
            self.dataset_sequence_is_messed_up = True
            assert downsample_dataset_keep_ratio < 1.0, downsample_dataset_keep_ratio
            print(
                f"Downsampling dataset by {downsample_dataset_keep_ratio}. From {len(self.sample_files)} to.."
            )
            assert not self.shuffle, "probably broken when shuffling"

            selected_idxs = np.load(
                Path(getsourcefile(lambda: 0)).parent.parent
                / "assets"
                / "nuscenes_selected_samples.npz",
                allow_pickle=True,
            )["arr_0"][: int(downsample_dataset_keep_ratio * len(self.sample_files))]
            self.sample_files = np.array(self.sample_files)[selected_idxs].tolist()

            # self.sample_files = np.random.choice(
            #     self.sample_files,
            #     size=int(downsample_dataset_keep_ratio * len(self.sample_files)),
            #     replace=False,
            # ).tolist()
            print(f"... {len(self.sample_files)} samples.")
            print("preloading databses!")
            self.initialize_dbs_if_necessary()

        self.sample_files = np.array(self.sample_files).astype(np.string_)
        self.data_load_times = []

    @cached_property
    def sequence_sample_mapping(self) -> Dict[str, List[str]]:
        mapping = defaultdict(list)
        assert not self.dataset_sequence_is_messed_up
        for sample_idx, full_path in enumerate(self.sample_files):
            full_path = str(full_path, encoding="utf-8")
            sample_name = Path(full_path).stem
            scene_name, timestamp_str = sample_name.split("_")[:2]
            mapping[scene_name].append(
                LidarSample(
                    sample_idx,
                    sample_name,
                    timestamp=int(timestamp_str),
                    full_path=full_path,
                )
            )

        return mapping

    @cached_property
    def sequence_lens(self):
        assert not self.dataset_sequence_is_messed_up
        assert not self.shuffle, "probably broken when shuffling"
        # this is just dummy values so that we get the total number of sequences
        return [None for _, _ in self.sequence_sample_mapping.items()]

    def get_consecutive_sample_idxs_for_sequence(
        self,
        sequence_idx: int,
    ) -> Tuple[LidarSample]:
        assert not self.shuffle
        assert not self.dataset_sequence_is_messed_up
        sequence_ids = sorted(self.sequence_sample_mapping.keys())
        if sequence_idx < len(sequence_ids):
            sequence_id = sequence_ids[sequence_idx]
        else:
            print("Ran out of sequences!")
            return None
        samples_in_sequence = self.sequence_sample_mapping[sequence_id]
        return samples_in_sequence

    def object_is_movable(self, obj_category: str) -> bool:
        is_movable = False
        if obj_category not in NuScenesDataset.NameMapping:
            # weird objects -> ignore
            return False
        simple_name = NuScenesDataset.NameMapping[obj_category]
        if simple_name in NUSC_MOVABLE_CLASSES:
            is_movable = True
        return is_movable

    def get_label_idxs_from_label_name(self, object_categories: List[str]):
        label_idxs = []
        for obj_category in object_categories:
            if obj_category not in NuScenesDataset.NameMapping:
                label_idxs.append(UNKNOWN_CLASS_ID)
            else:
                simple_name = NuScenesDataset.NameMapping[obj_category]
                cls_idx = self.class_name_to_idx_mapping[simple_name]
                label_idxs.append(cls_idx)
        return np.array(label_idxs).astype(np.int32)

    def extract_boxes_for_timestamp(
        self,
        sample_content: Dict[str, np.ndarray],
        src_key: str,
        target_key: str,
    ) -> Shape:
        objects = sample_content["gt"].get("objects", None)
        get_objects = (
            objects is not None
            and len(objects) > 0
            and all([f"pose_{src_key}" in obj for obj in objects])
        )
        if get_objects:
            obj_pose_ta = np.stack([obj[f"pose_{src_key}"] for obj in objects], axis=0)
            obj_pos_ta = obj_pose_ta[:, 0:3, 3]
            obj_rot_ta = np.stack(
                [decompose_matrix(obj[f"pose_{src_key}"])[2][2] for obj in objects],
                axis=0,
            )[..., None]
            obj_dims = np.stack([obj["size"][0:3] for obj in objects], axis=0)
            obj_probs_ta = np.ones_like(obj_rot_ta)

            target_pose_key = f"pose_{target_key}"
            if f"odom_{src_key}_{target_key}" in sample_content["gt"] and all(
                target_pose_key in obj for obj in objects
            ):
                obj_pose_tb = np.stack(
                    [obj[f"pose_{target_key}"] for obj in objects], axis=0
                )
                obj_velo_ta_tb = np.linalg.norm(
                    self.get_object_velocity_in_obj_coords(
                        sample_content["gt"][f"odom_{src_key}_{target_key}"],
                        obj_pose_ta,
                        obj_pose_tb,
                    )[:, :3],
                    axis=-1,
                    keepdims=True,
                )
            else:
                obj_velo_ta_tb = np.zeros_like(obj_probs_ta)

            class_names = np.array([obj["category"] for obj in objects])
            class_idxs = self.get_label_idxs_from_label_name(class_names)
            gt_objects = Shape(
                pos=obj_pos_ta,
                dims=obj_dims,
                rot=obj_rot_ta,
                probs=obj_probs_ta,
                velo=obj_velo_ta_tb,
                class_id=class_idxs[..., None],
                valid=np.ones_like(np.squeeze(obj_probs_ta, axis=-1), dtype=bool),
            )
        else:
            gt_objects = Shape.createEmpty()
            class_names = np.array((), dtype=str)

        return gt_objects, class_names

    def __getitem__(self, index):
        self.initialize_loader_saver_if_necessary()

        self.initialize_dbs_if_necessary()

        fname = str(self.sample_files[index], encoding="utf-8")
        num_profiling_samples = 10
        if index < num_profiling_samples:
            start_time = time.time()

        sample_content = self.loader_saver_helper.load_sample(
            fname, np.load, allow_pickle=True
        ).item()

        if index < num_profiling_samples:
            end_time = time.time()
            loading_time = end_time - start_time
            self.data_load_times.append(loading_time)
            # print(
            #     f"Loading takes {loading_time} (avg: {np.mean(self.data_load_times)})"
            # )
        assert "sample_id" not in sample_content["meta_data_t0"]
        meta = {"sample_id": Path(fname).stem, **sample_content["meta_data_t0"]}

        sample_content["odom_t0_tx"] = sample_content["kitti_lid_t0_T_tx_kitti_lid"]
        sample_content["kiss_odom_t0_tx"] = sample_content.pop(
            "kitti_lid_t0_Tkiss_icp_tx_kitti_lid"
        )

        if self.cfg.data.use_lidar_intensity:
            self.concatenate_nuscenes_intensities_to_pcl(sample_content)
        if self.for_tracking:
            src_key = "t0"
            target_key = "t1"
            src_trgt_time_delta_s = 0.1
            self.drop_unused_timed_keys_from_sample(sample_content, "foo", "bar", "baz")
        else:
            (
                src_key,
                target_key,
                delete_target_key,
                src_trgt_time_delta_s,
            ) = self.select_time_keys()
            self.drop_unused_timed_keys_from_sample(
                sample_content, src_key, target_key, delete_target_key
            )
        # restructure_sample
        sample_content = self.move_keys_to_subdict(
            sample_content,
            move_these_keys=("kiss_",),
            subdict_target_key="kiss_icp",
            drop_substr_from_moved_keys="kiss_",
        )
        sample_content = self.move_keys_to_subdict(sample_content)
        self.add_reverse_odometry_to_sample(sample_content)
        # for sk in ("t0", "t1", "t2"):
        #     for tk in ("t0", "t1", "t2"):
        #         if sk == tk:
        #             continue
        #         else:
        #             assert np.allclose(
        #                 sample_content["gt"][f"odom_{sk}_{tk}"],
        #                 np.linalg.inv(sample_content["gt"][f"odom_{tk}_{sk}"]),
        #             )

        if not self.pure_inference_mode:  # we only have slim flow for train dataset
            if self.mined_boxes_db is not None:
                # assert src_key == "t0", src_key
                # num_skip_files = int(target_key[-1])
                # assert num_skip_files in (1, 2), {target_key: num_skip_files}
                # target_fname =

                self.load_add_mined_boxes_to_sample_content(
                    Path(fname).stem,
                    sample_content,
                )

            if self.need_flow and self.cfg.data.flow_source != "gt":
                self.load_add_flow_to_sample_content(
                    fname, sample_content, src_key, target_key
                )
                if self.for_tracking:
                    # also add flow from t1->t2
                    assert src_key == "t0", src_key
                    assert target_key == "t1", target_key
                    self.load_add_flow_to_sample_content(
                        fname, sample_content, src_key="t1", target_key="t2"
                    )
            if (
                not self.for_tracking  # WE CANT AUGMENT WHEN TRACKING!
                and self.use_geom_augmentation
                and self.cfg.data.augmentation.active
                and self.mode == "train"
            ):
                # TODO, REFACTOR THIS TO HANDLE AUGM
                self.augment_sample_content(
                    sample_content,
                    src_key,
                    target_key,
                    "nuscenes",
                )
        if self.for_tracking:
            sample_data_ta = self.assemble_sample_data(
                deepcopy(sample_content), "t0", "t1", src_trgt_time_delta_s
            )
            if self.need_reverse_time_sample_data:
                sample_data_tb = self.assemble_sample_data(
                    sample_content, "t1", "t2", src_trgt_time_delta_s
                )
            else:
                sample_data_tb = {"gt": {}}
            for gt_source in {"gt", self.cfg.data.flow_source}:
                self.drop_unused_timed_keys_from_sample(
                    sample_data_ta[gt_source],
                    "ta",
                    "tb",
                    "t2",  # stuff will have been remapped
                )
                self.drop_unused_timed_keys_from_sample(
                    sample_data_tb[gt_source],
                    "ta",
                    "tb",
                    "t0",  # stuff will have been remapped
                )
            return (
                recursive_npy_dict_to_torch(sample_data_ta),
                recursive_npy_dict_to_torch(sample_data_tb),
                {},
                meta,
            )
        else:
            sample_data_ta = self.assemble_sample_data(
                deepcopy(sample_content), src_key, target_key, src_trgt_time_delta_s
            )
            if self.need_reverse_time_sample_data:
                sample_data_tb = self.assemble_sample_data(
                    sample_content, target_key, src_key, src_trgt_time_delta_s
                )
            else:
                sample_data_tb = {"gt": {}}

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
                sample_data_ta["pcl_full_w_ground_tx"] = sample_data_tx[
                    "pcl_full_w_ground_tx"
                ]
                sample_data_ta["pcl_full_no_ground_tx"] = sample_data_tx[
                    "pcl_full_no_ground_tx"
                ]
                sample_data_tb.pop("pcl_tx", None)  # otherwise collate will fail
        else:
            # don't need the _tx samples, delete if there
            sample_data_ta.pop("pcl_tx", None)
            sample_data_tb.pop("pcl_tx", None)

        if self.verbose:
            print("Loaded sample: {0}".format(Path(fname).stem))

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
                src_trgt_time_delta_s, sample_data_ta
            )
        else:
            augm_sample_ta = {}
        return (
            sample_data_ta,
            sample_data_tb,
            augm_sample_ta,
            meta,
        )

    def concatenate_nuscenes_intensities_to_pcl(self, sample_content):
        for time_key in ("t0", "t1", "t2", "tx"):
            pcl_time_key = f"pcl_{time_key}"
            if pcl_time_key in sample_content:
                assert sample_content[pcl_time_key].shape[-1] == 3, sample_content[
                    pcl_time_key
                ].shape[-1]
                sample_content[pcl_time_key] = np.concatenate(
                    (
                        sample_content[pcl_time_key],
                        sample_content[f"lidar_intensities_{time_key}"][..., None]
                        / 255.0,
                    ),
                    axis=-1,
                )

    def get_has_valid_scene_flow_label(self, sample_content, src_key):
        return np.ones_like(sample_content[f"pcl_{src_key}"]["pcl"][:, 0], dtype=bool)


def get_nuscenes_train_dataset(
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
    train_dataset = NuscenesDataset(
        use_geom_augmentation=use_geom_augmentation,
        use_skip_frames=use_skip_frames,
        mode="train",
        cfg=cfg,
        size=size,
        verbose=verbose,
        get_only_these_specific_samples=get_only_these_specific_samples,
        path_to_augmentation_db=path_to_augmentation_db,
        path_to_mined_boxes_db=path_to_mined_boxes_db,
        shuffle=False,  # we leave shuffling to the loader for max randomness
        need_flow=need_flow_during_training,
    )

    if path_to_mined_boxes_db is not None:
        sample_file_stems = [
            Path(str(sf, encoding="utf-8")).stem for sf in train_dataset.sample_files
        ]
        weighted_random_sampler = (
            get_weighted_random_sampler_dropping_samples_without_boxes(
                path_to_mined_boxes_db,
                extra_loader_kwargs,
                train_dataset,
                sample_file_stems,
            )
        )

        extra_loader_kwargs["sampler"] = weighted_random_sampler

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


def get_nuscenes_val_dataset(
    cfg,
    size,
    use_skip_frames: str,
    batch_size=None,
    pure_inference_mode=False,
    shuffle=False,
):
    prefetch_args = {}
    if batch_size is None:
        batch_size = cfg.data.batch_size
    val_dataset = NuscenesDataset(
        use_geom_augmentation=False,
        use_skip_frames=use_skip_frames,
        shuffle=shuffle,
        size=size,
        mode="val",
        cfg=cfg,
        pure_inference_mode=pure_inference_mode,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=lidar_dataset_collate_fn,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        **prefetch_args,
    )
    return val_loader


def main():
    from config_helper.config import parse_config

    default_cfg_file = (
        Path(getsourcefile(lambda: 0)).parent.parent
        / Path("config")
        / Path("liso_config.yml")
    )

    cfg = parse_config(
        default_cfg_file,
    )
    test_ds = NuscenesDataset("/tmp/nusc_OD", cfg)
    num_workers = 0
    train_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=4,
        num_workers=0,
        collate_fn=lidar_dataset_collate_fn,
        worker_init_fn=lambda id: np.random.seed(id + num_workers),
    )
    test_el = next(iter(train_loader))
    print(test_el)


if __name__ == "__main__":
    main()
