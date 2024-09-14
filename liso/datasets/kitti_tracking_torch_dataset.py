from collections import defaultdict
from copy import deepcopy
from glob import glob
from inspect import getsourcefile
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from config_helper.config import parse_config
from liso.datasets.kitti_object_torch_dataset import KittiObjectDataset
from liso.datasets.torch_dataset_commons import (
    KITTI_IGNORE_NON_MOVABLE_CLASSMAPPING,
    KITTI_MAP_TO_SIMPLE_CLASSES,
    KITTI_MOVABLE_CLASSES,
    LidarDataset,
    LidarSample,
    add_lidar_rows_to_kitti_sample,
    lidar_dataset_collate_fn,
    recursive_npy_dict_to_torch,
    worker_init_fn,
)
from liso.kabsch.shape_utils import UNKNOWN_CLASS_ID, Shape


class KittiTrackingDataset(LidarDataset):
    movable_class_names = KITTI_MOVABLE_CLASSES
    class_names = KITTI_MOVABLE_CLASSES
    class_name_to_idx_mapping = {k: i for i, k in enumerate(class_names)}
    idx_to_class_name_mapping = dict(enumerate(class_names))

    def __init__(
        self,
        cfg,
        shuffle: bool,
        use_geom_augmentation: bool,
        use_skip_frames: str,
        mode="train",
        allow_data_augmentation=False,
        size=None,
        verbose=False,
        pure_inference_mode=False,
        get_only_these_specific_samples=None,
        path_to_augmentation_db=None,
        path_to_mined_boxes_db=None,
        for_tracking=False,
    ) -> None:
        super().__init__(
            cfg,
            mode=mode,
            shuffle=shuffle,
            use_geom_augmentation=use_geom_augmentation,
            use_skip_frames=use_skip_frames,
            path_to_augmentation_db=path_to_augmentation_db,
            path_to_mined_boxes_db=path_to_mined_boxes_db,
            for_tracking=for_tracking,  # this is a nuscenes hack...
            need_flow=True,
        )
        # assert mode == "val", "only mode val makes sense, but got {mode}"
        self.verbose = verbose
        self.pure_inference_mode = pure_inference_mode
        dataset_root = Path(cfg.data.paths.kitti.local)

        dataset_root = dataset_root.joinpath("kitti_tracking")
        sample_files = sorted(glob(str(Path(dataset_root).joinpath("*.npy"))))
        if pure_inference_mode:
            assert not allow_data_augmentation
            self.sample_files = sample_files
        else:
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
        sequences_with_sample_names = defaultdict(list)
        seq_samples = [  # sample_name: 0001_000000.npy
            (Path(el).stem.split("_")[0], Path(el)) for el in self.sample_files
        ]
        for seq_name, sample_name in seq_samples:
            sequences_with_sample_names[seq_name].append(sample_name)
        for seq_name in sequences_with_sample_names:
            sequences_with_sample_names[seq_name] = sorted(
                sequences_with_sample_names[seq_name]
            )
        self.per_seq_sample_paths = []
        for seq_name in sequences_with_sample_names:
            self.per_seq_sample_paths.append(sequences_with_sample_names[seq_name])

        self.sequence_lens = [len(el) for el in self.per_seq_sample_paths]

        print("sequence lengths: ", self.sequence_lens)

        self.allow_data_augmentation = allow_data_augmentation
        assert self.data_use_skip_frames in ("only", "both", "never")

        if self.cfg.data.flow_source != "gt":
            self.pred_flow_path = self.get_pred_flow_path()
            print(f"Loading flow seperately from source {self.pred_flow_path}")

        if self.shuffle:
            np.random.shuffle(self.sample_files)

    def get_samples_for_sequence(
        self,
        sequence_idx: int,
        start_idx_in_sequence: int,
        sequence_length: int,
    ) -> List[LidarSample]:
        assert not self.shuffle, self.shuffle
        global_start_idx_of_sequence = int(np.sum(self.sequence_lens[:sequence_idx]))
        global_start_idx = global_start_idx_of_sequence + start_idx_in_sequence
        # global_end_idx = global_start_idx + sequence_length
        chosen_samples = [
            LidarSample(
                idx=global_start_idx + idx,
                sample_name=str(
                    self.per_seq_sample_paths[sequence_idx][
                        start_idx_in_sequence + idx
                    ].stem
                ),
                timestamp=0,
                full_path=str(
                    self.per_seq_sample_paths[sequence_idx][start_idx_in_sequence + idx]
                ),
            )
            for idx in range(sequence_length)
        ]

        return chosen_samples

    def get_label_idxs_from_label_name(self, object_categories: List[str]):
        label_idxs = []
        for obj_category in object_categories:
            if KITTI_IGNORE_NON_MOVABLE_CLASSMAPPING[obj_category] is None:
                cls_idx = UNKNOWN_CLASS_ID
            else:
                simple_name = KITTI_MAP_TO_SIMPLE_CLASSES[obj_category]
                cls_idx = self.class_name_to_idx_mapping[simple_name]
            label_idxs.append(cls_idx)
        return np.array(label_idxs).astype(np.int32)

    def get_consecutive_sample_idxs_for_sequence(
        self,
        sequence_idx: int,
    ) -> List[LidarSample]:
        assert not self.shuffle, self.shuffle
        if sequence_idx >= len(self.sequence_lens):
            print("Ran out of sequences!")
            return None
        samples_in_sequence = self.get_samples_for_sequence(
            sequence_idx=sequence_idx,
            start_idx_in_sequence=0,
            sequence_length=self.sequence_lens[sequence_idx],
        )
        return samples_in_sequence

    def object_is_movable(self, obj_category) -> bool:
        return KITTI_IGNORE_NON_MOVABLE_CLASSMAPPING[obj_category] == "movable"

    def __getitem__(self, index):
        self.initialize_loader_saver_if_necessary()

        fname = self.sample_files[index]
        sample_content = self.loader_saver_helper.load_sample(
            fname, np.load, allow_pickle=True
        ).item()

        src_key, target_key, delete_target_key, src_trgt_time_delta_s = (
            "t0",
            "t1",
            "t2",
            0.1,
        )
        if not self.cfg.data.use_lidar_intensity:
            self.drop_intensities_from_pcls_in_sample(sample_content)

        self.drop_unused_timed_keys_from_sample(
            sample_content, src_key, target_key, delete_target_key
        )
        add_lidar_rows_to_kitti_sample(sample_content, time_keys=(src_key, target_key))

        sample_content = self.drop_points_on_kitti_vehicle(
            sample_content, src_key, target_key
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
        if self.cfg.data.flow_source != "gt":
            self.load_add_flow_to_sample_content(
                fname, sample_content, src_key, target_key
            )

        if (
            not self.pure_inference_mode
            and self.allow_data_augmentation
            and self.cfg.data.augmentation.active
        ):
            self.augment_sample_content(
                sample_content,
                src_key,
                target_key,
                "kitti",
            )

        meta = {"sample_id": sample_content["name"]}
        del sample_content["name"]
        sample_data_ta = self.assemble_sample_data(
            deepcopy(sample_content), src_key, target_key, src_trgt_time_delta_s
        )
        if self.need_reverse_time_sample_data:
            sample_data_tb = self.assemble_sample_data(
                sample_content, target_key, src_key, src_trgt_time_delta_s
            )
        else:
            sample_data_tb = {"gt": {}}
        if self.verbose:
            print("Loaded sample: {0}".format(Path(fname).stem))
        sample_data_ta["gt"].pop("objects_ta", None)
        sample_data_ta["gt"].pop("objects_tb", None)
        sample_data_tb.get("gt", {}).pop("objects_ta", None)
        sample_data_tb.get("gt", {}).pop("objects_tb", None)
        return (
            recursive_npy_dict_to_torch(sample_data_ta),
            recursive_npy_dict_to_torch(sample_data_tb),
            {},  # augmented_sample_data_ta,
            meta,
        )

    def get_valid_fov_labels_mask(self, pcl):
        assert (
            not self.allow_data_augmentation
        ), "fov limit makes no sense for augmented data due to shifted fov"

        point_yaw_angle = np.arctan2(pcl[:, 1], pcl[:, 0])
        has_label_info_available = np.abs(point_yaw_angle) < np.deg2rad(41.0)
        return has_label_info_available

    def get_has_valid_scene_flow_label(self, sample_content, src_key):
        return self.get_valid_fov_labels_mask(sample_content[f"pcl_{src_key}"]["pcl"])

    def extract_boxes_for_timestamp(
        self, sample_content: Dict[str, np.ndarray], src_key: str, target_key: str
    ) -> Shape:
        return self.kitti_extract_boxes_for_timestamp(sample_content, src_key)


def get_kitti_val_dataset(
    cfg,
    size,
    use_skip_frames="never",
    target="flow",
    batch_size=None,
    pure_inference_mode=False,
    shuffle=False,
    mode="val",
):
    prefetch_args = {}
    if batch_size is None:
        batch_size = cfg.data.batch_size
    dataset_kwargs = {
        "size": size,
        "mode": mode,
        "cfg": cfg,
        "use_skip_frames": use_skip_frames,
        "allow_data_augmentation": False,
        "use_geom_augmentation": False,
        "pure_inference_mode": pure_inference_mode,
        "shuffle": shuffle,
    }
    if target == "flow":
        val_dataset = KittiTrackingDataset(
            **dataset_kwargs,
        )
    elif target == "object":
        val_dataset = KittiObjectDataset(
            **dataset_kwargs,
        )
    else:
        raise NotImplementedError(target)
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
    return val_loader, val_dataset


def main():
    default_cfg_file = (
        Path(getsourcefile(lambda: 0)).parent.parent
        / Path("config")
        / Path("liso_config.yml")
    )

    cfg = parse_config(default_cfg_file, extra_cfg_args=("data_source_kitti",))
    train_ds = KittiTrackingDataset(cfg, allow_data_augmentation=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=1,
        num_workers=0,
        collate_fn=lidar_dataset_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    test_el = next(iter(train_loader))
    print(test_el)


if __name__ == "__main__":
    main()
