import time
from collections import defaultdict
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from liso.datasets.argoverse2.av2_classes import AV2_ALL_CLASSES, AV2_MOVABLE_CLASSES
from liso.datasets.torch_dataset_commons import (
    LidarDataset,
    LidarSample,
    get_weighted_random_sampler_dropping_samples_without_boxes,
    lidar_dataset_collate_fn,
    recursive_npy_dict_to_torch,
    worker_init_fn,
)
from liso.kabsch.shape_utils import UNKNOWN_CLASS_ID, Shape


class AV2Dataset(LidarDataset):
    movable_class_names = AV2_MOVABLE_CLASSES
    class_names = AV2_ALL_CLASSES
    class_name_to_idx_mapping = {k: i for i, k in enumerate(class_names)}
    idx_to_class_name_mapping = dict(enumerate(class_names))
    movable_class_frequencies = np.ones(len(movable_class_names)) / len(
        movable_class_names
    )

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
        dataset_root = Path(cfg.data.paths.av2.local).joinpath(mode)

        max_num_retries = 20
        success = False
        for retry in range(max_num_retries):
            # list all samples from unreliable mount
            try:
                sample_files = sorted(Path(dataset_root).rglob("**/*.npz"))
                success = True
            except Exception as e:
                print(f"Failed to load {dataset_root} on retry {retry}: {e}")
                time.sleep(10)

            if success:
                break

        if pure_inference_mode:
            assert not self.use_geom_augmentation, self.use_geom_augmentation
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

        if self.cfg.data.flow_source != "gt":
            pred_flow_path = self.get_pred_flow_path()
            self.pred_flow_path = pred_flow_path
            print(f"Loading flow seperately from source {self.pred_flow_path}")

        if self.shuffle:
            assert self.mode != "train", self.mode
            np.random.shuffle(self.sample_files)
        # see thread https://github.com/pytorch/pytorch/issues/13246#issuecomment-715050814

        if self.mode == "train" and self.cfg.data.downsample_dataset_keep_ratio != 1.0:
            self.dataset_sequence_is_messed_up = True
            raise NotImplementedError("not implemented yet")

        self.sample_files = np.array(self.sample_files).astype(np.string_)

    @cached_property
    def sequence_sample_mapping(self) -> Dict[str, List[str]]:
        mapping = defaultdict(list)
        assert not self.dataset_sequence_is_messed_up

        for sample_idx, full_path in enumerate(self.sample_files):
            full_path = str(full_path, encoding="utf-8")
            sample_name = "/".join(full_path.split("/")[-6:]).replace(
                ".npz", ".feather"
            )
            timestamp_str = Path(full_path).stem
            scene_name = full_path.split("/")[-4]
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
        if obj_category in AV2_MOVABLE_CLASSES:
            is_movable = True
        return is_movable

    def get_label_idxs_from_label_name(self, object_categories: List[str]):
        label_idxs = []
        for obj_category in object_categories:
            if obj_category not in AV2_ALL_CLASSES:
                label_idxs.append(UNKNOWN_CLASS_ID)
            else:
                cls_idx = self.class_name_to_idx_mapping[obj_category]
                label_idxs.append(cls_idx)
        return np.array(label_idxs).astype(np.int32)

    def extract_boxes_for_timestamp(
        self,
        sample_content: Dict[str, np.ndarray],
        src_key: str,
        target_key: str,
    ) -> Shape:
        boxes = sample_content["gt"].get(f"boxes_{src_key}", None)
        if boxes:
            class_names = sample_content["gt"][f"box_category_{src_key}"]
            class_idxs = self.get_label_idxs_from_label_name(class_names)
            boxes.class_id = class_idxs[..., None]
            gt_objects = boxes
        else:
            gt_objects = Shape.createEmpty()
            class_names = np.array((), dtype=str)

        return gt_objects, class_names

    def __getitem__(self, index):
        self.initialize_loader_saver_if_necessary()

        self.initialize_dbs_if_necessary()

        fname = str(self.sample_files[index], encoding="utf-8")

        sample_content = self.loader_saver_helper.load_sample(
            fname, np.load, allow_pickle=True
        )["arr_0"].item()

        sample_content["gt"]["boxes_t0"] = Shape(**sample_content["gt"]["boxes_t0"])
        sample_content["gt"]["boxes_t1"] = Shape(**sample_content["gt"]["boxes_t1"])

        assert "sample_id" not in sample_content["meta_data_t0"]
        meta = {"sample_id": sample_content.pop("meta_data_t0")}

        src_key = "t0"
        target_key = "t1"
        src_trgt_time_delta_s = 0.1

        # self.drop_unused_timed_keys_from_sample(
        #     sample_content, src_key, target_key, delete_target_key
        # )

        if not self.pure_inference_mode:  # we only have slim flow for train dataset
            if self.mined_boxes_db is not None:
                # assert src_key == "t0", src_key
                # num_skip_files = int(target_key[-1])
                # assert num_skip_files in (1, 2), {target_key: num_skip_files}
                # target_fname =

                self.load_add_mined_boxes_to_sample_content(
                    meta["sample_id"],
                    sample_content,
                )

            if self.need_flow and self.cfg.data.flow_source != "gt":
                flow_src_file = self.pred_flow_path / Path(fname).relative_to(
                    Path(fname).parents[5]
                )
                self.load_add_flow_to_sample_content(
                    fname,
                    sample_content,
                    src_key,
                    target_key,
                    specific_pred_flow_path=flow_src_file,
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
                    "av2",
                )
        sample_data_ta = self.assemble_sample_data(
            deepcopy(sample_content), "t0", "t1", src_trgt_time_delta_s
        )
        if self.for_tracking:
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
            if self.need_reverse_time_sample_data:
                sample_data_tb = self.assemble_sample_data(
                    sample_content, target_key, src_key, src_trgt_time_delta_s
                )
            else:
                sample_data_tb = {"gt": {}}
        sample_data_ta["gt"].pop("box_category_ta")
        sample_data_ta["gt"].pop("box_category_tb")
        if self.verbose:
            print("Loaded sample: {0}".format(Path(fname).stem))

        sample_data_ta = recursive_npy_dict_to_torch(sample_data_ta)
        if self.cfg.loss.supervised.centermaps.active:
            sample_data_ta["gt"].update(
                self.get_motion_based_centermaps(sample_data_ta)
            )

        if "gt" in sample_data_tb:
            sample_data_tb["gt"].pop("box_category_ta", None)
            sample_data_tb["gt"].pop("box_category_tb", None)

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

    def get_has_valid_scene_flow_label(self, sample_content, src_key):
        return np.ones_like(sample_content[f"pcl_{src_key}"]["pcl"][:, 0], dtype=bool)


def get_av2_train_dataset(
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
    train_dataset = AV2Dataset(
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


def get_av2_val_dataset(
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
    val_dataset = AV2Dataset(
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
