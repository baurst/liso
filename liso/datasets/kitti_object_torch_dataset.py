from glob import glob
from inspect import getsourcefile
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from config_helper.config import parse_config
from liso.datasets.torch_dataset_commons import (
    KITTI_IGNORE_NON_MOVABLE_CLASSMAPPING,
    KITTI_MAP_TO_SIMPLE_CLASSES,
    KITTI_MOVABLE_CLASSES,
    LidarDataset,
    add_lidar_rows_to_kitti_sample,
    lidar_dataset_collate_fn,
    recursive_npy_dict_to_torch,
    worker_init_fn,
)
from liso.kabsch.shape_utils import UNKNOWN_CLASS_ID, Shape

KITTI_OBJ_MOVABLE_CLASS_FREQS = {
    "Car": 14357,
    "Cyclist": 734,
    # "DontCare": 5399,
    # "Misc": 337,
    "Pedestrian": 2207,
    # "Person_sitting": 56,
    # "Tram": 224,
    # "Truck": 488,
    # "Van": 1297,
}


class KittiObjectDataset(LidarDataset):
    movable_class_names = KITTI_MOVABLE_CLASSES
    class_names = KITTI_MOVABLE_CLASSES
    class_name_to_idx_mapping = {k: i for i, k in enumerate(class_names)}
    idx_to_class_name_mapping = dict(enumerate(class_names))
    movable_class_frequencies = np.array(
        [KITTI_OBJ_MOVABLE_CLASS_FREQS[k] for k in movable_class_names]
    ) / sum(KITTI_OBJ_MOVABLE_CLASS_FREQS.values())

    def __init__(
        self,
        cfg,
        shuffle: bool,
        use_geom_augmentation: bool,
        use_skip_frames: str,
        path_to_augmentation_db: str = None,
        allow_data_augmentation=False,
        mode="val",
        size=None,
        verbose=False,
        pure_inference_mode=False,
        get_only_these_specific_samples=None,
    ) -> None:
        super().__init__(
            cfg,
            mode=mode,
            shuffle=shuffle,
            use_geom_augmentation=use_geom_augmentation,
            use_skip_frames=use_skip_frames,
            path_to_augmentation_db=path_to_augmentation_db,
            path_to_mined_boxes_db=None,
            for_tracking=False,
            need_flow=False,
        )
        assert mode in ("val", "train", "test"), mode
        self.verbose = verbose
        if mode == "test":
            assert pure_inference_mode, pure_inference_mode
            assert not use_geom_augmentation, use_geom_augmentation
        self.pure_inference_mode = pure_inference_mode
        dataset_root = Path(cfg.data.paths.kitti.local)

        dataset_root = dataset_root.joinpath("kitti_object_w_future_pcl", mode)
        sample_files = sorted(glob(str(Path(dataset_root).joinpath("*.npy"))))
        if len(sample_files) == 0:
            raise FileNotFoundError(f"No sample files found in {dataset_root}")
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
        if self.shuffle:
            np.random.shuffle(self.sample_files)

        self.allow_data_augmentation = allow_data_augmentation
        assert self.data_use_skip_frames in ("only", "both", "never")

    def object_is_movable(self, obj_category) -> bool:
        return KITTI_IGNORE_NON_MOVABLE_CLASSMAPPING[obj_category] == "movable"

    def __getitem__(self, index):
        self.initialize_loader_saver_if_necessary()
        self.initialize_dbs_if_necessary()
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

        add_lidar_rows_to_kitti_sample(sample_content, time_keys=(src_key, target_key))

        if "pcl_t1" not in sample_content:
            sample_content["pcl_t1"] = np.zeros(
                (0, 4), dtype=sample_content["pcl_t0"].dtype
            )
            sample_content["is_ground_t1"] = np.zeros(
                (0,), dtype=sample_content["is_ground_t0"].dtype
            )

        if not self.cfg.data.use_lidar_intensity:
            self.drop_intensities_from_pcls_in_sample(sample_content)

        self.drop_unused_timed_keys_from_sample(
            sample_content, src_key, target_key, delete_target_key
        )
        sample_content = self.drop_points_on_kitti_vehicle(
            sample_content, src_key, target_key
        )
        ignore_box_width = 2 * np.linalg.norm(self.bev_range_m_np / 2)
        kitti_ignore_region_boxes = Shape(
            pos=np.array(
                [
                    [-self.bev_range_m_np[0] / 2, self.bev_range_m_np[1] / 2, 0.0],
                    [
                        -self.bev_range_m_np[0] / 2,
                        -self.bev_range_m_np[1] / 2,
                        0.0,
                    ],
                ]
            ),
            # pos=np.array([[-0.0,0.0,0.0],
            #              [-0.0,-0.0,0.0,]]),
            # dims=np.array([[200.0,200.0,100.0],[200.0,200.0,100.0]]),
            dims=np.array(
                [[200.0, ignore_box_width, 100.0], [200, ignore_box_width, 100.0]]
            ),
            rot=np.array([[np.deg2rad(42)], [np.deg2rad(-42)]]),
            probs=np.array([[1.0], [1.0]]),
        )
        sample_content = self.move_keys_to_subdict(sample_content)
        sample_content["gt"]["kitti_ignore_region_boxes_t0"] = kitti_ignore_region_boxes
        sample_content["gt"][
            "kitti_ignore_region_boxes_t1"
        ] = kitti_ignore_region_boxes.clone()
        meta_info = {"sample_id": sample_content.pop("name")}
        meta_info["filename"] = str(Path(fname).stem)

        if (
            not self.for_tracking  # WE CANT AUGMENT WHEN TRACKING!
            and self.use_geom_augmentation
            and self.cfg.data.augmentation.active
            and self.mode == "train"
        ):
            self.augment_sample_content(
                sample_content,
                src_key,
                target_key,
                "kitti_object",
            )

        sample_data_ta = self.assemble_sample_data(
            sample_content,
            src_key,
            target_key,
            src_trgt_time_delta_s=src_trgt_time_delta_s,
        )
        sample_data_ta["gt"].pop("objects_ta", None)
        sample_data_ta = recursive_npy_dict_to_torch(sample_data_ta)
        if self.cfg.loss.supervised.centermaps.active and self.mode == "train":
            sample_data_ta["gt"].update(
                self.get_motion_based_centermaps(sample_data_ta)
            )
        if self.mode == "train":
            augm_sample_ta = self.create_augmented_sample_from_flow_cluster_detector_and_box_snippet_db(
                src_trgt_time_delta_s, sample_data_ta
            )
        else:
            augm_sample_ta = {}
        return (
            sample_data_ta,
            {},
            augm_sample_ta,
            meta_info,
        )

    def get_valid_fov_labels_mask(self, pcl):
        point_yaw_angle = np.arctan2(pcl[:, 1], pcl[:, 0])
        has_label_info_available = np.abs(point_yaw_angle) < np.deg2rad(41.0)
        return has_label_info_available

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

    def get_has_valid_scene_flow_label(self, sample_content, src_key):
        return self.get_valid_fov_labels_mask(sample_content[f"pcl_{src_key}"]["pcl"])

    def extract_boxes_for_timestamp(
        self, sample_content: Dict[str, np.ndarray], src_key: str, target_key: str
    ) -> Shape:
        return self.kitti_extract_boxes_for_timestamp(sample_content, src_key)


def main():
    default_cfg_file = (
        Path(getsourcefile(lambda: 0)).parent.parent
        / Path("config")
        / Path("liso_config.yml")
    )

    cfg = parse_config(default_cfg_file, extra_cfg_args=("kitti",))
    train_ds = KittiObjectDataset(cfg, use_geom_augmentation=True)
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
