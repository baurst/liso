import os
from copy import deepcopy
from datetime import datetime
from operator import itemgetter
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from liso.kabsch.shape_utils import Shape


def load_sanitize_box_augmentation_database(
    path_to_augmentation_db: Union[str, Path], confidence_threshold_mined_boxes: float
):
    print(f"Loadeding augmentation boxes from db at {path_to_augmentation_db}")
    box_augm_db = np.load(path_to_augmentation_db, allow_pickle=True).item()
    num_pts_per_box = np.array([el.shape[0] for el in box_augm_db["pcl_in_box_cosy"]])
    min_num_pts_per_box = 10
    box_is_confident_enough = torch.from_numpy(
        np.squeeze(box_augm_db["boxes"]["probs"], axis=-1)
        >= confidence_threshold_mined_boxes
    )
    print(
        f"Dropping {torch.count_nonzero(~box_is_confident_enough)}/{box_is_confident_enough.size()} "
        f"boxes from augmentation db - they are not more confident than "
        f"{confidence_threshold_mined_boxes}!"
    )
    enough_points_in_box = torch.from_numpy(num_pts_per_box > min_num_pts_per_box)
    keep_this_box = enough_points_in_box & box_is_confident_enough
    box_augm_db["pcl_in_box_cosy"] = [
        box_augm_db["pcl_in_box_cosy"][idx]
        for idx in range(len(box_augm_db["pcl_in_box_cosy"]))
        if keep_this_box[idx]
    ]
    box_augm_db["lidar_rows"] = [
        box_augm_db["lidar_rows"][idx]
        for idx in range(len(box_augm_db["lidar_rows"]))
        if keep_this_box[idx]
    ]
    box_augm_db["boxes"] = Shape(**box_augm_db["boxes"]).to_tensor()[keep_this_box]
    box_augm_db["box_T_sensor"] = torch.from_numpy(box_augm_db["box_T_sensor"])[
        keep_this_box
    ]
    assert box_augm_db["box_T_sensor"].shape[0] == box_augm_db["boxes"].shape[0], (
        box_augm_db["box_T_sensor"].shape,
        box_augm_db["boxes"].shape,
    )

    assert len(box_augm_db["pcl_in_box_cosy"]) == box_augm_db["boxes"].shape[0], (
        len(box_augm_db["pcl_in_box_cosy"]),
        box_augm_db["boxes"].shape,
    )

    num_loaded_augm_boxes = sum(box_augm_db["boxes"].shape)
    print(
        f"Loaded {num_loaded_augm_boxes} augmentation boxes from db at {path_to_augmentation_db}"
    )
    return box_augm_db


def get_empty_augm_box_db():
    return {
        "pcl_in_box_cosy": [],
        "lidar_rows": [],
        "boxes": [],
        "box_T_sensor": [],
        "unique_track_id": [],
    }


def estimate_augm_db_size_mb(db):
    sum_bytes = sum([v.nbytes for v in db["pcl_in_box_cosy"]])
    total_megabytes = sum_bytes * 1e-6
    return total_megabytes


def drop_boxes_from_augmentation_db(db: Dict[str, List], max_size_mb: int):
    before_db_size_mb = estimate_augm_db_size_mb(db)
    if before_db_size_mb <= max_size_mb:
        return db

    all_box_confidences = np.squeeze(
        np.stack([box.probs for box in db["boxes"]]), axis=-1
    )
    oversize_ratio = before_db_size_mb / max_size_mb
    num_boxes_to_keep = int(len(db["boxes"]) / oversize_ratio)
    if len(np.unique(all_box_confidences) == 1):
        # all samples have the same confidence: random dropping
        keep_idxs = np.random.choice(
            np.arange(0, len(db["boxes"])), num_boxes_to_keep, replace=False
        )

    else:
        min_confidence = all_box_confidences.min()
        delta_confidence = 0.001
        keep_mask = np.ones_like(all_box_confidences, dtype=bool)
        while keep_mask.sum() > num_boxes_to_keep:
            new_min_confidence = min_confidence + delta_confidence
            keep_mask[all_box_confidences < min_confidence + delta_confidence] = False
            min_confidence = new_min_confidence

        keep_idxs = np.arange(0, len(db["boxes"]))[keep_mask]
    downsized_db = {k: list(itemgetter(*keep_idxs)(v)) for k, v in db.items()}
    after_db_size_mb = estimate_augm_db_size_mb(downsized_db)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(
        f"{time_str}: Dropped from {before_db_size_mb}Mb to {after_db_size_mb}Mb ({before_db_size_mb - after_db_size_mb})Mb from db!"
    )
    return downsized_db


def save_augmentation_database(
    db, export_raw_tracked_detections_to: Path, global_step: int
):
    save_db = {k: deepcopy(v) for k, v in db.items()}
    export_raw_tracked_detections_to = Path(export_raw_tracked_detections_to)
    export_raw_tracked_detections_to.mkdir(exist_ok=True, parents=True)
    if len(save_db["box_T_sensor"]) == 0:
        # THIS IS DUMMY DATA!
        print("WARNING: THIS IS DUMMY DATA!")
        save_db["unique_track_id"] = np.array([0], dtype=np.uint32)
        save_db["box_T_sensor"] = np.eye(4, dtype=np.float64)[None]
        save_db["boxes"] = Shape(
            pos=np.array([10.0, 0.0, 0.0]),
            dims=np.array([10.0, 5.0, 1.0]),
            rot=np.array(
                [0.0],
            ),
            probs=np.array(
                [1.0],
            ),
            velo=np.array(
                [1.0],
            ),
        )[None].__dict__
        dummy_pcl = np.array(
            [
                # need at least 10 points in the dummy box, or it will be filtered in
                # load_sanitize_box_augmentation_database
                [2.0, 3.0, 1.0, 1.0],
                [3.0, 3.0, 1.0, 1.0],
                [4.0, -3.0, 1.0, 1.0],
                [5.0, 1.0, 1.0, 1.0],
                [2.0, 3.0, 1.0, 1.0],
                [2.0, 3.0, 1.0, 1.0],
                [2.0, 3.0, 1.0, 1.0],
                [2.0, 3.0, 1.0, 1.0],
                [2.0, 3.0, 1.0, 1.0],
                [2.0, 3.0, 1.0, 1.0],
                [2.0, 3.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        save_db["lidar_rows"] = [
            np.arange(
                dummy_pcl.shape[0],
                dtype=np.uint8,
            )
        ]

        save_db["pcl_in_box_cosy"] = [
            dummy_pcl,
        ]
        print("WARNING: Not a single object was mined!")
    else:
        save_db["unique_track_id"] = np.stack(
            save_db["unique_track_id"], axis=0
        ).astype(np.uint32)
        save_db["box_T_sensor"] = np.stack(save_db["box_T_sensor"], axis=0)
        save_db["boxes"] = (
            Shape.from_list_of_shapes(save_db["boxes"]).cpu().numpy().__dict__
        )
    save_name = (
        Path(export_raw_tracked_detections_to)
        / f"boxes_db_global_step_{global_step}.npy"
    )
    np.save(save_name, save_db)
    size_in_mb = os.path.getsize(save_name) >> 20
    print(
        f"Saving {len(save_db['pcl_in_box_cosy'])} boxes ({size_in_mb} Mb)with point clouds to {save_name}"
    )
    return save_name, size_in_mb
