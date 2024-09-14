from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from liso.datasets.torch_dataset_commons import lidar_dataset_collate_fn, worker_init_fn
from liso.kabsch.shape_utils import Shape
from liso.tracker.augm_box_db_utils import (
    get_empty_augm_box_db,
    save_augmentation_database,
)
from liso.tracker.tracking import get_clean_train_dataset_single_batch
from liso.utils.config_helper_helper import load_handle_args_cfg_logdir
from tqdm import tqdm


@torch.no_grad()
def main():
    args, cfg, log_dir = load_handle_args_cfg_logdir()

    target_dir_box_augm_db = (
        # f"/mnt/LISO_DATA_DIR/selfsupervised_OD/real_gt_augm_dbs/{cfg.data.source}"
        "/tmp/db"
    )

    assert cfg.data.bev_range_m[0] > 80.0, "too limited!"
    assert cfg.data.source in ["waymo", "nuscenes"], cfg.data.source

    print(
        build_augmentation_db_from_actual_groundtruth(
            cfg,
            target_dir_box_augm_db,
        )
    )


def build_augmentation_db_from_actual_groundtruth(
    cfg,
    target_dir_box_augm_db,
    save_every_n_samples=100,
    min_num_points_in_box=5,
    max_size_of_db_mb=100,
):
    start_time = datetime.now()
    copy_cfg = deepcopy(cfg)
    copy_cfg.data.waymo_downsample_factor = 1
    copy_cfg.data.augmentation.active = False
    dataset = get_clean_train_dataset_single_batch(
        copy_cfg,
        need_flow=False,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        pin_memory=True,
        batch_size=1,
        shuffle=True,
        num_workers=copy_cfg.data.num_workers,
        collate_fn=lidar_dataset_collate_fn,
        worker_init_fn=worker_init_fn,
    )

    db = get_empty_augm_box_db()

    for sample_count, train_el in enumerate(tqdm(train_loader, disable=False)):
        sample_data_t0, _, _, meta = train_el

        all_gt_boxes: Shape = sample_data_t0["gt"]["boxes"]

        pcl = sample_data_t0["pcl_ta"]["pcl"]
        lidar_row_idxs = sample_data_t0["lidar_rows_ta"][0].detach().cpu()

        (
            all_point_is_in_box_mask,
            all_pts_in_box_homog,
        ) = all_gt_boxes.get_points_in_box_bool_mask(
            pcl[..., :3], return_points_in_box_coords=True
        )
        enough_points_in_boxes_mask = (
            all_point_is_in_box_mask.sum(dim=1) >= min_num_points_in_box
        )

        if enough_points_in_boxes_mask.sum() == 0:
            # no boxes with enough points found
            pass
        else:
            all_gt_boxes.valid = enough_points_in_boxes_mask & all_gt_boxes.valid
            assert len(all_gt_boxes.valid.shape) == 2, "need batched data here:"
            pcl = pcl[0]
            assert len(pcl.shape) == 2, "need batched data here:"

            all_sensor_T_box = all_gt_boxes.get_poses()[0]
            all_gt_boxes = all_gt_boxes[0].drop_padding_boxes()
            enough_points_in_boxes_mask = enough_points_in_boxes_mask[0]
            all_sensor_T_box = all_sensor_T_box[enough_points_in_boxes_mask]
            all_box_T_sensor = torch.linalg.inv(all_sensor_T_box)
            all_point_is_in_box_mask = all_point_is_in_box_mask[0][
                :, enough_points_in_boxes_mask
            ]
            all_pts_in_box_homog = all_pts_in_box_homog[
                0, :, enough_points_in_boxes_mask
            ]

            for box_idx in range(all_gt_boxes.shape[0]):
                this_box = all_gt_boxes[box_idx]
                point_is_in_this_box_mask = all_point_is_in_box_mask[:, box_idx]
                rows_box = (
                    lidar_row_idxs[point_is_in_this_box_mask].detach().cpu().numpy()
                )
                pts_in_this_box_homog = all_pts_in_box_homog[:, box_idx][
                    point_is_in_this_box_mask
                ]
                assert (
                    0.6 * this_box.dims[None, ...]
                    >= np.abs(pts_in_this_box_homog[:, :3])
                ).all()
                this_box_T_sensor = all_box_T_sensor[box_idx]

                intensity = (
                    pcl[point_is_in_this_box_mask, 3].cpu().numpy().astype(np.float32)
                )
                rows_box = lidar_row_idxs[point_is_in_this_box_mask].cpu().numpy()

                assert (
                    rows_box.shape[0]
                    == intensity.shape[0]
                    == pts_in_this_box_homog.shape[0]
                )
                db["pcl_in_box_cosy"].append(
                    np.concatenate(
                        [pts_in_this_box_homog[:, :3], intensity[:, None]], axis=-1
                    )
                )
                db["boxes"].append(this_box)
                db["box_T_sensor"].append(this_box_T_sensor.numpy())
                db["lidar_rows"].append(rows_box)
                db["unique_track_id"].append(0)

        if sample_count % save_every_n_samples == 0 and sample_count > 0:
            save_path, size_mb = save_augmentation_database(
                db,
                Path(target_dir_box_augm_db),
                global_step=0,
            )
            if size_mb > max_size_of_db_mb:
                print("db getting too large, stopping")
                break
    if sample_count < save_every_n_samples:
        save_path, _ = save_augmentation_database(
            db,
            Path(target_dir_box_augm_db),
            global_step=0,
        )
    end_time = datetime.now()
    print(f"Creating Box Augm DB took {end_time-start_time}s")
    return save_path


if __name__ == "__main__":
    main()
