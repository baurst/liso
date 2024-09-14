from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.structures.sweep import Sweep
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
from liso.datasets.argoverse2.av2_classes import AV2_MOVABLE_CLASSES
from liso.jcp.jcp import JPCGroundRemove
from liso.kabsch.box_groundtruth_matching import (
    slow_greedy_match_boxes_by_desending_confidence_by_dist,
)
from liso.kabsch.shape_utils import Shape
from liso.tracker.tracking_helpers import accumulate_pcl
from liso.transformations.transformations import decompose_matrix
from liso.utils.torch_transformation import homogenize_pcl
from liso.visu.pcl_image import create_topdown_f32_pcl_image_variable_extent
from PIL import Image
from tqdm import tqdm


def get_object_velocity_in_obj_coords(
    odom_ta_tb: np.ndarray, obj_pose_ta: np.ndarray, obj_pose_tb: np.ndarray
) -> np.ndarray:
    dyn_flow_trafo = obj_pose_tb @ np.linalg.inv(obj_pose_ta) - np.eye(4)
    stat_flow_trafo = np.linalg.inv(odom_ta_tb) - np.eye(4)
    obj_pos_ta = obj_pose_ta[:, 0:2, 3]

    obj_norig_flow_trafo = dyn_flow_trafo - stat_flow_trafo
    obj_pos_homog_ta = np.concatenate(
        [
            obj_pos_ta,
            np.zeros_like(obj_pos_ta[..., [0]]),
            np.ones_like(obj_pos_ta[..., [0]]),
        ],
        axis=-1,
    )
    norig_flow_sensor_cosy = np.einsum(
        "nij,nj->ni", obj_norig_flow_trafo, obj_pos_homog_ta
    )[..., :3]

    norig_flow_sensor_cosy_homog = np.concatenate(
        [
            norig_flow_sensor_cosy,
            np.zeros_like(norig_flow_sensor_cosy[..., [0]]),
        ],
        axis=-1,
    )
    norig_flow_obj_cosy = np.einsum(
        "nij,nj->ni", obj_pose_ta, norig_flow_sensor_cosy_homog
    )[..., :3]

    return norig_flow_obj_cosy


def create_odom_registered_bev_pcl_img(point_clouds_sensor, odoms_t0_t1, img_grid_size):
    pcl_accum = accumulate_pcl(point_clouds_sensor, odoms_t0_t1)
    min_pcl_extent = pcl_accum.min(dim=0)[0][:2]  # only need x and y
    max_pcl_extent = pcl_accum.max(dim=0)[0][:2]  # only need x and y
    assert pcl_accum.shape[-1] == 4, pcl_accum.shape  # need intensity attribute
    intensity_img, pixel_occup_mask = create_topdown_f32_pcl_image_variable_extent(
        pcl_accum,
        pcl_accum[:, -1],
        min_pcl_extent,
        max_pcl_extent,
        img_grid_size,
    )

    intensity_values = intensity_img[pixel_occup_mask].cpu().numpy()
    blank_canvas = np.zeros((img_grid_size[0], img_grid_size[1], 3), dtype=np.float32)

    intensity_color_f32 = (
        plt.cm.jet(
            intensity_values,
        )[..., :3]
    ).astype(np.float32)
    blank_canvas[pixel_occup_mask.cpu().numpy()] = intensity_color_f32
    return blank_canvas, pixel_occup_mask


def main():
    verbose = False

    parser = ArgumentParser()
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument(
        "--av2_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--target_dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    WORLD_SIZE = args.world_size
    WORKER_ID = args.worker_id

    data_src_root_dir: Path = args.av2_root
    data_target_dir: Path = args.target_dir

    for split_name in ("train", "val"):
        data_loader = AV2SensorDataLoader(
            data_dir=data_src_root_dir / split_name,
            labels_dir=data_src_root_dir / split_name,
        )

        log_ids = data_loader.get_log_ids()
        if WORLD_SIZE > 1:
            log_ids = np.array_split(log_ids, WORLD_SIZE)[WORKER_ID]

        kiss_config = KISSConfig()
        kiss_config.mapping.voxel_size = 0.01 * kiss_config.data.max_range
        for seq_id in log_ids:
            print(f"Processing sequence {seq_id}")

            timestamps_in_seq = data_loader.get_ordered_log_lidar_timestamps(seq_id)
            # timestamps_in_seq = timestamps_in_seq[:5]
            world_Ts_lidar = []
            kiss_odom = KissICP(kiss_config)
            vis_pcls = []
            for timestamp_ns in tqdm(timestamps_in_seq, disable=False):
                lidar_fpath = data_loader.get_lidar_fpath(seq_id, timestamp_ns)

                sweep = Sweep.from_feather(lidar_fpath)
                vehicle_T_lidar = sweep.ego_SE3_up_lidar.transform_matrix
                # vehicle_Tdown_lidar = sweep.ego_SE3_down_lidar.transform_matrix
                world_T_vehicle = data_loader.get_city_SE3_ego(
                    seq_id, timestamp_ns
                ).transform_matrix
                world_Ts_lidar.append(world_T_vehicle @ vehicle_T_lidar)

                lidar_T_vehicle = np.linalg.inv(vehicle_T_lidar)

                homog_pcl_lidar = np.einsum(
                    "ij,nj->ni", lidar_T_vehicle, homogenize_pcl(sweep.xyz[:, :3])
                )
                vis_pcls.append(
                    torch.from_numpy(
                        np.concatenate(
                            [
                                homog_pcl_lidar[:, :3],
                                sweep.intensity[..., None] / 255.0,
                            ],
                            axis=-1,
                        )
                    ).detach()
                )
                per_point_timestamps_normalized = (
                    sweep.offset_ns - sweep.offset_ns.min()
                ) / (np.ptp(sweep.offset_ns))
                kiss_odom.register_frame(
                    homog_pcl_lidar[:, :3],
                    per_point_timestamps_normalized,
                )

            world_Tkiss_lidar = kiss_odom.poses

            kiss_odoms = []
            gt_odoms = []
            for time_idx in range(0, len(world_Ts_lidar) - 1, 1):
                kiss_odoms_t0_t1 = (
                    np.linalg.inv(world_Tkiss_lidar[time_idx])
                    @ world_Tkiss_lidar[time_idx + 1]
                )
                kiss_odoms.append(kiss_odoms_t0_t1)
                gt_odoms.append(
                    np.linalg.inv(world_Ts_lidar[time_idx])
                    @ world_Ts_lidar[time_idx + 1]
                )

            intensity_img_f32, _ = create_odom_registered_bev_pcl_img(
                vis_pcls,
                [torch.from_numpy(odom).detach() for odom in kiss_odoms],
                torch.tensor((2048, 2048)),
            )
            kiss_icp_bev_img_target = (
                data_target_dir / "kiss_icp_bevs" / f"{seq_id}.png"
            )
            kiss_icp_bev_img_target.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray((intensity_img_f32 * 255).astype(np.uint8)).save(
                kiss_icp_bev_img_target
            )

            del vis_pcls

            per_timestamp_box_labels = {}
            for timestamp_ns in timestamps_in_seq:
                box_labels_t0 = data_loader.get_labels_at_lidar_timestamp(
                    seq_id, timestamp_ns
                )
                per_timestamp_box_labels[timestamp_ns] = box_labels_t0

            for time_idx in tqdm(range(len(timestamps_in_seq) - 1), disable=False):
                timestamp_t0 = timestamps_in_seq[time_idx]
                timestamp_t1 = timestamps_in_seq[time_idx + 1]

                lidar_fpath_t0 = data_loader.get_lidar_fpath(seq_id, timestamp_t0)
                file_id = "/".join(lidar_fpath_t0.as_posix().split("/")[-6:])
                target_fname = (data_target_dir / file_id).with_suffix(".npz")
                target_fname.parent.mkdir(exist_ok=True, parents=True)
                sweep_t0, is_ground_t0 = get_sweep_and_ground_label(lidar_fpath_t0)
                odom_t0_t1 = gt_odoms[time_idx]
                odom_t1_t0 = np.linalg.inv(odom_t0_t1)
                kiss_odom_t0_t1 = kiss_odoms[time_idx]
                homog_pcl_lidar_t0 = np.einsum(
                    "ij,nj->ni", lidar_T_vehicle, homogenize_pcl(sweep_t0.xyz[:, :3])
                )
                flow_t0_t1 = np.einsum(
                    "ij,nj->ni", (odom_t1_t0 - np.eye(4)), homog_pcl_lidar_t0
                )[:, :3]

                timestamp_t1 = timestamps_in_seq[time_idx + 1]
                lidar_fpath_t1 = data_loader.get_lidar_fpath(seq_id, timestamp_t1)
                sweep_t1, is_ground_t1 = get_sweep_and_ground_label(lidar_fpath_t1)
                homog_pcl_lidar_t1 = np.einsum(
                    "ij,nj->ni", lidar_T_vehicle, homogenize_pcl(sweep_t1.xyz[:, :3])
                )
                flow_t1_t0 = np.einsum(
                    "ij,nj->ni", (odom_t0_t1 - np.eye(4)), homog_pcl_lidar_t1
                )[:, :3]

                obj_categories_t0, boxes_t0 = get_boxes_for_timestamp(
                    timestamp_t0, per_timestamp_box_labels, lidar_T_vehicle
                )
                obj_categories_t1, boxes_t1 = get_boxes_for_timestamp(
                    timestamp_t1, per_timestamp_box_labels, lidar_T_vehicle
                )

                # boxes_t0 = boxes_t0.to_tensor()
                all_box_idxs_t0 = np.arange(boxes_t0.shape[0])

                # boxes_t1 = boxes_t1.to_tensor()
                all_box_idxs_t1 = np.arange(boxes_t1.shape[0])

                uniq_categories = np.unique(obj_categories_t0)
                for obj_category in uniq_categories:
                    if obj_category in AV2_MOVABLE_CLASSES:
                        is_cat_t0 = obj_categories_t0 == np.array(obj_category)
                        boxes_of_cat_t0 = boxes_t0.clone()
                        boxes_of_cat_t0.valid = is_cat_t0
                        boxes_of_cat_t0 = boxes_of_cat_t0.drop_padding_boxes()

                        is_cat_t1 = obj_categories_t1 == np.array(obj_category)
                        boxes_of_cat_t1 = boxes_t1.clone()
                        boxes_of_cat_t1.valid = is_cat_t1
                        boxes_of_cat_t1 = boxes_of_cat_t1.drop_padding_boxes()

                        (
                            idxs_into_t0,
                            idxs_into_t1,
                            _,
                            _matched_t1_mask,
                            _matched_t0_mask,
                        ) = slow_greedy_match_boxes_by_desending_confidence_by_dist(
                            boxes_of_cat_t0.pos,
                            boxes_of_cat_t1.pos,
                            non_batched_pred_confidence=np.squeeze(
                                boxes_of_cat_t1.probs, axis=-1
                            ),
                            matching_threshold=1.0,  # in meters
                            match_in_nd=2,
                        )

                        matched_boxes_of_cat_t0 = boxes_of_cat_t0[idxs_into_t0].numpy()
                        matched_boxes_of_cat_t1 = boxes_of_cat_t1[idxs_into_t1].numpy()

                        flow_t0_t1 = update_dynamic_flow(
                            homog_pcl_lidar_t0,
                            flow_t0_t1,
                            matched_boxes_of_cat_t0,
                            matched_boxes_of_cat_t1,
                        )
                        flow_t1_t0 = update_dynamic_flow(
                            homog_pcl_lidar_t1,
                            flow_t1_t0,
                            matched_boxes_of_cat_t1,
                            matched_boxes_of_cat_t0,
                        )

                        matched_obj_velos_t0 = np.linalg.norm(
                            get_object_velocity_in_obj_coords(
                                odom_t0_t1,
                                matched_boxes_of_cat_t0.get_poses(),
                                matched_boxes_of_cat_t1.get_poses(),
                            ),
                            axis=-1,
                            keepdims=True,
                        )

                        boxes_t0.velo[
                            all_box_idxs_t0[is_cat_t0][idxs_into_t0]
                        ] = matched_obj_velos_t0

                        matched_obj_velos_t1 = np.linalg.norm(
                            get_object_velocity_in_obj_coords(
                                odom_t1_t0,
                                matched_boxes_of_cat_t1.get_poses(),
                                matched_boxes_of_cat_t0.get_poses(),
                            ),
                            axis=-1,
                            keepdims=True,
                        )

                        boxes_t1.velo[
                            all_box_idxs_t1[is_cat_t1][idxs_into_t1]
                        ] = matched_obj_velos_t1

                data_dict = {
                    "meta_data_t0": file_id,
                    "pcl_t0": np.concatenate(
                        [
                            homog_pcl_lidar_t0[:, :3],
                            sweep_t0.intensity[..., None] / 255.0,
                        ],
                        axis=-1,
                    ).astype(np.float32),
                    "pcl_t1": np.concatenate(
                        [
                            homog_pcl_lidar_t1[:, :3],
                            sweep_t1.intensity[..., None] / 255.0,
                        ],
                        axis=-1,
                    ).astype(np.float32),
                    "gt": {
                        "boxes_t0": boxes_t0.__dict__,
                        "boxes_t1": boxes_t1.__dict__,
                        "flow_t0_t1": flow_t0_t1[:, 0:3].astype(np.float32),
                        "flow_t1_t0": flow_t1_t0[:, 0:3].astype(np.float32),
                        "odom_t0_t1": odom_t0_t1.astype(np.float64),
                        "odom_t1_t0": odom_t1_t0.astype(np.float64),
                        "is_ground_t0": is_ground_t0.astype(bool),
                        "is_ground_t1": is_ground_t1.astype(bool),
                        "box_category_t0": np.array(obj_categories_t0),
                        "box_category_t1": np.array(obj_categories_t1),
                    },
                    "kiss_icp": {
                        "odom_t0_t1": kiss_odom_t0_t1.astype(np.float64),
                        "odom_t1_t0": np.linalg.inv(kiss_odom_t0_t1).astype(np.float64),
                    },
                    "lidar_rows_t0": sweep_t0.laser_number.astype(np.uint8),
                    "lidar_rows_t1": sweep_t1.laser_number.astype(np.uint8),
                }
                np.savez_compressed(target_fname, data_dict)

                if verbose:
                    print(f"Saved {target_fname}")


def update_dynamic_flow(
    homog_pcl_ta, flow_ta_tb, matched_boxes_of_cat_ta, matched_boxes_of_cat_tb
):
    sensort0_T_box = matched_boxes_of_cat_ta.get_poses()
    sensort1_T_box = matched_boxes_of_cat_tb.get_poses()
    cur_dyn_flow_trafo_t0_t1 = sensort1_T_box @ np.linalg.inv(sensort0_T_box)

    point_is_in_box_mask_t0 = matched_boxes_of_cat_ta.get_points_in_box_bool_mask(
        homog_pcl_ta[:, :3]
    )
    point_is_in_more_than_one_box = point_is_in_box_mask_t0.sum(axis=-1) > 1
    if point_is_in_more_than_one_box.any():
        point_is_in_box_mask_t0[point_is_in_more_than_one_box] = (
            point_is_in_box_mask_t0[point_is_in_more_than_one_box]
            .cumsum(axis=-1)
            .cumsum(axis=-1)
            == 1
        )
    cur_dyn_flow_t0_t1 = np.einsum(
        "kij,nj->nki",
        cur_dyn_flow_trafo_t0_t1 - np.eye(4),
        homog_pcl_ta,
    )[..., :3]
    flow_ta_tb[point_is_in_box_mask_t0.any(axis=-1)] = cur_dyn_flow_t0_t1[
        point_is_in_box_mask_t0
    ]
    return flow_ta_tb


def get_boxes_for_timestamp(timestamp_ns, per_timestamp_box_labels, lidar_T_vehicle):
    box_labels_t0 = per_timestamp_box_labels[timestamp_ns]
    box_shapes = []
    obj_categories = []
    for cuboid in box_labels_t0.cuboids:
        obj_categories.append(cuboid.category)
        vehicle_T_box = np.eye(4)
        vehicle_T_box[:3, :3] = cuboid.dst_SE3_object.rotation
        vehicle_T_box[:3, 3] = cuboid.dst_SE3_object.translation
        lidar_T_box = lidar_T_vehicle @ vehicle_T_box
        _, _, rot_angles, pos_m, _ = decompose_matrix(lidar_T_box)

        theta = np.array(rot_angles[-1])[..., None]
        shape = Shape(
            pos=pos_m,
            dims=cuboid.dims_lwh_m,
            rot=theta,
            probs=np.ones_like(theta),
        )
        box_shapes.append(shape)
    boxes_t0 = Shape.from_list_of_npy_shapes(
        box_shapes,
    )

    return obj_categories, boxes_t0


@lru_cache(maxsize=3)
def get_sweep_and_ground_label(lidar_fpath_t0):
    sweep_t0 = Sweep.from_feather(lidar_fpath_t0)

    is_ground_t0 = JPCGroundRemove(
        pcl=sweep_t0.xyz[:, :3],
        range_img_width=2000,
        range_img_height=64,
        sensor_height=1.8,
        delta_R=2,
    )
    return sweep_t0, is_ground_t0


if __name__ == "__main__":
    main()
