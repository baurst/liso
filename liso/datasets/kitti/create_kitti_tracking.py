#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pykitti
from liso.datasets.kitti.create_kitti_raw import (
    correct_kitti_scan,
    load_kitti_pcl_image_projection_get_ground_label,
)
from tqdm import tqdm


def get_points_in_box_mask(pcl_homog, lidar_T_obj, obj_size):
    pcl_obj = np.einsum("ij,nj->ni", np.linalg.inv(lidar_T_obj), pcl_homog)
    point_is_in_obj_mask = (np.abs(pcl_obj[:, 0:3]) < obj_size / 2.0).all(axis=-1)
    return point_is_in_obj_mask


def get_kitti_tracking_object_cam_pose(obj):
    T_cam2_obj = np.eye(4, dtype=np.float64)
    T_cam2_obj[0:3, 3] = [
        obj.location.x,
        obj.location.y - obj.dimensions.height / 2.0,
        obj.location.z,
    ]
    rot_y = obj.rotation_y
    cos_rot_y = np.cos(rot_y)
    sin_rot_y = np.sin(rot_y)
    rot_mat_y = np.eye(3, dtype=np.float64)
    rot_mat_y[0, 0] = cos_rot_y
    rot_mat_y[2, 2] = cos_rot_y
    rot_mat_y[2, 0] = -sin_rot_y
    rot_mat_y[0, 2] = sin_rot_y
    T_cam2_obj[0:3, 0:3] = rot_mat_y
    return T_cam2_obj


def get_corrected_lidar_frame_idx_for_seq(tracking_seq: str, raw_frame_idx: int):
    if tracking_seq == "0001":
        # this sequence had some lidar framedrops -> missing velo files...
        if raw_frame_idx in [177, 178, 179, 180]:
            return None
        elif raw_frame_idx > 180:
            frame_idx = raw_frame_idx - 4
        else:
            frame_idx = raw_frame_idx
    else:
        frame_idx = raw_frame_idx

    return frame_idx


def assemble_object_dict(velo_T_obj, obj_dims, track_ids, obj_class):
    return {
        "poses": velo_T_obj,
        "size": obj_dims,
        "track_ids": track_ids,
        "category": obj_class,
    }


def main():
    from kiss_icp.config import KISSConfig
    from kiss_icp.datasets.kitti_raw import KITTIRawDataset
    from kiss_icp.kiss_icp import KissICP

    argparser = ArgumentParser(
        description="Convert kitti tracking data to training format."
    )
    argparser.add_argument(
        "--target_dir",
        default=Path("/mnt/LISO_DATA_DIR/selfsupervised_OD/kitti"),
        type=Path,
    )
    argparser.add_argument(
        "--kitti_tracking_root",
        default=Path(
            "/mnt/LISO_DATA_DIR/datasets/datasets_raw/kitti/kitti_tracking/training"
        ),
        type=Path,
    )
    args = argparser.parse_args()

    target_base_dir = args.target_dir
    target_dir = target_base_dir.joinpath("kitti_tracking")
    kiss_icp_db_file_wo_ext = target_base_dir.joinpath("kitti_tracking_kiss_icp_poses")

    target_dir.mkdir(parents=True, exist_ok=True)

    tracking_seqs = [str(el).zfill(4) for el in range(21)]

    # transform from imu to velo is consistent for all sequences
    Tr_imu_to_velo_kitti = (
        "9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 "
        "-7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 "
        "2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01 "
        "0.0 0.0 0.0 1.0"
    )

    velo_T_imu = np.fromstring(Tr_imu_to_velo_kitti, dtype="float64", sep=" ").reshape(
        (4, 4)
    )
    imu_T_velo = np.linalg.inv(velo_T_imu)

    all_poses_w_T_lidar = {}
    try:
        all_poses_w_T_lidar = np.load(
            kiss_icp_db_file_wo_ext.with_suffix(".npy"),
            allow_pickle=True,
        ).item()
    except FileNotFoundError as e:
        print(e)
        print("Recomputing full KISS odometry from scratch")

    # compute odometry first
    kiss_config = KISSConfig()
    kiss_config.mapping.voxel_size = 0.01 * kiss_config.data.max_range
    for tracking_seq_str in tqdm(tracking_seqs):
        if tracking_seq_str in all_poses_w_T_lidar:
            # skip it, use precomputed result
            continue
        kitti_tracking = pykitti.tracking(
            args.kitti_tracking_root, tracking_seq_str, ignore_dontcare=True
        )
        kitti_tracking._load_oxts()

        cam2_T_velo = kitti_tracking.calib.T_cam2_velo
        velo_T_cam2 = np.linalg.inv(cam2_T_velo)
        odometry = KissICP(config=kiss_config)
        pcl_fnames = []
        for obj_frame_idx in tqdm(range(len(kitti_tracking.objects)), leave=False):
            corrected_lidar_frame_idx = get_corrected_lidar_frame_idx_for_seq(
                tracking_seq_str, obj_frame_idx
            )
            if corrected_lidar_frame_idx is None:
                continue

            (
                pcl_t0,
                pcl_homog_t0,
                is_ground_t0,
                lidar_Tt0_obj,
                obj_dims_t0,
                track_ids_t0,
                obj_class_t0,
                _,
                lidar_fname,
            ) = get_data_for_index(
                kitti_tracking, velo_T_cam2, obj_frame_idx, corrected_lidar_frame_idx
            )
            pcl_fnames.append(lidar_fname)
            odometry.register_frame(
                correct_kitti_scan(pcl_t0[:, :3].astype(np.float64)),
                KITTIRawDataset.get_timestamps(pcl_t0),
            )
        all_poses_w_T_lidar[tracking_seq_str] = dict(zip(pcl_fnames, odometry.poses))
        np.save(kiss_icp_db_file_wo_ext, all_poses_w_T_lidar, allow_pickle=True)

    for tracking_seq_str in tqdm(tracking_seqs):
        kitti_tracking = pykitti.tracking(
            args.kitti_tracking_root, tracking_seq_str, ignore_dontcare=True
        )
        kitti_tracking._load_oxts()

        cam2_T_velo = kitti_tracking.calib.T_cam2_velo
        velo_T_cam2 = np.linalg.inv(cam2_T_velo)
        if tracking_seq_str == "0000":
            np.save(
                "/tmp/cam2_T_velo.npy", cam2_T_velo
            )  # to convert OGC to camera coordnate system
        for obj_frame_idx in tqdm(range(len(kitti_tracking.objects) - 2)):
            corrected_lidar_frame_idx = get_corrected_lidar_frame_idx_for_seq(
                tracking_seq_str, obj_frame_idx
            )
            if corrected_lidar_frame_idx is None:
                continue

            (
                pcl_t0,
                pcl_homog_t0,
                is_ground_t0,
                lidar_Tt0_obj,
                obj_dims_t0,
                track_ids_t0,
                obj_class_t0,
                _,
                lidar_fname_t0,
            ) = get_data_for_index(
                kitti_tracking, velo_T_cam2, obj_frame_idx, corrected_lidar_frame_idx
            )

            (
                pcl_t1,
                pcl_homog_t1,
                is_ground_t1,
                lidar_Tt1_obj,
                obj_dims_t1,
                track_ids_t1,
                obj_class_t1,
                _,
                lidar_fname_t1,
            ) = get_data_for_index(
                kitti_tracking,
                velo_T_cam2,
                obj_frame_idx + 1,
                corrected_lidar_frame_idx + 1,
            )

            (
                pcl_t2,
                pcl_homog_t2,
                is_ground_t2,
                lidar_Tt2_obj,
                obj_dims_t2,
                track_ids_t2,
                obj_class_t2,
                _,
                lidar_fname_t2,
            ) = get_data_for_index(
                kitti_tracking,
                velo_T_cam2,
                obj_frame_idx + 1,
                corrected_lidar_frame_idx + 1,
            )

            w_T_imu_t0 = kitti_tracking.oxts[obj_frame_idx].T_w_imu.astype(np.float64)
            w_T_imu_t1 = kitti_tracking.oxts[obj_frame_idx + 1].T_w_imu.astype(
                np.float64
            )
            w_T_velo_t0 = np.matmul(w_T_imu_t0, imu_T_velo)
            w_T_velo_t1 = np.matmul(w_T_imu_t1, imu_T_velo)
            odom_t0_t1 = np.matmul(np.linalg.inv(w_T_velo_t0), w_T_velo_t1)
            odom_t1_t0 = np.linalg.inv(odom_t0_t1)
            w_T_imu_t2 = kitti_tracking.oxts[obj_frame_idx + 2].T_w_imu.astype(
                np.float64
            )
            w_T_velo_t2 = np.matmul(w_T_imu_t2, imu_T_velo)
            odom_t0_t2 = np.matmul(np.linalg.inv(w_T_velo_t0), w_T_velo_t2)
            odom_t2_t0 = np.linalg.inv(odom_t0_t2)

            # t0->t1
            lidar_flow_t0_t1, track_ids_mask_t0 = extract_lidar_flow_ta_tb(
                pcl_homog_ta=pcl_homog_t0,
                lidar_Tta_obj=lidar_Tt0_obj,
                lidar_Ttb_obj=lidar_Tt1_obj,
                obj_dims_ta=obj_dims_t0,
                track_ids_ta=track_ids_t0,
                track_ids_tb=track_ids_t1,
                odom_ta_tb=odom_t0_t1,
            )
            # t1->t0
            lidar_flow_t1_t0, track_ids_mask_t1 = extract_lidar_flow_ta_tb(
                pcl_homog_ta=pcl_homog_t1,
                lidar_Tta_obj=lidar_Tt1_obj,
                lidar_Ttb_obj=lidar_Tt0_obj,
                obj_dims_ta=obj_dims_t1,
                track_ids_ta=track_ids_t1,
                track_ids_tb=track_ids_t0,
                odom_ta_tb=odom_t1_t0,
            )
            # t0->t2
            lidar_flow_t0_t2, track_ids_mask_t0_t2 = extract_lidar_flow_ta_tb(
                pcl_homog_ta=pcl_homog_t0,
                lidar_Tta_obj=lidar_Tt0_obj,
                lidar_Ttb_obj=lidar_Tt2_obj,
                obj_dims_ta=obj_dims_t0,
                track_ids_ta=track_ids_t0,
                track_ids_tb=track_ids_t2,
                odom_ta_tb=odom_t0_t2,
            )

            assert (track_ids_mask_t0 == track_ids_mask_t0_t2).all()
            # t2->t0
            lidar_flow_t2_t0, track_ids_mask_t2 = extract_lidar_flow_ta_tb(
                pcl_homog_ta=pcl_homog_t2,
                lidar_Tta_obj=lidar_Tt2_obj,
                lidar_Ttb_obj=lidar_Tt0_obj,
                obj_dims_ta=obj_dims_t2,
                track_ids_ta=track_ids_t2,
                track_ids_tb=track_ids_t0,
                odom_ta_tb=odom_t2_t0,
            )

            sample_name = (
                tracking_seq_str + "_" + str(corrected_lidar_frame_idx).zfill(6)
            )
            kiss_odom_t0_t1 = (
                np.linalg.inv(all_poses_w_T_lidar[tracking_seq_str][lidar_fname_t0])
                @ all_poses_w_T_lidar[tracking_seq_str][lidar_fname_t1]
            )
            kiss_odom_t0_t2 = (
                np.linalg.inv(all_poses_w_T_lidar[tracking_seq_str][lidar_fname_t0])
                @ all_poses_w_T_lidar[tracking_seq_str][lidar_fname_t2]
            )
            kiss_odom_t1_t2 = (
                np.linalg.inv(all_poses_w_T_lidar[tracking_seq_str][lidar_fname_t1])
                @ all_poses_w_T_lidar[tracking_seq_str][lidar_fname_t2]
            )
            data_dict = {
                "pcl_t0": pcl_t0.astype(np.float32),
                "pcl_t1": pcl_t1.astype(np.float32),
                "pcl_t2": pcl_t2.astype(np.float32),
                "flow_t0_t1": lidar_flow_t0_t1.astype(np.float32),
                "flow_t1_t0": lidar_flow_t1_t0.astype(np.float32),
                "flow_t0_t2": lidar_flow_t0_t2.astype(np.float32),
                "flow_t2_t0": lidar_flow_t2_t0.astype(np.float32),
                "track_ids_mask_t0": track_ids_mask_t0,
                "track_ids_mask_t1": track_ids_mask_t1,
                "track_ids_mask_t2": track_ids_mask_t2,
                "is_ground_t0": is_ground_t0,
                "is_ground_t1": is_ground_t1,
                "is_ground_t2": is_ground_t2,
                "odom_t0_t1": odom_t0_t1.astype(np.float64),
                "odom_t0_t2": odom_t0_t2.astype(np.float64),
                "objects_t0": assemble_object_dict(
                    lidar_Tt0_obj, obj_dims_t0, track_ids_t0, obj_class_t0
                ),
                "objects_t1": assemble_object_dict(
                    lidar_Tt1_obj, obj_dims_t1, track_ids_t1, obj_class_t1
                ),
                "objects_t2": assemble_object_dict(
                    lidar_Tt2_obj, obj_dims_t2, track_ids_t2, obj_class_t2
                ),
                "name": sample_name,
                "kiss_odom_t0_t1": kiss_odom_t0_t1,
                "kiss_odom_t1_t0": np.linalg.inv(kiss_odom_t0_t1),
                "kiss_odom_t0_t2": kiss_odom_t0_t2,
                "kiss_odom_t2_t0": np.linalg.inv(kiss_odom_t0_t2),
                "kiss_odom_t1_t2": kiss_odom_t1_t2,
                "kiss_odom_t2_t1": np.linalg.inv(kiss_odom_t1_t2),
            }

            np.save(
                target_dir / Path(sample_name),
                data_dict,
            )


def extract_lidar_flow_ta_tb(
    *,
    pcl_homog_ta: np.ndarray,
    lidar_Tta_obj: np.ndarray,
    obj_dims_ta: np.ndarray,
    track_ids_ta: np.ndarray,
    track_ids_tb: np.ndarray,
    lidar_Ttb_obj: np.ndarray,
    odom_ta_tb: np.ndarray,
):
    dummy_track_idx = np.iinfo(np.uint16).max
    lidar_flow_ta_tb = np.einsum(
        "ij,kj->ki",
        np.linalg.inv(odom_ta_tb) - np.eye(4),
        pcl_homog_ta,
    )[:, :3]
    track_ids_mask_ta = (
        np.ones_like(lidar_flow_ta_tb[:, 0], dtype=np.uint16) * dummy_track_idx
    )
    for obj_idx, track_id in enumerate(track_ids_ta):
        lidar_Tta_single_obj = lidar_Tta_obj[obj_idx]
        point_is_in_box_mask_ta = get_points_in_box_mask(
            pcl_homog=pcl_homog_ta,
            lidar_T_obj=lidar_Tta_single_obj,
            obj_size=obj_dims_ta[obj_idx],
        )
        track_ids_mask_ta = np.where(
            point_is_in_box_mask_ta, track_id, track_ids_mask_ta
        )
        if track_id in track_ids_tb:
            corresp_obj_idx_tb = np.argwhere(track_id == track_ids_tb)[0, 0]

            lidar_Ttb_single_obj = lidar_Ttb_obj[corresp_obj_idx_tb]
            cur_dyn_flow_trafo_ta_tb = lidar_Ttb_single_obj @ np.linalg.inv(
                lidar_Tta_single_obj
            )
            cur_dyn_flow_t0_t1 = np.einsum(
                "ij,nj->ni",
                cur_dyn_flow_trafo_ta_tb - np.eye(4),
                pcl_homog_ta,
            )[:, 0:3]
            lidar_flow_ta_tb = np.where(
                point_is_in_box_mask_ta[..., None],
                cur_dyn_flow_t0_t1,
                lidar_flow_ta_tb,
            )

    return lidar_flow_ta_tb, track_ids_mask_ta


def get_data_for_index(
    kitti_tracking, velo_T_cam2, obj_frame_idx, corrected_lidar_frame_idx
):
    pcl_fname = kitti_tracking.velo_files[corrected_lidar_frame_idx]
    (
        pcl_t0,
        pcl_homog_t0,
        is_ground_t0,
    ) = load_kitti_pcl_image_projection_get_ground_label(
        pcl_fname,
        kitti_desc="tracking",
    )
    (
        objs_t0,
        velo_Tt0_obj,
        obj_dims_t0,
        track_ids_t0,
        obj_class_t0,
    ) = get_objects_poses_for_idx(kitti_tracking, velo_T_cam2, obj_frame_idx)

    return (
        pcl_t0,
        pcl_homog_t0,
        is_ground_t0,
        velo_Tt0_obj,
        obj_dims_t0,
        track_ids_t0,
        obj_class_t0,
        objs_t0,
        pcl_fname,
    )


def get_objects_poses_for_idx(
    kitti_tracking: pykitti.tracking, velo_T_cam2: np.ndarray, obj_frame_idx: int
):
    objs_t0 = kitti_tracking.objects[obj_frame_idx]
    if len(objs_t0) > 0:
        cam2_T_obj = np.stack(
            [get_kitti_tracking_object_cam_pose(obj) for obj in objs_t0],
            axis=0,
        )
        velo_T_obj = np.einsum("ik,nkj->nij", velo_T_cam2, cam2_T_obj)
        obj_dims = np.stack(
            [
                np.array(
                    [obj.dimensions.length, obj.dimensions.width, obj.dimensions.height]
                )
                for obj in objs_t0
            ],
            axis=0,
        )
        track_ids = np.stack(
            [obj.track_id for obj in objs_t0],
            axis=0,
        )
        obj_class = np.stack(
            [obj.type for obj in objs_t0],
            axis=0,
        )
    else:
        velo_T_obj = np.empty((0, 4, 4), dtype=np.float64)
        obj_dims = np.empty((0, 3), dtype=np.float64)
        track_ids = np.empty((0), dtype=np.int64)
        obj_class = np.empty((0), dtype=np.array("foo").dtype)
    return objs_t0, velo_T_obj, obj_dims, track_ids, obj_class


if __name__ == "__main__":
    main()
