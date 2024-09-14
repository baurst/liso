#!/usr/bin/env python3
import os.path as osp
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Dict

import matplotlib
import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
from liso.datasets.labelmap import get_label_map_from_file
from liso.datasets.nuscenes.ground_segmentation import transpose_split_nusc_pcl
from liso.datasets.nuscenes.npimgtools.transforms import Transform
from liso.datasets.nuscenes.nuscenes_parser import NuScenesParser
from liso.datasets.nuscenes.trafo_conversion import (
    kitti_lidar_T_nusc_vehicle,
    nusc_vehicle_pcl_to_kitti_lidar,
    nusc_vehicle_T_kitti_lidar,
)
from liso.jcp.jcp import JPCGroundRemove
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm

matplotlib.use("agg")


def create_and_write_sample(
    sample_t0,
    target_path: str,
    *,
    kiss_icp_all_poses_w_T_lidar: Dict[str, Dict[str, np.ndarray]],
    nusc: NuScenesParser,
    skip_existing_files=False,
    force_write_file=False,
) -> str:
    include_data_t0_t2 = True
    scene = nusc.get("scene", sample_t0["scene_token"])
    sample_tokens = nusc.get_token_list("sample", sample_t0["token"], recurse_by=-1)
    sample_idx = sample_tokens.index(sample_t0["token"])
    filename = "%s_%02d_%s" % (scene["name"], sample_idx, sample_t0["token"])
    filename = osp.join(target_path, filename)

    this_scene_kiss_icp_all_poses_w_T_lidar = kiss_icp_all_poses_w_T_lidar[
        scene["name"]
    ]

    if skip_existing_files and osp.isfile(filename + ".npy"):
        return "skipping(exists)"

    # #region compute data
    sd_tok_t0 = sample_t0["data"]["LIDAR_TOP"]
    sample_data_t0 = nusc.get("sample_data", sd_tok_t0)
    timestamp_t0 = sample_t0["timestamp"]
    world_T_t0_nusc_vehicle = nusc.get_ego_pose_at_timestamp(
        sample_t0["scene_token"], timestamp_t0
    ).as_htm()
    world_Tt0_kitti_lid = world_T_t0_nusc_vehicle @ nusc_vehicle_T_kitti_lidar

    (
        ego_mask_t0,
        pcl_t0_kitti_lidar,
        is_ground_t0,
        lidar_intensities_t0,
        lidar_rows_t0,
        lidar_fname_t0,
    ) = load_convert_nusc_lidar_data_kitti_w_ground_labels(
        nusc,
        cur_sd=sample_data_t0,
    )
    inference_base_data_t0 = {
        "world_T_t0_nusc_veh": world_T_t0_nusc_vehicle,
        "world_T_t0_kitti_lid": world_Tt0_kitti_lid,
        "is_ground_t0": is_ground_t0.astype(bool),
        "pcl_t0": pcl_t0_kitti_lidar[:, 0:3].astype(np.float32),
        "lidar_intensities_t0": lidar_intensities_t0.astype(np.float32),
        "lidar_rows_t0": lidar_rows_t0.astype(np.uint8),
        "meta_data_t0": sample_data_t0,
    }
    if sample_t0["next"] != "":
        next_labeled_sample_tok = nusc.get("sample", sample_t0["next"])["data"][
            "LIDAR_TOP"
        ]
        next_labeled_sample_data = nusc.get("sample_data", next_labeled_sample_tok)
        world_T_tx_nusc_veh = nusc.get_ego_pose_at_timestamp(
            sample_t0["scene_token"], next_labeled_sample_data["timestamp"]
        ).as_htm()
        inference_base_data_t0["world_T_tx_nusc_veh"] = world_T_tx_nusc_veh
        inference_base_data_t0["world_T_tx_kitti_lid"] = (
            world_T_tx_nusc_veh @ nusc_vehicle_T_kitti_lidar
        )
        inference_base_data_t0["nusc_veh_t0_T_tx_nusc_veh"] = np.linalg.inv(
            world_T_t0_nusc_vehicle
        ) @ (world_T_tx_nusc_veh)
        inference_base_data_t0["kitti_lid_t0_T_tx_kitti_lid"] = np.linalg.inv(
            world_T_t0_nusc_vehicle @ nusc_vehicle_T_kitti_lidar
        ) @ (world_T_tx_nusc_veh @ nusc_vehicle_T_kitti_lidar)

        (
            _,
            pcl_tx_kitti_lidar,
            is_ground_tx,
            lidar_intensities_tx,
            lidar_rows_tx,
            lidar_fname_tx,
        ) = load_convert_nusc_lidar_data_kitti_w_ground_labels(
            nusc,
            cur_sd=next_labeled_sample_data,
        )
        inference_base_data_t0["kitti_lid_t0_Tkiss_icp_tx_kitti_lid"] = (
            np.linalg.inv(this_scene_kiss_icp_all_poses_w_T_lidar[lidar_fname_t0])
            @ this_scene_kiss_icp_all_poses_w_T_lidar[lidar_fname_tx]
        )

        inference_base_data_t0["pcl_tx"] = pcl_tx_kitti_lidar[:, 0:3].astype(np.float32)
        inference_base_data_t0["lidar_intensities_tx"] = lidar_intensities_tx.astype(
            np.float32
        )
        inference_base_data_t0["is_ground_tx"] = is_ground_tx.astype(bool)
        inference_base_data_t0["lidar_rows_tx"] = lidar_rows_tx.astype(np.uint8)

    if force_write_file:
        # save early, so that we have at least the point cloud for inference available
        np.save(filename + ".npy", inference_base_data_t0)

    nusc2carla_labelmap = get_label_map_from_file("nuscenes", "nuscenes2carla")
    nusc2statdynground_labelmap = get_label_map_from_file(
        "nuscenes", "nuscenes2static_dynamic_ground"
    )
    if force_write_file:
        minimal_object_list = []
        for ann_idx, ann_tok in enumerate(sample_t0["anns"]):
            ann = nusc.get("sample_annotation", ann_tok)
            size = np.array(ann["size"])[[1, 0, 2]]
            instance = nusc.get("instance", ann["instance_token"])
            category_name = ann["category_name"]
            if (
                category_name
                not in nusc2statdynground_labelmap.mname_rnames_dict["dynamic"]
            ):
                # object is not capable of moving
                continue
            vehicle_Tt0_obj = nusc.get_annotation_pose_EGO__m(ann_tok).as_htm()

            lidar_Tt0_obj = kitti_lidar_T_nusc_vehicle @ vehicle_Tt0_obj
            del vehicle_Tt0_obj

            object_attrs = {
                "pose_t0": lidar_Tt0_obj,
                "size": size,
                "annotation_idx": ann_idx,
                "annotation_token": ann_tok,
                "category": category_name,
            }
            minimal_object_list.append(object_attrs)
        # #endregion compute data
        minimal_object_list = np.array(minimal_object_list)
        # save early, so that we have at least the point cloud for inference available
        val_base_data = {**inference_base_data_t0, "objects": minimal_object_list}
        np.save(filename + ".npy", val_base_data)
    framerate__Hz = 10.0
    skip_frames_t0_t1 = 2
    skip_frames_t0_t2 = 4
    assert skip_frames_t0_t1 > 0
    assert skip_frames_t0_t2 > 0
    max_skip = skip_frames_t0_t2 if include_data_t0_t2 else skip_frames_t0_t1
    # take skip-times next to have the target framerate__Hz like in KITTI/CARLA
    next_sd_tokens = nusc.get_token_list("sample_data", sd_tok_t0, check_if_start=False)
    if len(next_sd_tokens) <= max_skip:
        return "not enough follow up sample datas"
    sd_tok_t1 = next_sd_tokens[skip_frames_t0_t1]
    sample_data_t1 = nusc.get("sample_data", sd_tok_t1)

    timestamp_t1 = sample_data_t1["timestamp"]
    frame_diff_0_1 = (timestamp_t1 - timestamp_t0) / (1e6 / 20.0)
    frame_diff_0_1_int = int(round(frame_diff_0_1))
    assert np.abs(frame_diff_0_1 - frame_diff_0_1_int) < 0.3
    if frame_diff_0_1_int != skip_frames_t0_t1:
        return "irregular temporal sampling"
    assert frame_diff_0_1_int == skip_frames_t0_t1
    assert np.allclose(
        timestamp_t1 - timestamp_t0, 1e6 / framerate__Hz, rtol=0.1, atol=5000
    ), (
        timestamp_t1 - timestamp_t0,
        sample_t0["token"],
    )

    world_Tt1_nusc_vehicle = nusc.get_ego_pose_at_timestamp(
        sample_t0["scene_token"], timestamp_t1
    ).as_htm()
    odom_kitti_lidar_t0_t1 = np.linalg.inv(  # kitti lidar means that x is facing forward, but position is nusc lidar
        world_T_t0_nusc_vehicle @ nusc_vehicle_T_kitti_lidar
    ) @ (
        world_Tt1_nusc_vehicle @ nusc_vehicle_T_kitti_lidar
    )
    (
        _,
        pcl_t1_kitti_lidar,
        is_ground_t1,
        lidar_intensities_t1,
        lidar_rows_t1,
        lidar_fname_t1,
    ) = load_convert_nusc_lidar_data_kitti_w_ground_labels(
        nusc,
        cur_sd=sample_data_t1,
    )
    kiss_odom_kitti_lidar_t0_t1 = (
        np.linalg.inv(this_scene_kiss_icp_all_poses_w_T_lidar[lidar_fname_t0])
        @ this_scene_kiss_icp_all_poses_w_T_lidar[lidar_fname_t1]
    )
    try:
        semantics_t0 = nusc.get_lidar_semseg(sample_t0)
    except KeyError as kerr:
        print(kerr)
        return "semantics not found"

    semantics_t0 = semantics_t0[ego_mask_t0]
    # semantics_t0[
    #     semantics_t0 == nusc2carla_labelmap.rname_ridx_dict["vehicle.ego"]
    # ] = nusc2carla_labelmap.rname_ridx_dict["vehicle.car"]

    points_found_norig_t0 = np.zeros_like(semantics_t0, dtype=bool)
    points_found_norig_t1 = np.zeros(pcl_t1_kitti_lidar.shape[0], dtype=bool)
    point_belongs_to_this_object_idx_t0 = np.zeros_like(semantics_t0, dtype=np.uint8)
    point_belongs_to_this_object_idx_t1 = np.zeros_like(
        points_found_norig_t1, dtype=np.uint8
    )

    lidar_flow_t0_t1 = np.einsum(
        "ij,kj->ki",
        np.linalg.inv(odom_kitti_lidar_t0_t1) - np.eye(4),
        pcl_t0_kitti_lidar,
    )
    lidar_flow_t1_t0 = np.einsum(
        "ij,kj->ki",
        odom_kitti_lidar_t0_t1 - np.eye(4),
        pcl_t1_kitti_lidar,
    )
    if include_data_t0_t2:
        sd_tok_t2 = next_sd_tokens[skip_frames_t0_t2]
        sample_data_t2 = nusc.get("sample_data", sd_tok_t2)
        timestamp_t2 = sample_data_t2["timestamp"]
        world_Tt2_nusc_vehicle = nusc.get_ego_pose_at_timestamp(
            sample_t0["scene_token"], timestamp_t2
        ).as_htm()
        odom_kitti_lidar_t0_t2 = np.linalg.inv(  # kitti lidar means that x is facing forward, but position is nusc lidar
            world_T_t0_nusc_vehicle @ nusc_vehicle_T_kitti_lidar
        ) @ (
            world_Tt2_nusc_vehicle @ nusc_vehicle_T_kitti_lidar
        )
        (
            _,
            pcl_t2_kitti_lidar,
            is_ground_t2,
            lidar_intensities_t2,
            lidar_rows_t2,
            lidar_fname_t2,
        ) = load_convert_nusc_lidar_data_kitti_w_ground_labels(
            nusc,
            cur_sd=sample_data_t2,
        )
        kiss_odom_kitti_lidar_t0_t2 = (
            np.linalg.inv(this_scene_kiss_icp_all_poses_w_T_lidar[lidar_fname_t0])
            @ this_scene_kiss_icp_all_poses_w_T_lidar[lidar_fname_t2]
        )
        points_found_norig_t2 = np.zeros(pcl_t2_kitti_lidar.shape[0], dtype=bool)
        point_belongs_to_this_object_idx_t2 = np.zeros_like(
            points_found_norig_t2, dtype=np.uint8
        )

        lidar_flow_t0_t2 = np.einsum(
            "ij,kj->ki",
            np.linalg.inv(odom_kitti_lidar_t0_t2) - np.eye(4),
            pcl_t0_kitti_lidar,
        )
        lidar_flow_t2_t0 = np.einsum(
            "ij,kj->ki",
            odom_kitti_lidar_t0_t2 - np.eye(4),
            pcl_t2_kitti_lidar,
        )
        odom_kitti_lidar_t1_t2 = np.linalg.inv(  # kitti lidar means that x is facing forward, but position is nusc lidar
            world_Tt1_nusc_vehicle @ nusc_vehicle_T_kitti_lidar
        ) @ (
            world_Tt2_nusc_vehicle @ nusc_vehicle_T_kitti_lidar
        )
        kiss_odom_kitti_lidar_t1_t2 = (
            np.linalg.inv(this_scene_kiss_icp_all_poses_w_T_lidar[lidar_fname_t1])
            @ this_scene_kiss_icp_all_poses_w_T_lidar[lidar_fname_t2]
        )
        lidar_flow_t1_t2 = np.einsum(
            "ij,kj->ki",
            np.linalg.inv(odom_kitti_lidar_t1_t2) - np.eye(4),
            pcl_t1_kitti_lidar,
        )
        lidar_flow_t2_t1 = np.einsum(
            "ij,kj->ki",
            odom_kitti_lidar_t1_t2 - np.eye(4),
            pcl_t2_kitti_lidar,
        )

    object_list = []
    for ann_idx, ann_tok in enumerate(sample_t0["anns"]):
        ann = nusc.get("sample_annotation", ann_tok)
        size = np.array(ann["size"])[[1, 0, 2]]
        instance = nusc.get("instance", ann["instance_token"])
        category_name = ann["category_name"]
        if (
            category_name
            not in nusc2statdynground_labelmap.mname_rnames_dict["dynamic"]
        ):
            # object is not capable of moving
            continue
        vehicle_Tt0_obj = nusc.get_annotation_pose_EGO__m(ann_tok).as_htm()
        try:
            wTt1v = Transform(pose_data=world_Tt1_nusc_vehicle, data_format="matrix")

            vehicle_Tt1_obj = (
                wTt1v.copy().invert()
                * nusc.get_interpolated_instance_poses__m(
                    instance, [sample_data_t1["timestamp"]]
                )[0]
            ).as_htm()
        except AssertionError:
            print("Interpolation failed!")
            return "pose interpolation failed"

        lidar_Tt0_obj = kitti_lidar_T_nusc_vehicle @ vehicle_Tt0_obj
        lidar_Tt1_obj = kitti_lidar_T_nusc_vehicle @ vehicle_Tt1_obj
        del vehicle_Tt0_obj
        del vehicle_Tt1_obj

        pcl_t0_OBJ0 = np.einsum(
            "ij,nj->ni", np.linalg.inv(lidar_Tt0_obj), pcl_t0_kitti_lidar
        )
        point_is_in_obj_mask_t0 = (np.abs(pcl_t0_OBJ0[:, 0:3]) < size / 2.0).all(
            axis=-1
        )

        cur_semseg_mask_t0 = (
            semantics_t0 == nusc2carla_labelmap.rname_ridx_dict[ann["category_name"]]
        )
        cur_obj_points_mask_t0 = point_is_in_obj_mask_t0 & cur_semseg_mask_t0

        cur_dyn_flow_trafo_t0_t1 = lidar_Tt1_obj @ np.linalg.inv(lidar_Tt0_obj)
        cur_dyn_flow_t0_t1 = np.einsum(
            "ij,nj->ni", cur_dyn_flow_trafo_t0_t1 - np.eye(4), pcl_t0_kitti_lidar
        )
        lidar_flow_t0_t1[cur_obj_points_mask_t0] = cur_dyn_flow_t0_t1[
            cur_obj_points_mask_t0
        ]
        point_belongs_to_this_object_idx_t0[cur_obj_points_mask_t0] = ann_idx
        points_found_norig_t0[cur_obj_points_mask_t0] = True

        pcl_t1_OBJ0 = np.einsum(
            "ij,nj->ni", np.linalg.inv(lidar_Tt1_obj), pcl_t1_kitti_lidar
        )
        cur_obj_points_mask_t1 = (np.abs(pcl_t1_OBJ0[:, 0:3]) < size / 2.0).all(axis=-1)
        point_belongs_to_this_object_idx_t1[cur_obj_points_mask_t1] = ann_idx
        points_found_norig_t1[cur_obj_points_mask_t1] = True
        cur_dyn_flow_trafo_t1_t0 = lidar_Tt0_obj @ np.linalg.inv(lidar_Tt1_obj)
        cur_dyn_flow_t1_t0 = np.einsum(
            "ij,nj->ni", cur_dyn_flow_trafo_t1_t0 - np.eye(4), pcl_t1_kitti_lidar
        )
        lidar_flow_t1_t0[cur_obj_points_mask_t1] = cur_dyn_flow_t1_t0[
            cur_obj_points_mask_t1
        ]
        object_attrs = {
            "pose_t0": lidar_Tt0_obj,
            "pose_t1": lidar_Tt1_obj,
            "size": size,
            "annotation_idx": ann_idx,
            "annotation_token": ann_tok,
            "category": category_name,
            "num_lidar_pts_t0": np.count_nonzero(point_is_in_obj_mask_t0),
            "box_velocity": nusc.box_velocity(ann["token"]),
        }
        if include_data_t0_t2:
            try:
                wTt2v = Transform(
                    pose_data=world_Tt2_nusc_vehicle, data_format="matrix"
                )
                vehicle_Tt2_obj = (
                    wTt2v.copy().invert()
                    * nusc.get_interpolated_instance_poses__m(
                        instance, [sample_data_t2["timestamp"]]
                    )[0]
                ).as_htm()
            except AssertionError:
                print("Interpolation failed!")
                return "pose interpolation at t2 failed"
            lidar_Tt2_obj = kitti_lidar_T_nusc_vehicle @ vehicle_Tt2_obj
            cur_dyn_flow_trafo_t0_t2 = lidar_Tt2_obj @ np.linalg.inv(lidar_Tt0_obj)
            cur_dyn_flow_t0_t2 = np.einsum(
                "ij,nj->ni", cur_dyn_flow_trafo_t0_t2 - np.eye(4), pcl_t0_kitti_lidar
            )
            lidar_flow_t0_t2[cur_obj_points_mask_t0] = cur_dyn_flow_t0_t2[
                cur_obj_points_mask_t0
            ]
            pcl_t2_OBJ0 = np.einsum(
                "ij,nj->ni", np.linalg.inv(lidar_Tt2_obj), pcl_t2_kitti_lidar
            )
            cur_obj_points_mask_t2 = (np.abs(pcl_t2_OBJ0[:, 0:3]) < size / 2.0).all(
                axis=-1
            )
            point_belongs_to_this_object_idx_t2[cur_obj_points_mask_t2] = ann_idx
            points_found_norig_t2[cur_obj_points_mask_t2] = True
            cur_dyn_flow_trafo_t2_t0 = lidar_Tt0_obj @ np.linalg.inv(lidar_Tt2_obj)
            cur_dyn_flow_t2_t0 = np.einsum(
                "ij,nj->ni", cur_dyn_flow_trafo_t2_t0 - np.eye(4), pcl_t2_kitti_lidar
            )
            lidar_flow_t2_t0[cur_obj_points_mask_t2] = cur_dyn_flow_t2_t0[
                cur_obj_points_mask_t2
            ]
            object_attrs["pose_t2"] = lidar_Tt2_obj

            cur_dyn_flow_trafo_t1_t2 = lidar_Tt2_obj @ np.linalg.inv(lidar_Tt1_obj)
            cur_dyn_flow_t1_t2 = np.einsum(
                "ij,nj->ni", cur_dyn_flow_trafo_t1_t2 - np.eye(4), pcl_t1_kitti_lidar
            )
            lidar_flow_t1_t2[cur_obj_points_mask_t1] = cur_dyn_flow_t1_t2[
                cur_obj_points_mask_t1
            ]
            cur_dyn_flow_trafo_t2_t1 = lidar_Tt1_obj @ np.linalg.inv(lidar_Tt2_obj)
            cur_dyn_flow_t2_t1 = np.einsum(
                "ij,nj->ni", cur_dyn_flow_trafo_t2_t1 - np.eye(4), pcl_t2_kitti_lidar
            )
            lidar_flow_t2_t1[cur_obj_points_mask_t2] = cur_dyn_flow_t2_t1[
                cur_obj_points_mask_t2
            ]

        object_list.append(object_attrs)
    # #endregion compute data
    object_list = np.array(object_list)

    # #region check that all dynamic symantics non rigid flow got
    dynamic_semantics = np.zeros_like(semantics_t0, dtype=bool)
    for rname in nusc2statdynground_labelmap.mname_rnames_dict["dynamic"]:
        ridx = nusc2statdynground_labelmap.rname_ridx_dict[rname]
        assert not (dynamic_semantics & (semantics_t0 == ridx)).any()
        dynamic_semantics[semantics_t0 == ridx] = True
    assert (dynamic_semantics >= points_found_norig_t0).all()

    data_dict = {
        "flow_t0_t1": lidar_flow_t0_t1[:, 0:3].astype(np.float32),
        "flow_t1_t0": lidar_flow_t1_t0[:, 0:3].astype(np.float32),
        "pcl_t1": pcl_t1_kitti_lidar[:, 0:3].astype(np.float32),
        "odom_t0_t1": odom_kitti_lidar_t0_t1.astype(np.float64),
        "kiss_odom_t0_t1": kiss_odom_kitti_lidar_t0_t1.astype(np.float64),
        "is_ground_t1": is_ground_t1.astype(bool),
        "track_ids_mask_t0": point_belongs_to_this_object_idx_t0,
        "track_ids_mask_t1": point_belongs_to_this_object_idx_t1,
        "objects": object_list,
        "lidar_intensities_t1": lidar_intensities_t1.astype(np.float32),
        "lidar_rows_t1": lidar_rows_t1.astype(np.uint8),
        **inference_base_data_t0,
    }
    if include_data_t0_t2:
        data_dict["flow_t0_t2"] = lidar_flow_t0_t2[:, 0:3].astype(np.float32)
        data_dict["flow_t2_t0"] = lidar_flow_t2_t0[:, 0:3].astype(np.float32)
        data_dict["pcl_t2"] = pcl_t2_kitti_lidar[:, 0:3].astype(np.float32)
        data_dict["odom_t0_t2"] = odom_kitti_lidar_t0_t2.astype(np.float64)
        data_dict["kiss_odom_t0_t2"] = kiss_odom_kitti_lidar_t0_t2.astype(np.float64)
        data_dict["is_ground_t2"] = is_ground_t2.astype(bool)
        data_dict["track_ids_mask_t2"] = point_belongs_to_this_object_idx_t2
        data_dict["lidar_intensities_t2"] = lidar_intensities_t2.astype(np.float32)
        data_dict["lidar_rows_t2"] = lidar_rows_t2.astype(np.uint8)
        data_dict["flow_t1_t2"] = lidar_flow_t1_t2[:, 0:3].astype(np.float32)
        data_dict["flow_t2_t1"] = lidar_flow_t2_t1[:, 0:3].astype(np.float32)
        data_dict["odom_t1_t2"] = odom_kitti_lidar_t1_t2.astype(np.float64)
        data_dict["kiss_odom_t1_t2"] = kiss_odom_kitti_lidar_t1_t2.astype(np.float64)
    np.save(filename + ".npy", data_dict)

    return "fine"


def load_convert_nusc_lidar_data_kitti_w_ground_labels(
    nusc: NuScenesParser,
    *,
    cur_sd,
):
    pcl_t0_vehicle, ego_mask_t0 = nusc.get_pointcloud(cur_sd, ref_frame="ego")
    assert cur_sd["channel"] == "LIDAR_TOP", cur_sd["channel"]
    pcl_filename = cur_sd["filename"]
    pcl_t0_3d, intensities, rows = transpose_split_nusc_pcl(pcl_t0_vehicle)
    pcl_t0_kitti_lidar = nusc_vehicle_pcl_to_kitti_lidar(pcl_t0_3d)
    is_ground_label = JPCGroundRemove(
        pcl=pcl_t0_kitti_lidar[:, :3],
        range_img_width=1024,
        range_img_height=32,
        sensor_height=1.8,
        delta_R=1,
    )
    return (
        ego_mask_t0,
        pcl_t0_kitti_lidar,
        is_ground_label,
        intensities,
        rows,
        pcl_filename,
    )


def main(
    path_out: str,
    nusc_root: str,
    version="v1.0-mini",
    split=None,
    skip_existing_files=False,
):
    Path(path_out).mkdir(parents=True, exist_ok=True)
    if split:
        map_scene_to_split = defaultdict(list)
        for data_category, scene_names in split.items():
            for scene_name in scene_names:
                map_scene_to_split[scene_name].append(data_category)
        for data_category, scene_names in map_scene_to_split.items():
            assert len(scene_names) == 1
            map_scene_to_split[data_category] = scene_names[0]
        assert len(map_scene_to_split) == sum(
            [len(v) for v in split.values()]
        ), "scene cannot be in multiple categories (i.e. train and test at the same time)- duplicates!"
        for category in split:
            Path(path_out).joinpath(category).mkdir(exist_ok=True)
    nusc = NuScenesParser(version=version, dataroot=nusc_root, verbose=True)
    results = {}
    count_results = {}
    poses_file_wo_ext = path_out / "poses_kiss_icp"
    all_poses_w_T_lidar = (
        try_load_or_compute_nuscenes_kiss_icp_lidar_poses_in_kitti_coordinates(
            nusc, poses_file_wo_ext
        )
    )

    for sample in tqdm(nusc.sample):
        if split:
            scene_name = nusc.get("scene", sample["scene_token"])["name"]
            split_name = map_scene_to_split[scene_name]
            out_dir = path_out / Path(split_name)
            if "val" in split_name or "test" in split_name:
                # we do not need t0_t2 flow for validation data
                is_test_file = True
            else:
                is_test_file = False

        else:
            out_dir = path_out

        cur_result = create_and_write_sample(
            sample,
            out_dir,
            nusc=nusc,
            kiss_icp_all_poses_w_T_lidar=all_poses_w_T_lidar,
            skip_existing_files=skip_existing_files,
            force_write_file=is_test_file,
        )
        results[sample["token"]] = cur_result
        if cur_result not in count_results:
            count_results[cur_result] = 0
        count_results[cur_result] += 1
        tqdm.write(str(count_results))


def try_load_or_compute_nuscenes_kiss_icp_lidar_poses_in_kitti_coordinates(
    nusc, poses_file_wo_ext
):
    all_poses_w_T_lidar = {}
    try:
        all_poses_w_T_lidar = np.load(
            poses_file_wo_ext.with_suffix(".npy"), allow_pickle=True
        ).item()
    except FileNotFoundError as e:
        print(e)
        print("Recomputing full KISS odometry from scratch")

    print("Getting all scenes...")
    all_scenes = {}
    for sample in nusc.sample:
        scene = nusc.get("scene", sample["scene_token"])
        scene_name = scene["name"]
        all_scenes[scene_name] = scene
    print("Done.")
    print("Computing KISS-ICP poses for all scenes.")

    for scene_name, scene in tqdm(all_scenes.items()):
        if scene_name in all_poses_w_T_lidar:
            continue
        sample_record = nusc.get("sample", scene["first_sample_token"])
        sample_record_lidar = nusc.get(
            "sample_data", sample_record["data"]["LIDAR_TOP"]
        )
        lidar_timestamp_micros = sample_record_lidar["timestamp"]
        kiss_config = KISSConfig()
        kiss_config.mapping.voxel_size = 0.01 * kiss_config.data.max_range
        pcl_filenames = []
        odometry = KissICP(config=kiss_config)
        (
            _,
            pcl_t0_kitti_lidar,
            _,
            _,
            _,
            pcl_t0_filename,
        ) = load_convert_nusc_lidar_data_kitti_w_ground_labels(
            nusc,
            cur_sd=sample_record_lidar,
        )
        pcl_filenames.append(pcl_t0_filename)

        odometry.register_frame(
            pcl_t0_kitti_lidar[:, :3],
            np.zeros(pcl_t0_kitti_lidar.shape[0]).astype(np.float64),
        )
        while not sample_record_lidar["next"] == "":
            sample_record_lidar = nusc.get("sample_data", sample_record_lidar["next"])
            time_delta = sample_record_lidar["timestamp"] - lidar_timestamp_micros
            assert (
                time_delta < 150000
            ), f"time gap {time_delta} too large, something went wrong - expected less than 150ms"
            lidar_timestamp_micros = sample_record_lidar["timestamp"]
            (
                _,
                pcl_t0_kitti_lidar,
                _,
                _,
                _,
                pcl_t0_filename,
            ) = load_convert_nusc_lidar_data_kitti_w_ground_labels(
                nusc,
                cur_sd=sample_record_lidar,
            )
            pcl_filenames.append(pcl_t0_filename)

            odometry.register_frame(
                pcl_t0_kitti_lidar[:, :3],
                np.zeros(pcl_t0_kitti_lidar.shape[0]).astype(np.float64),
            )

        all_poses_w_T_lidar[scene_name] = dict(zip(pcl_filenames, odometry.poses))

        np.save(poses_file_wo_ext, all_poses_w_T_lidar, allow_pickle=True)
    return all_poses_w_T_lidar


if __name__ == "__main__":
    argparser = ArgumentParser(description="Convert nuscenes data to traning format")
    argparser.add_argument("--skip_existing_files", action="store_true")
    argparser.add_argument(
        "--target_dir",
        type=Path,
        required=True,
        help="if path starts with /tmp, we will only process nuscenes-mini",
    )
    argparser.add_argument("--nusc_root", required=True, type=Path)
    args = argparser.parse_args()

    prod = not str(args.target_dir).startswith("/tmp")
    nusc_split = create_splits_scenes()
    if prod:
        split = {"train": nusc_split["train"], "val": nusc_split["val"]}
        version = "v1.0-trainval"
    else:
        split = {
            "mini_train": nusc_split["mini_train"],
            "mini_val": nusc_split["mini_val"],
        }
        version = "v1.0-mini"

    main(
        path_out=args.target_dir,
        nusc_root=args.nusc_root,
        version=version,
        split=split,
        skip_existing_files=args.skip_existing_files,
    )
    if prod:
        split = {
            "test": nusc_split["test"],
        }
        version = "v1.0-test"
        main(
            path_out=args.target_dir,
            nusc_root=args.nusc_root,
            version=version,
            split=split,
            skip_existing_files=args.skip_existing_files,
        )
