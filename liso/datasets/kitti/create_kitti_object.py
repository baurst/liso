#!/usr/bin/env python3
import os
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from liso.datasets.kitti.create_kitti_raw import (
    load_kitti_pcl_image_projection_get_ground_label,
)
from liso.utils.torch_transformation import torch_compose_matrix
from mmdet3d.datasets.kitti_dataset import KittiDataset
from mmdetection3d.tools.create_data import kitti_data_prep
from tqdm import tqdm


def get_kitti_filenames_for_sample(
    sample_id, kitti_obj_base_dir, kitti_raw_base_dir, hist_size=5, into_past=True
):
    """
    In case of the object dataset, the sample_id is in [0, 7481) for the training dataset.

    This function returns in this order:
        - list of PCL binary filenames [len: hist_size], from old to new/current

    If the hist_size of PCLs cannot be found in the raw data, None is returned.
    """
    assert 0 <= sample_id < 7481
    mapping_dir = os.path.join(kitti_obj_base_dir, "devkit_object", "mapping")
    map_id_filename = os.path.join(mapping_dir, "train_rand.txt")
    mapping_filename = os.path.join(mapping_dir, "train_mapping.txt")
    map_ids = load_map_ids(map_id_filename)
    map_id = map_ids[sample_id] - 1  # id in file is 1-based
    map_file_content = load_mapping_file_content(mapping_filename)

    map_result = map_file_content[map_id].split(" ")
    raw_frame_idx = map_result[2]
    nbr_digits = len(raw_frame_idx)
    assert raw_frame_idx == ("%%0%dd" % nbr_digits) % int(raw_frame_idx)
    if into_past:
        if int(raw_frame_idx) + 1 < hist_size:
            return None  # not enough history recorded for this event
    raw_dir = os.path.join(
        kitti_raw_base_dir, map_result[0], map_result[1], "velodyne_points", "data"
    )
    past_pcl_binary_filenames = [
        os.path.join(raw_dir, ("%%0%dd.bin" % nbr_digits) % i)
        for i in range(int(raw_frame_idx) + 1 - hist_size, int(raw_frame_idx) + 1)
    ]
    future_pcl_binary_filenames = [
        os.path.join(raw_dir, ("%%0%dd.bin" % nbr_digits) % i)
        for i in range(int(raw_frame_idx), int(raw_frame_idx) + hist_size)
    ]
    assert future_pcl_binary_filenames[0] == past_pcl_binary_filenames[-1], (
        future_pcl_binary_filenames,
        past_pcl_binary_filenames,
    )
    if into_past:
        pcl_binary_filenames = past_pcl_binary_filenames
    else:
        pcl_binary_filenames = future_pcl_binary_filenames
    assert len(pcl_binary_filenames) == hist_size
    for pcl_binary_filename in pcl_binary_filenames:
        if not os.path.exists(pcl_binary_filename):
            print(
                "This pcl file for sample id %d does not exist: %s"
                % (sample_id, pcl_binary_filename)
            )
            print("Excluding this sample from the dataset.")
            return None
    return pcl_binary_filenames


@lru_cache
def load_map_ids(map_id_filename):
    with open(map_id_filename, "r") as map_id_file:
        map_ids = list(map(int, map_id_file.readline().strip().split(",")))
    return map_ids


@lru_cache
def load_mapping_file_content(mapping_filename):
    with open(mapping_filename, "r") as mapping_file:
        map_file_content = mapping_file.read().strip().split("\n")
    return map_file_content


def main():
    argparser = ArgumentParser(
        description="Convert Kitti Object data to training format."
    )
    argparser.add_argument(
        "--target_dir",
        required=True,
        type=Path,
    )
    argparser.add_argument(
        "--kitti_object_root",
        required=True,
        type=Path,
    )
    argparser.add_argument(
        "--kitti_raw_root",
        required=True,
        type=Path,
    )
    args = argparser.parse_args()

    kitti_object_target_dir = args.target_dir / "kitti_object_w_future_pcl"
    info_prefix = "liso"

    kitti_data_prep(
        args.kitti_object_root,
        info_prefix=info_prefix,
        version="foobarbaz",  # version seems to have no influence?
        out_dir="/tmp/mmdet3d_kitti_prep_out",  # we only need the infos_{train|val|test}.pkls, rest can go into garbage bin
        with_plane=False,
    )

    kitti_object_train_ds = KittiDataset(
        args.kitti_object_root,
        ann_file=args.kitti_object_root / f"{info_prefix}_infos_train.pkl",
        modality={"use_lidar": True, "use_camera": False},
        split="training",
    )

    kitti_object_val_ds = KittiDataset(
        args.kitti_object_root,
        ann_file=args.kitti_object_root / f"{info_prefix}_infos_val.pkl",
        modality={"use_lidar": True, "use_camera": False},
        split="training",
    )
    kitti_object_test_ds = KittiDataset(
        args.kitti_object_root,
        ann_file=args.kitti_object_root / f"{info_prefix}_infos_test.pkl",
        modality={"use_lidar": True, "use_camera": False},
        split="testing",
        test_mode=True,
    )

    processed_velo_files = set()
    for dataset_category, kitti_ds in {
        "val": kitti_object_val_ds,
        "train": kitti_object_train_ds,
        "test": kitti_object_test_ds,
    }.items():
        target_dir = kitti_object_target_dir / Path(dataset_category)
        target_dir.mkdir(exist_ok=True, parents=True)
        for mmdet_sample_idx in tqdm(range(len(kitti_ds))):
            data_infos = kitti_ds.get_data_info(mmdet_sample_idx)
            del mmdet_sample_idx
            (
                kitti_pcl,
                _,
                is_ground,
            ) = load_kitti_pcl_image_projection_get_ground_label(
                data_infos["pts_filename"], kitti_desc="object"
            )
            assert (
                data_infos["pts_filename"] not in processed_velo_files
            ), f"processed the same sample twice - please fix and delete {kitti_object_target_dir}"
            processed_velo_files.add(data_infos["pts_filename"])

            sample_name = (
                dataset_category + "_" + str(data_infos["sample_idx"]).zfill(6)
            )
            data_dict = {
                "pcl_t0": kitti_pcl.astype(np.float32),
                "is_ground_t0": is_ground,
                "name": sample_name,
                "img_T_lidar": data_infos["lidar2img"].astype(np.float64),
            }
            if "ann_info" in data_infos:
                annos = data_infos["ann_info"]
                gt_boxes = annos["gt_bboxes_3d"]

                # grab values and add batch dimenson
                obj_dims = gt_boxes.dims.detach().cpu()
                t_x = gt_boxes.center[:, 0][None, ...].to(torch.float64)
                t_y = gt_boxes.center[:, 1][None, ...].to(torch.float64)
                t_z = (
                    gt_boxes.center[:, 2][None, ...].to(torch.float64)
                    + obj_dims[..., -1] * 0.5
                )
                theta_z = gt_boxes.yaw[None, ...].to(torch.float64)

                lidar_T_obj = np.squeeze(
                    torch_compose_matrix(t_x=t_x, t_y=t_y, theta_z=theta_z, t_z=t_z)
                    .detach()
                    .cpu()
                    .numpy(),
                    axis=0,
                )
                obj_dims = obj_dims.numpy()
                data_dict["objects_t0"] = {
                    "poses": lidar_T_obj,
                    "size": obj_dims,
                    "category": annos["gt_names"],
                    "gt_labels_3d": annos["gt_labels_3d"],
                }

            if dataset_category in ("train", "val"):
                # add previous frame to dict
                kitti_obj_sample_idx = data_infos["sample_idx"]
                raw_pcl_fnames = get_kitti_filenames_for_sample(
                    kitti_obj_sample_idx,
                    args.kitti_object_root,
                    kitti_raw_base_dir=args.kitti_raw_root,
                    hist_size=2,
                    into_past=False,
                )
                if raw_pcl_fnames is not None:
                    (
                        pcl_must_have_equal_shape_to_pcl_t0,
                        _,
                        _,
                    ) = load_kitti_pcl_image_projection_get_ground_label(
                        raw_pcl_fnames[0], kitti_desc="raw"
                    )
                    assert (
                        pcl_must_have_equal_shape_to_pcl_t0.shape == kitti_pcl.shape
                    ), "pcl shape mismatch - probably pcl_t1 is not the next frame!"
                    (
                        kitti_pcl_t1,
                        _,
                        is_ground_t1,
                    ) = load_kitti_pcl_image_projection_get_ground_label(
                        raw_pcl_fnames[1], kitti_desc="raw"
                    )
                    data_dict["pcl_t1"] = kitti_pcl_t1.astype(np.float32)
                    data_dict["is_ground_t1"] = is_ground_t1
                else:
                    print(f"No previous data found for {sample_name}.")

            np.save(target_dir / Path(sample_name), data_dict)


if __name__ == "__main__":
    main()
