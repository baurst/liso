#!/usr/bin/env python3
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path

import numpy as np
import pykitti
from liso.jcp.jcp import JPCGroundRemove
from tqdm import tqdm


@lru_cache(maxsize=32)
def load_kitti_pcl_image_projection_get_ground_label(velo_file: str, kitti_desc="raw"):
    assert kitti_desc in ("raw", "tracking", "object"), kitti_desc
    kitti_pcl = pykitti.utils.load_velo_scan(velo_file)
    is_ground = JPCGroundRemove(
        pcl=kitti_pcl[:, :3],
        range_img_width=2083,
        range_img_height=64,
        sensor_height=1.73,
        delta_R=1,
    )
    homog_pcl = np.copy(kitti_pcl)
    assert len(homog_pcl.shape) == 2, homog_pcl.shape
    assert homog_pcl.shape[-1] == 4, homog_pcl.shape
    homog_pcl[:, -1] = 1.0
    return kitti_pcl, homog_pcl, is_ground


def correct_kitti_scan(frame: np.ndarray):
    from kiss_icp.pybind import kiss_icp_pybind

    assert frame.dtype == np.float64
    return np.asarray(
        kiss_icp_pybind._correct_kitti_scan(kiss_icp_pybind._Vector3dVector(frame))
    )


def main():
    from kiss_icp.config import KISSConfig
    from kiss_icp.datasets.kitti_raw import KITTIRawDataset
    from kiss_icp.kiss_icp import KissICP

    argparser = ArgumentParser(description="Convert kitti raw data to training format.")
    argparser.add_argument(
        "--target_dir",
        required=True,
        type=Path,
    )
    argparser.add_argument(
        "--kitti_raw_root",
        required=True,
        type=Path,
    )
    args = argparser.parse_args()

    target_dir = args.target_dir / "kitti_raw"

    target_dir.mkdir(parents=True, exist_ok=True)

    dates = ["2011_09_26", "2011_09_28", "2011_09_29", "2011_09_30", "2011_10_03"]

    skipped_sequences = 0
    success = 0
    for date in tqdm(dates):
        drives_strs = [str(i).zfill(4) for i in range(1000)]

        for drive_str in tqdm(drives_strs, leave=False):
            try:
                kitti = pykitti.raw(args.kitti_raw_root, date, drive_str)
            except FileNotFoundError:
                skipped_sequences += 1
                continue
            kiss_config = KISSConfig()
            kiss_config.mapping.voxel_size = 0.01 * kiss_config.data.max_range
            odometry = KissICP(config=kiss_config)
            seq_idxs = list(range(0, len(kitti.velo_files) - 2, 1))
            fnames = []
            for idx in tqdm(seq_idxs, leave=False):
                idx_str_t0 = Path(kitti.velo_files[idx]).stem

                (
                    pcl_t0,
                    _,
                    is_ground_t0,
                ) = load_kitti_pcl_image_projection_get_ground_label(
                    kitti.velo_files[idx]
                )
                timestamps = KITTIRawDataset.get_timestamps(pcl_t0).astype(np.float64)
                odometry.register_frame(
                    correct_kitti_scan(np.copy(pcl_t0[:, :3]).astype(np.float64)),
                    timestamps=timestamps,
                )
                (
                    pcl_t1,
                    _,
                    is_ground_t1,
                ) = load_kitti_pcl_image_projection_get_ground_label(
                    kitti.velo_files[idx + 1]
                )
                (
                    pcl_t2,
                    _,
                    is_ground_t2,
                ) = load_kitti_pcl_image_projection_get_ground_label(
                    kitti.velo_files[idx + 2]
                )

                w_T_imu_t0 = kitti.oxts[idx].T_w_imu.astype(np.float64)
                w_T_imu_t1 = kitti.oxts[idx + 1].T_w_imu.astype(np.float64)
                w_T_imu_t2 = kitti.oxts[idx + 2].T_w_imu.astype(np.float64)
                imu_T_velo = np.linalg.inv(kitti.calib.T_velo_imu.astype(np.float64))

                w_T_velo_t0 = np.matmul(w_T_imu_t0, imu_T_velo)
                w_T_velo_t1 = np.matmul(w_T_imu_t1, imu_T_velo)
                w_T_velo_t2 = np.matmul(w_T_imu_t2, imu_T_velo)

                odom_t0_t1 = np.matmul(np.linalg.inv(w_T_velo_t0), w_T_velo_t1)
                odom_t0_t2 = np.matmul(np.linalg.inv(w_T_velo_t0), w_T_velo_t2)
                sample_name = "{0}_{1}_{2}".format(date, drive_str, idx_str_t0)
                data_dict = {
                    "pcl_t0": pcl_t0.astype(np.float32),
                    "pcl_t1": pcl_t1.astype(np.float32),
                    "pcl_t2": pcl_t2.astype(np.float32),
                    "is_ground_t0": is_ground_t0,
                    "is_ground_t1": is_ground_t1,
                    "is_ground_t2": is_ground_t2,
                    "odom_t0_t1": odom_t0_t1.astype(np.float64),
                    "odom_t0_t2": odom_t0_t2.astype(np.float64),
                    "name": sample_name,
                }

                target_fname = target_dir / Path(sample_name)
                fnames.append(target_fname)
                np.save(
                    target_fname,
                    data_dict,
                )
                if idx == seq_idxs[-1]:
                    timestamps = KITTIRawDataset.get_timestamps(pcl_t1).astype(
                        np.float64
                    )
                    odometry.register_frame(
                        correct_kitti_scan(np.copy(pcl_t1[:, :3]).astype(np.float64)),
                        timestamps=timestamps,
                    )
                    timestamps = KITTIRawDataset.get_timestamps(pcl_t2).astype(
                        np.float64
                    )
                    odometry.register_frame(
                        correct_kitti_scan(np.copy(pcl_t2[:, :3]).astype(np.float64)),
                        timestamps=timestamps,
                    )

                success += 1
            w_Ts_si = odometry.poses
            for file_idx, fname in enumerate(fnames):
                content = np.load(fname.with_suffix(".npy"), allow_pickle=True).item()
                kiss_odom_t0_t1 = (
                    np.linalg.inv(w_Ts_si[file_idx]) @ w_Ts_si[file_idx + 1]
                )
                kiss_odom_t0_t2 = (
                    np.linalg.inv(w_Ts_si[file_idx]) @ w_Ts_si[file_idx + 2]
                )
                kiss_odom_t1_t2 = (
                    np.linalg.inv(w_Ts_si[file_idx + 1]) @ w_Ts_si[file_idx + 2]
                )
                content["kiss_odom_t0_t1"] = kiss_odom_t0_t1
                content["kiss_odom_t1_t0"] = np.linalg.inv(kiss_odom_t0_t1)
                content["kiss_odom_t0_t2"] = kiss_odom_t0_t2
                content["kiss_odom_t2_t0"] = np.linalg.inv(kiss_odom_t0_t2)
                content["kiss_odom_t1_t2"] = kiss_odom_t1_t2
                content["kiss_odom_t2_t1"] = np.linalg.inv(kiss_odom_t1_t2)

                np.save(fname, content)

    print("Skipped: {0} Success: {1}".format(skipped_sequences, success))


if __name__ == "__main__":
    main()
