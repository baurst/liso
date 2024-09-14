import numpy as np


def find_jumps(
    pointcloud: np.ndarray, threshold: float = -0.005, auto_correct=False
) -> np.ndarray:
    azimuth_flipped = -np.arctan2(pointcloud[:, 1], -pointcloud[:, 0])
    jumps = np.argwhere(np.ediff1d(azimuth_flipped) < threshold)
    rows = np.zeros(shape=azimuth_flipped.shape, dtype=np.int32)
    rows[jumps + 1] = 1
    rows = np.cumsum(rows, dtype=np.int32)

    if rows[-1] < 63 and auto_correct:
        rows += 63 - rows[-1]

    return rows


def find_cols(pointcloud: np.ndarray, num_columns=2000) -> np.ndarray:
    np_azi = np.arctan2(pointcloud[:, 1], pointcloud[:, 0])
    np_col_indices = (num_columns - 1) * (np.pi - np_azi) / (2 * np.pi)
    np_col_indices = np_col_indices.astype(np.int32)
    return np_col_indices


def kitti_pcl_projection_get_rows_cols(pointcloud: np.ndarray, num_columns=2000):
    np_rows = find_jumps(pointcloud)
    np_cols = find_cols(pointcloud, num_columns=num_columns)
    return np_rows, np_cols
