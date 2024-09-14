import numpy as np


def transpose_split_nusc_pcl(pcl):
    pcl = pcl.T
    pcl_3d, intensity, rows = np.split(pcl, [3, 4], axis=1)
    pcl_homog = np.concatenate([pcl_3d, np.ones_like(pcl_3d[:, :1])], axis=-1)
    return pcl_homog, np.squeeze(intensity), np.squeeze(rows.astype(np.int64))
