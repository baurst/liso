import numpy as np
from liso.transformations.transformations import compose_matrix

nusc_vehicle_T_kitti_lidar = compose_matrix(
    shear=None,
    angles=None,
    translate=np.array([0.95, 0.0, 1.73]),
    perspective=None,
)

kitti_lidar_T_nusc_vehicle = np.linalg.inv(nusc_vehicle_T_kitti_lidar)


def nusc_vehicle_pcl_to_kitti_lidar(pcl_homog):
    assert pcl_homog.shape[-1] == 4, pcl_homog.shape
    assert np.allclose(pcl_homog[:, -1], 1.0)
    return np.einsum("ij,nj->ni", kitti_lidar_T_nusc_vehicle, pcl_homog)
