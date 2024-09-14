import numpy as np
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

"""
Extracted FROM: https://github.com/YurongYou/MODEST/blob/ddd77f9e8750a15f94a062b033c85bf8a1598583/generate_cluster_mask/utils/pointcloud_utils.py
"""


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.
    https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """

    pi2 = np.pi / 2.0

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points) - 1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack(
        [np.cos(angles), np.cos(angles - pi2), np.cos(angles + pi2), np.cos(angles)]
    ).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval, angles[best_idx], areas[best_idx]


def PCA_rectangle(cluster_ptc):
    components = PCA(n_components=2).fit(cluster_ptc).components_
    on_component_ptc = cluster_ptc @ components.T
    min_x, max_x = on_component_ptc[:, 0].min(), on_component_ptc[:, 0].max()
    min_y, max_y = on_component_ptc[:, 1].min(), on_component_ptc[:, 1].max()
    area = (max_x - min_x) * (max_y - min_y)

    rval = np.array(
        [
            [max_x, min_y],
            [min_x, min_y],
            [min_x, max_y],
            [max_x, max_y],
        ]
    )
    rval = rval @ components
    angle = np.arctan2(components[0, 1], components[0, 0])
    return rval, angle, area


# numba: | 31/700 [05:09<1:51:19,  9.98s/it]
# no numba | 33/700 [04:59<1:41:00,  9.09s/it]
# @numba.jit(nopython=True)
def closeness_rectangle(cluster_ptc, delta=5.0, d0=1e-2):
    max_beta = -np.inf
    choose_angle = 0.0
    for angle in np.arange(0, 90 + delta, delta):
        angle = angle / 180.0 * np.pi
        components = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        projection = cluster_ptc @ components.T
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()
        Dx = np.minimum(projection[:, 0] - min_x, max_x - projection[:, 0])
        Dy = np.minimum(projection[:, 1] - min_y, max_y - projection[:, 1])
        beta = np.minimum(Dx, Dy)
        beta = np.maximum(beta, d0)
        beta = 1 / beta
        beta = beta.sum()
        if beta > max_beta:
            max_beta = beta
            choose_angle = angle
    angle = choose_angle
    components = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    projection = cluster_ptc @ components.T
    min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
    min_y, max_y = projection[:, 1].min(), projection[:, 1].max()

    if (max_x - min_x) < (max_y - min_y):
        angle = choose_angle + np.pi / 2
        components = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        projection = cluster_ptc @ components.T
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()

    area = (max_x - min_x) * (max_y - min_y)

    rval = np.array(
        [
            [max_x, min_y],
            [min_x, min_y],
            [min_x, max_y],
            [max_x, max_y],
        ]
    )
    rval = rval @ components
    return rval, angle, area


def variance_rectangle(cluster_ptc, delta=0.1):
    max_var = -float("inf")
    choose_angle = None
    for angle in np.arange(0, 90 + delta, delta):
        angle = angle / 180.0 * np.pi
        components = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        projection = cluster_ptc @ components.T
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()
        Dx = np.vstack((projection[:, 0] - min_x, max_x - projection[:, 0])).min(axis=0)
        Dy = np.vstack((projection[:, 1] - min_y, max_y - projection[:, 1])).min(axis=0)
        Ex = Dx[Dx < Dy]
        Ey = Dy[Dy < Dx]
        var = 0
        if (Dx < Dy).sum() > 0:
            var += -np.var(Ex)
        if (Dy < Dx).sum() > 0:
            var += -np.var(Ey)
        # print(angle, var)
        if var > max_var:
            max_var = var
            choose_angle = angle
    # print(choose_angle, max_var)
    angle = choose_angle
    components = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    projection = cluster_ptc @ components.T
    min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
    min_y, max_y = projection[:, 1].min(), projection[:, 1].max()

    if (max_x - min_x) < (max_y - min_y):
        angle = choose_angle + np.pi / 2
        components = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        projection = cluster_ptc @ components.T
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()

    area = (max_x - min_x) * (max_y - min_y)

    rval = np.array(
        [
            [max_x, min_y],
            [min_x, min_y],
            [min_x, max_y],
            [max_x, max_y],
        ]
    )
    rval = rval @ components
    return rval, angle, area


# this was the original code: coords seem to be xzy?
# def get_lowest_point_rect(ptc, xz_center, l, w, ry):
#     ptc_xz = ptc[:, [0, 2]] - xz_center
#     rot = np.array([[np.cos(ry), -np.sin(ry)], [np.sin(ry), np.cos(ry)]])
#     ptc_xz = ptc_xz @ rot.T
#     mask = (
#         (ptc_xz[:, 0] > -l / 2)
#         & (ptc_xz[:, 0] < l / 2)
#         & (ptc_xz[:, 1] > -w / 2)
#         & (ptc_xz[:, 1] < w / 2)
#     )
#     ys = ptc[mask, 1]
#     return ys.max()
# def get_obj(ptc, full_ptc, fit_method="min_zx_area_fit"):
#     if fit_method == "min_zx_area_fit":
#         corners, ry, area = minimum_bounding_rectangle(ptc[:, [0, 2]])
#     elif fit_method == "PCA":
#         corners, ry, area = PCA_rectangle(ptc[:, [0, 2]])
#     elif fit_method == "variance_to_edge":
#         corners, ry, area = variance_rectangle(ptc[:, [0, 2]])
#     elif fit_method == "closeness_to_edge":
#         corners, ry, area = closeness_rectangle(ptc[:, [0, 2]])
#     else:
#         raise NotImplementedError(fit_method)
#     ry *= -1
#     l = np.linalg.norm(corners[0] - corners[1])
#     w = np.linalg.norm(corners[0] - corners[-1])
#     c = (corners[0] + corners[2]) / 2
#     # bottom = ptc[:, 1].max()
#     bottom = get_lowest_point_rect(full_ptc, c, l, w, ry)
#     h = bottom - ptc[:, 1].min()
#     obj = types.SimpleNamespace()
#     obj.t = np.array([c[0], bottom, c[1]])
#     obj.l = l
#     obj.w = w
#     obj.h = h
#     obj.ry = ry
#     obj.volume = area * h
#     return obj
#


def fit_2d_box_modest(ptc, fit_method):
    assert ptc.shape[-1] == 3, ptc.shape
    if fit_method == "min_zx_area_fit":
        corners, ry, area = minimum_bounding_rectangle(ptc[:, [0, 1]])
    elif fit_method == "PCA":
        corners, ry, area = PCA_rectangle(ptc[:, [0, 1]])
    elif fit_method == "variance_to_edge":
        corners, ry, area = variance_rectangle(ptc[:, [0, 1]])
    elif fit_method == "closeness_to_edge":
        corners, ry, area = closeness_rectangle(ptc[:, [0, 1]])
    else:
        raise NotImplementedError(fit_method)
    # ry *= -1
    box_length = np.linalg.norm(corners[0] - corners[1])
    box_width = np.linalg.norm(corners[0] - corners[-1])
    box_center = (corners[0] + corners[2]) / 2
    return box_center, box_length, box_width, ry
