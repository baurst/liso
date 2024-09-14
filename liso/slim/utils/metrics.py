import numpy as np


def aggregate_metrics(list_of_metrics_dicts_overall):
    in_out_liers_dict_overall = {}
    just_accumulate = ["num_pts_used"]
    for k, _v in list_of_metrics_dicts_overall[0].items():
        if k in just_accumulate:
            in_out_liers_dict_overall[k] = sum(
                el[k] for el in list_of_metrics_dicts_overall
            )
            continue
        in_out_liers_dict_overall[k] = sum(
            el[k] * el["num_pts_used"] for el in list_of_metrics_dicts_overall
        ) / sum(el["num_pts_used"] for el in list_of_metrics_dicts_overall)
    return in_out_liers_dict_overall


def get_inlier_outlier_ratios(
    pred_flow,
    gt_flow,
    inspect_these_points_mask,
):
    end_point_errors = np.linalg.norm(pred_flow - gt_flow, axis=-1)

    acc_3d_0_05 = get_ratio_for_thresh(
        end_point_errors,
        abs_thresh=0.05,
        rel_thresh=0.05,
        gt_flow=gt_flow,
        inspect_these_points_mask=inspect_these_points_mask,
        mode="inliers",
        abs_AND_rel=False,
    )
    acc_3d_0_1 = get_ratio_for_thresh(
        end_point_errors,
        abs_thresh=0.1,
        rel_thresh=0.1,
        gt_flow=gt_flow,
        inspect_these_points_mask=inspect_these_points_mask,
        mode="inliers",
        abs_AND_rel=False,
    )
    outliers_3d = get_ratio_for_thresh(
        end_point_errors,
        abs_thresh=0.3,
        rel_thresh=0.1,
        gt_flow=gt_flow,
        inspect_these_points_mask=inspect_these_points_mask,
        mode="outliers",
        abs_AND_rel=False,
    )
    robust_outliers_3d = get_ratio_for_thresh(
        end_point_errors,
        abs_thresh=0.3,
        rel_thresh=0.3,
        gt_flow=gt_flow,
        inspect_these_points_mask=inspect_these_points_mask,
        mode="outliers",
        abs_AND_rel=True,
    )

    return {
        "ACC3D_0_05": acc_3d_0_05,
        "ACC3D_0_1": acc_3d_0_1,
        "Outliers3D": outliers_3d,
        "RobustOutliers3D": robust_outliers_3d,
    }


def get_ratio_for_thresh(
    end_point_errors,
    abs_thresh,
    rel_thresh,
    gt_flow,
    inspect_these_points_mask,
    mode,
    abs_AND_rel: bool,
):
    # if we are woring in mode=="inlier", variable names with "inlier" refer to inliers
    # otherwise its outliers
    assert mode in ["inliers", "outliers"]

    relative_error = end_point_errors / np.linalg.norm(gt_flow, axis=-1)
    if mode == "inliers":
        point_is_inlier_absolute = end_point_errors < abs_thresh
        point_is_inlier_relative = relative_error < rel_thresh
    else:
        point_is_inlier_absolute = end_point_errors > abs_thresh
        point_is_inlier_relative = relative_error > rel_thresh
    if abs_AND_rel:
        point_is_inlier = np.logical_and(
            point_is_inlier_absolute, point_is_inlier_relative
        )
    else:
        point_is_inlier = np.logical_or(
            point_is_inlier_absolute, point_is_inlier_relative
        )
    num_inliers = np.count_nonzero(
        np.logical_and(point_is_inlier, inspect_these_points_mask)
    )
    num_pts_total = np.count_nonzero(inspect_these_points_mask)
    ratio = num_inliers / num_pts_total
    return ratio


def compute_scene_flow_metrics_for_points_in_this_mask(pred_flow, gt_flow, mask):
    endpoint_errors = np.linalg.norm(pred_flow - gt_flow, axis=-1)

    in_out_liers_dict = get_inlier_outlier_ratios(
        pred_flow,
        gt_flow,
        mask,
    )
    avg_endpoint_error = np.mean(endpoint_errors[mask])

    mean_flow = np.mean(np.linalg.norm(gt_flow, axis=-1)[mask])

    num_pts_used_for_metric = np.count_nonzero(mask)
    return {
        **in_out_liers_dict,
        "AEE": avg_endpoint_error,
        "AVG_FLOW_VECTOR": pred_flow[mask].mean(axis=0),
        "AVG_FLOW_VECTOR_LENGTH": np.mean(np.linalg.norm(pred_flow[mask], axis=-1)),
        "AVG_GT_FLOW_VECTOR": gt_flow[mask].mean(axis=0),
        "AVG_GT_FLOW_VECTOR_LENGTH": np.mean(np.linalg.norm(gt_flow[mask], axis=-1)),
        "AVG_ERROR_FLOW_VECTOR": (pred_flow - gt_flow)[mask].mean(axis=0),
        "num_pts_used": num_pts_used_for_metric,
        "mean_gt_flow": mean_flow,
    }
