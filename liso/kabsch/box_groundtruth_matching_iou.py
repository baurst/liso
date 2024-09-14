import numpy as np
import torch
from liso.kabsch.shape_utils import Shape
from liso.utils.nms_iou import box_iou_matrix
from scipy.optimize import linear_sum_assignment


def match_boxes_by_descending_confidence_iou(
    non_batched_gt_boxes: Shape,
    non_batched_pred_boxes: Shape,
    matching_threshold: float,
    iou_mode: str = "iou_bev",
    matching_mode: str = "greedy",
):
    assert iou_mode in ("iou_bev", "iou_3d"), iou_mode
    assert len(non_batched_gt_boxes.shape) == 1, non_batched_gt_boxes.shape
    assert len(non_batched_pred_boxes.shape) == 1, non_batched_pred_boxes.shape

    assert torch.all(non_batched_pred_boxes.valid), "need all valid predictions"
    assert torch.all(non_batched_gt_boxes.valid), "need all valid predictions"

    n_pred = non_batched_pred_boxes.shape[0]
    n_true = non_batched_gt_boxes.shape[0]
    # NUM_GT x NUM_PRED
    if n_pred == 0 or n_true == 0:
        iou_matrix = np.zeros((n_true, n_pred))
    else:
        iou_matrix = (
            box_iou_matrix(non_batched_gt_boxes, non_batched_pred_boxes, iou_mode)
            .cpu()
            .numpy()
        )
    if matching_mode == "greedy":
        sortind = (
            torch.argsort(
                torch.squeeze(non_batched_pred_boxes.probs, axis=-1), descending=True
            )
            .cpu()
            .numpy()
        )

        matched_preds_mask = np.zeros(n_pred, dtype=bool)
        det_gts_mask = np.zeros(n_true, dtype=bool)
        idxs_into_gt = []
        idxs_into_preds = []
        matching_dists = []
        for pred_idx in sortind:
            max_iou = -np.inf
            match_gt_idx = None

            for gt_idx in range(n_true):
                # Find closest match among ground truth boxes
                if gt_idx not in idxs_into_gt:
                    this_iou = iou_matrix[gt_idx, pred_idx]
                    if this_iou > max_iou:
                        max_iou = this_iou
                        match_gt_idx = gt_idx

                # If the closest match is close enough according to threshold we have a match!
            is_match = max_iou > matching_threshold
            if is_match:
                idxs_into_gt.append(match_gt_idx)
                idxs_into_preds.append(pred_idx)
                matched_preds_mask[pred_idx] = True
                det_gts_mask[match_gt_idx] = True
                matching_dists.append(max_iou)
        idxs_into_gt = np.array(idxs_into_gt, dtype=np.int64)
        idxs_into_preds = np.array(idxs_into_preds, dtype=np.int64)
        matching_dists = np.array(matching_dists)

    elif matching_mode == "hungarian":
        MIN_IOU = -1.0
        if n_pred > n_true:
            # add dummy rows (groundtruths)
            diff = n_pred - n_true
            iou_matrix = np.concatenate(
                (iou_matrix, np.full((diff, n_pred), MIN_IOU)), axis=0
            )

        if n_true > n_pred:
            # add dummy columns (predictions)
            diff = n_true - n_pred
            iou_matrix = np.concatenate(
                (iou_matrix, np.full((n_true, diff), MIN_IOU)), axis=1
            )

        nans_infs_mask = ~np.isfinite(iou_matrix)
        if np.any(nans_infs_mask):
            print("WARNING: nans or infs in iou matrix")
            iou_matrix[nans_infs_mask] = MIN_IOU

        idxs_true, matched_pred_idxs = linear_sum_assignment(iou_matrix, maximize=True)

        # remove dummy assignments
        sel_pred = matched_pred_idxs < n_pred
        idx_pred_actual = matched_pred_idxs[sel_pred]
        idx_gt_actual = idxs_true[sel_pred]
        ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
        matched_preds_mask = ious_actual >= matching_threshold

        idxs_into_gt = idx_gt_actual[matched_preds_mask]
        idxs_into_preds = idx_pred_actual[matched_preds_mask]
        matching_dists = ious_actual[matched_preds_mask]
        det_gts_mask = np.zeros(n_true, dtype=bool)
        det_gts_mask[idxs_into_gt] = True
        matched_preds_mask = np.zeros(n_pred, dtype=bool)
        matched_preds_mask[idxs_into_preds] = True
        if len(idxs_into_gt) > 0:
            assert idxs_into_gt.max() < n_true
        if len(idxs_into_preds) > 0:
            assert idxs_into_preds.max() < n_pred
        assert np.count_nonzero(det_gts_mask) == np.count_nonzero(matched_preds_mask)
        assert len(idxs_into_gt) == len(idxs_into_preds), (
            len(idxs_into_gt),
            len(idxs_into_preds),
        )
        assert np.all(matched_preds_mask[idxs_into_preds])

    else:
        raise NotImplementedError(matching_mode)

    return (
        idxs_into_gt,
        idxs_into_preds,
        matching_dists,
        matched_preds_mask,
        det_gts_mask,
    )
