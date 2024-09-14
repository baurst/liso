from typing import Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix


@torch.no_grad()
def batched_match_bboxes(
    groundtruth_bboxes,
    predicted_bboxes,
    MAX_DIST_PADDING_VALUE,
    DIST_MATCHING_THRESHOLD,
):
    pred_pos = predicted_bboxes.pos.detach()
    gt_pos = groundtruth_bboxes.pos.detach()
    # filter invalid gt boxes by padding with MAX_DIST_PADDING_VALUE
    dist_mat = torch.cdist(gt_pos.to(torch.float32), pred_pos)
    dist_mat[~groundtruth_bboxes.valid] = MAX_DIST_PADDING_VALUE

    # filter invalid predictions by padding with MAX_DIST_PADDING_VALUE
    dist_mat_perm_pred_first = dist_mat.permute(
        0, 2, 1
    )  # permute so that broadcasting works
    dist_mat_perm_pred_first[~predicted_bboxes.valid] = MAX_DIST_PADDING_VALUE
    dist_mat = dist_mat_perm_pred_first.permute(0, 2, 1)
    assert dist_mat.shape[1:] == (gt_pos.shape[1], pred_pos.shape[1]), (
        dist_mat.shape,
        gt_pos.shape,
        pred_pos.shape,
    )
    dist_matrix = dist_mat.cpu().numpy()

    bs, n_pred, _ = pred_pos.shape
    _, n_true, _ = gt_pos.shape

    if n_pred > n_true:
        # add dummy rows (groundtruths)
        diff = n_pred - n_true
        dist_matrix = np.concatenate(
            (dist_matrix, np.full((bs, diff, n_pred), MAX_DIST_PADDING_VALUE)), axis=1
        )

    if n_true > n_pred:
        # add dummy columns (predictions)
        diff = n_true - n_pred
        dist_matrix = np.concatenate(
            (dist_matrix, np.full((bs, n_true, diff), MAX_DIST_PADDING_VALUE)), axis=2
        )
    idxs_true, matched_pred_idxs = zip(
        *[linear_sum_assignment(dist_matrix[i]) for i in range(bs)]
    )
    idxs_true = np.stack(idxs_true, axis=0)
    matched_pred_idxs = np.stack(matched_pred_idxs, axis=0)
    max_pad = max(n_pred, n_true)
    batch_idxs = np.tile(np.arange(0, bs, 1, dtype=np.int64)[..., None], (1, max_pad))
    idxs_true = np.stack([batch_idxs, idxs_true], axis=-1)
    matched_pred_idxs = np.stack([batch_idxs, matched_pred_idxs], axis=-1)

    # remove dummy assignments
    sel_pred = matched_pred_idxs[..., 1] < n_pred
    idx_pred_actual = matched_pred_idxs[sel_pred]
    idx_gt_actual = idxs_true[sel_pred]
    assert (
        idx_gt_actual[..., 0] == idx_pred_actual[..., 0]
    ).all(), "cross-batch match occured"
    dists_actual = dist_matrix[
        idx_gt_actual[..., 0], idx_gt_actual[..., 1], idx_pred_actual[..., 1]
    ]
    matched_preds_mask = dists_actual < DIST_MATCHING_THRESHOLD

    idxs_gt = idx_gt_actual[matched_preds_mask]
    idxs_pred = idx_pred_actual[matched_preds_mask]
    matching_dists = dists_actual[matched_preds_mask]
    detected_gts_mask = np.zeros((bs, n_true), dtype=bool)
    detected_gts_mask[idxs_gt[..., 0], idxs_gt[..., 1]] = True

    return idxs_gt, idxs_pred, matching_dists, matched_preds_mask, detected_gts_mask


@torch.no_grad()
def match_bboxes(
    gt_pos: np.ndarray,
    pred_pos: np.ndarray,
    DIST_MATCHING_THRESHOLD=15.0,
    match_in_nd=3,
):
    """
    gt_pos: shape: [n_true x n_dims]
    pred_pos: shape: [n_pred x n_dims]
    see https://gist.github.com/AruniRC/c629c2df0e68e23aff7dcaeef87c72d4
    """

    assert len(gt_pos.shape) == 2, gt_pos.shape
    assert len(pred_pos.shape) == 2, pred_pos.shape

    n_pred = pred_pos.shape[0]
    n_true = gt_pos.shape[0]
    MAX_DIST = 1000.0
    # NUM_GT x NUM_PRED
    dist_matrix = torch.cdist(
        gt_pos[..., :match_in_nd].to(torch.float32),
        pred_pos[..., :match_in_nd].to(torch.float32),
    )
    dist_matrix = dist_matrix.cpu().numpy()

    if n_pred > n_true:
        # add dummy rows (groundtruths)
        diff = n_pred - n_true
        dist_matrix = np.concatenate(
            (dist_matrix, np.full((diff, n_pred), MAX_DIST)), axis=0
        )

    if n_true > n_pred:
        # add dummy columns (predictions)
        diff = n_true - n_pred
        dist_matrix = np.concatenate(
            (dist_matrix, np.full((n_true, diff), MAX_DIST)), axis=1
        )

    idxs_true, matched_pred_idxs = linear_sum_assignment(dist_matrix)

    # remove dummy assignments
    sel_pred = matched_pred_idxs < n_pred
    idx_pred_actual = matched_pred_idxs[sel_pred]
    idx_gt_actual = idxs_true[sel_pred]
    dists_actual = dist_matrix[idx_gt_actual, idx_pred_actual]
    matched_preds_mask = dists_actual < DIST_MATCHING_THRESHOLD

    idxs_gt = idx_gt_actual[matched_preds_mask]
    idxs_pred = idx_pred_actual[matched_preds_mask]
    matching_dists = dists_actual[matched_preds_mask]
    detected_gts_mask = np.zeros(n_true, dtype=bool)
    detected_gts_mask[idxs_gt] = True
    matched_preds_mask = np.zeros(n_pred, dtype=bool)
    matched_preds_mask[idxs_pred] = True
    if len(idxs_gt) > 0:
        assert idxs_gt.max() < n_true
    if len(idxs_pred) > 0:
        assert idxs_pred.max() < n_pred
    assert np.count_nonzero(detected_gts_mask) == np.count_nonzero(matched_preds_mask)
    assert len(idxs_gt) == len(idxs_pred), (len(idxs_gt), len(idxs_pred))
    assert np.all(matched_preds_mask[idxs_pred])
    return (
        idxs_gt,
        idxs_pred,
        matching_dists,
        matched_preds_mask,
        detected_gts_mask,
    )


def slow_greedy_match_boxes_by_desending_confidence_by_dist(
    non_batched_gt_boxes_pos: Union[torch.FloatTensor, np.ndarray],
    non_batched_pred_boxes_pos: Union[torch.FloatTensor, np.ndarray],
    non_batched_pred_confidence: Union[torch.FloatTensor, np.ndarray],
    matching_threshold: float,
    match_in_nd=3,
):
    assert len(non_batched_gt_boxes_pos.shape) == 2, non_batched_gt_boxes_pos.shape
    assert len(non_batched_pred_boxes_pos.shape) == 2, non_batched_pred_boxes_pos.shape
    assert (
        len(non_batched_pred_confidence.shape) == 1
    ), non_batched_pred_confidence.shape

    n_pred = non_batched_pred_boxes_pos.shape[0]

    assert non_batched_pred_confidence.shape[0] == n_pred, (
        non_batched_pred_confidence.shape[0],
        n_pred,
    )
    n_true = non_batched_gt_boxes_pos.shape[0]
    if torch.is_tensor(non_batched_gt_boxes_pos):
        # NUM_GT x NUM_PRED
        dist_matrix = torch.cdist(
            non_batched_gt_boxes_pos[..., :match_in_nd].to(torch.float32),
            non_batched_pred_boxes_pos[..., :match_in_nd].to(torch.float32),
        )
        dist_matrix = dist_matrix.cpu().numpy()

        sortind = (
            torch.argsort(non_batched_pred_confidence, descending=True).cpu().numpy()
        )
    else:
        # numpy:
        dist_matrix = distance_matrix(
            non_batched_gt_boxes_pos.astype(np.float32),
            non_batched_pred_boxes_pos.astype(np.float32),
        )
        sortind = np.argsort(non_batched_pred_confidence)[::-1]

    matched_preds_mask = np.zeros(n_pred, dtype=bool)
    det_gts_mask = np.zeros(n_true, dtype=bool)
    idxs_into_gt = []
    idxs_into_preds = []
    matching_dists = []
    for pred_idx in sortind:
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx in range(n_true):
            # Find closest match among ground truth boxes
            if gt_idx not in idxs_into_gt:
                this_distance = dist_matrix[gt_idx, pred_idx]
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

            # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < matching_threshold
        if is_match:
            idxs_into_gt.append(match_gt_idx)
            idxs_into_preds.append(pred_idx)
            matched_preds_mask[pred_idx] = True
            det_gts_mask[match_gt_idx] = True
            matching_dists.append(min_dist)

    idxs_into_gt = np.array(idxs_into_gt, dtype=np.int64)
    idxs_into_preds = np.array(idxs_into_preds, dtype=np.int64)
    matching_dists = np.array(matching_dists)

    return (
        idxs_into_gt,
        idxs_into_preds,
        matching_dists,
        matched_preds_mask,
        det_gts_mask,
    )
