from typing import Set

import torch
from liso.kabsch.box_groundtruth_matching import batched_match_bboxes
from liso.kabsch.shape_utils import Shape
from liso.losses.centerpoint_loss import compute_focal_loss


def hungarian_matching_loss(
    groundtruth_bboxes: Shape,
    predicted_bboxes: Shape,
    writer=None,
    global_step: int = 0,
    train_box_attrs: Set = None,
    prob_loss="focal",
    loss_extra_descr="supervised",
):
    if train_box_attrs is None:
        train_box_attrs = {"probs", "rot", "dims", "pos"}
    MAX_DIST = 1e5
    DIST_MATCHING_THRESHOLD = 2.0

    (idxs_gt, idxs_pred, _, _, _) = batched_match_bboxes(
        groundtruth_bboxes, predicted_bboxes, MAX_DIST, DIST_MATCHING_THRESHOLD
    )

    losses_dict = {}

    if (
        torch.count_nonzero(groundtruth_bboxes.valid) > 1
        and torch.count_nonzero(predicted_bboxes.valid) > 1
        and idxs_gt.size != 0
        and idxs_pred.size != 0
    ):
        for box_attr_name in train_box_attrs:
            if box_attr_name == "probs":
                if prob_loss == "bce":
                    raise AssertionError("cant handle ignore regions")
                elif prob_loss == "focal":
                    pred_logits = predicted_bboxes.probs
                    target_probs = torch.zeros_like(pred_logits)
                    is_matched_pred_box = torch.zeros_like(
                        pred_logits[..., 0], dtype=torch.bool
                    )
                    is_matched_pred_box[idxs_pred[..., 0], idxs_pred[..., 1]] = True
                    target_probs[is_matched_pred_box] = 1.0

                    attr_loss = compute_focal_loss(
                        is_matched_pred_box,
                        target_probs,
                        pred_logits,
                        gamma=2.0,
                        alpha=0.5,
                        ignore_where_true_mask=~predicted_bboxes.valid,
                    )
                else:
                    raise NotImplementedError(prob_loss)
            else:
                assert predicted_bboxes.valid[
                    idxs_pred[..., 0], idxs_pred[..., 1]
                ].all(), "training on invalid pred boxes - ignore region must have been handled badly?"
                target = groundtruth_bboxes.__dict__[box_attr_name][
                    idxs_gt[..., 0], idxs_gt[..., 1]
                ]
                attr_loss = torch.nn.functional.l1_loss(
                    target=target,
                    input=predicted_bboxes.__dict__[box_attr_name][
                        idxs_pred[..., 0], idxs_pred[..., 1]
                    ],
                )
            losses_dict[
                f"loss/{loss_extra_descr}/hungarian/{box_attr_name}"
            ] = attr_loss
        if writer:
            writer.add_scalar(
                f"metrics/{loss_extra_descr}/hungarian/mean_matched_box_prob",
                predicted_bboxes.probs[idxs_pred[..., 0], idxs_pred[..., 1], 0].mean(),
                global_step=global_step,
            )
            writer.add_scalar(
                f"metrics/{loss_extra_descr}/hungarian/min_matched_box_prob",
                predicted_bboxes.probs[idxs_pred[..., 0], idxs_pred[..., 1], 0].min(),
                global_step=global_step,
            )

    return losses_dict
