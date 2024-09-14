from typing import Dict

import torch
import torch.nn.functional as F


def to_positive_angle(angle_rad: torch.FloatTensor) -> torch.FloatTensor:
    angle_rad = angle_rad % (2 * torch.pi)
    angle_rad = torch.where(angle_rad < 0, angle_rad + 2 * torch.pi, angle_rad)
    return angle_rad


def centerpoint_loss(
    *,
    loss_cfg: Dict,
    decoded_pred_box_maps: Dict,
    raw_activated_pred_box_maps: Dict,
    gt_maps: Dict,
    gt_center_mask: torch.BoolTensor,
    rotation_loss_weights_map: torch.FloatTensor,
    box_prediction_cfg: Dict,
    ignore_region_is_true_mask: torch.BoolTensor,
):
    losses = {}
    num_pos = torch.clip(gt_center_mask.sum(), min=1.0)
    if "probs" in gt_maps:
        groundtruth_probs = gt_maps["probs"]
        pred_logits_bev = raw_activated_pred_box_maps["probs"]
        confidence_loss = prob_heatmap_loss(
            loss_cfg,
            gt_center_mask,
            groundtruth_probs,
            pred_logits_bev,
            ignore_region_is_true_mask,
        )
        losses["loss/supervised/centermaps/probs"] = confidence_loss
    if "rot" in gt_maps and gt_center_mask.sum() > 0:
        rot_loss_weights = torch.maximum(
            rotation_loss_weights_map[gt_center_mask & ~ignore_region_is_true_mask],
            torch.tensor(0.1, device=gt_center_mask.device),
        )
        rot_loss_weights = rot_loss_weights / torch.maximum(
            torch.sum(rot_loss_weights),
            torch.tensor(1.0, device=rot_loss_weights.device),
        )
        rot_pred = raw_activated_pred_box_maps["rot"][
            gt_center_mask & ~ignore_region_is_true_mask
        ]
        rot_target = gt_maps["rot"][gt_center_mask & ~ignore_region_is_true_mask]
        if box_prediction_cfg.rotation_representation.method in ("direct", "vector"):
            assert rot_target.shape[-1] == rot_pred.shape[-1], (
                rot_target.shape,
                rot_pred.shape,
            )
            rot_loss = torch.sum(
                torch.nn.functional.l1_loss(
                    input=rot_pred,
                    target=rot_target,
                    reduction="none",
                )
                * rot_loss_weights,
            )
        elif box_prediction_cfg.rotation_representation.method == "class_bins":
            assert (
                ignore_region_is_true_mask is None
            ), "ignore region not implemented here"
            num_rot_bins = 36
            assert rot_target.shape[-1] == 1, rot_target.shape
            assert rot_pred.shape[-1] == num_rot_bins, rot_pred.shape
            bin_size = 2 * torch.pi / num_rot_bins
            rot_target_rad_positive = to_positive_angle(
                torch.squeeze(rot_target, dim=-1)
            )
            rot_target_bin = torch.floor(rot_target_rad_positive / bin_size).long()
            right_neighbor_target_bin = (rot_target_bin + 1) % num_rot_bins
            left_neighbor_target_bin = (rot_target_bin - 1) % num_rot_bins
            left_neighbor_target_bin = torch.where(
                left_neighbor_target_bin == -1,
                left_neighbor_target_bin + num_rot_bins,
                left_neighbor_target_bin,
            )
            rot_target_distrib = (
                (
                    0.6
                    * torch.nn.functional.one_hot(rot_target_bin, num_rot_bins).float()
                )
                + 0.2
                * torch.nn.functional.one_hot(
                    left_neighbor_target_bin, num_rot_bins
                ).float()
                + 0.2
                * torch.nn.functional.one_hot(
                    right_neighbor_target_bin, num_rot_bins
                ).float()
            )
            # rot_target_distrib[:, right_neighbor_target_bin] = 0.2
            # rot_target_distrib[:, left_neighbor_target_bin] = 0.2

            rot_loss = torch.sum(
                rot_loss_weights
                * torch.nn.functional.cross_entropy(
                    rot_pred, rot_target_distrib, reduction="none"
                )
            )

        losses["loss/supervised/centermaps/rot"] = 10 * rot_loss

    if "dims" in gt_maps and (gt_center_mask & ~ignore_region_is_true_mask).sum() > 0:
        gt_dims = gt_maps["dims"][gt_center_mask & ~ignore_region_is_true_mask]
        pred_dims = decoded_pred_box_maps["dims"][
            gt_center_mask & ~ignore_region_is_true_mask
        ]
        dim_loss = (
            torch.nn.functional.l1_loss(
                input=pred_dims,
                target=gt_dims,
            ).sum()
            / num_pos
        )
        losses["loss/supervised/centermaps/dims"] = dim_loss

    if "pos" in gt_maps and (gt_center_mask & ~ignore_region_is_true_mask).sum() > 0:
        gt_pos = gt_maps["pos"][gt_center_mask & ~ignore_region_is_true_mask]
        pred_pos = decoded_pred_box_maps["pos"][
            gt_center_mask & ~ignore_region_is_true_mask
        ]
        position_loss = (
            torch.nn.functional.l1_loss(
                input=pred_pos,
                target=gt_pos,
            ).sum()
            / num_pos
        )
        losses["loss/supervised/centermaps/pos"] = position_loss

    return losses


def prob_heatmap_loss(
    loss_cfg,
    gt_center_mask,
    groundtruth_probs,
    pred_logits,
    ignore_where_true_mask=None,
):
    if loss_cfg.supervised.centermaps.confidence_target in ("gaussian",):
        gamma = 2.0
        alpha = 0.5
        focal_loss = compute_focal_loss(
            gt_center_mask,
            groundtruth_probs,
            pred_logits,
            gamma,
            alpha,
            ignore_where_true_mask,
        )
        confidence_loss = focal_loss
    else:
        raise NotImplementedError(loss_cfg.supervised.centermaps.confidence_target)
    return confidence_loss


def compute_focal_loss(
    gt_center_mask: torch.BoolTensor,
    groundtruth_probs: torch.FloatTensor,
    pred_logits: torch.FloatTensor,
    gamma: float,
    alpha: float,
    ignore_where_true_mask: torch.BoolTensor = None,
):
    num_pos = torch.clip(gt_center_mask.sum(), min=1.0)
    probs_pos = torch.sigmoid(pred_logits)
    probs_neg = torch.sigmoid(-pred_logits)
    positive_loss = alpha * torch.pow(probs_neg, gamma) * F.logsigmoid(pred_logits)
    negative_loss = (
        (1 - alpha)
        * torch.pow(probs_pos, gamma)
        * torch.pow(
            1.0 - groundtruth_probs,
            4.0,  # this is beta=4.0 in the formula in the paper
        )
        * F.logsigmoid(-pred_logits)
    )
    if ignore_where_true_mask is None:
        focal_loss = (
            -(
                positive_loss[gt_center_mask].sum()
                + negative_loss[~gt_center_mask].sum()
            )
            / num_pos
        )
    else:
        focal_loss = (
            -(
                positive_loss[gt_center_mask & ~ignore_where_true_mask].sum()
                + negative_loss[~gt_center_mask & ~ignore_where_true_mask].sum()
            )
            / num_pos
        )
    return focal_loss
