from typing import Dict

import torch
from liso.losses.centerpoint_loss import prob_heatmap_loss


def compute_transfusion_heatmap_loss(
    *,
    loss_cfg: Dict,
    box_vars: Dict,
    gt_maps: Dict,
    gt_center_mask: torch.BoolTensor,
    ignore_region_is_true_mask: torch.BoolTensor,
):
    groundtruth_probs = gt_maps["probs"]
    pred_logits = box_vars["probs"]
    confidence_loss = prob_heatmap_loss(
        loss_cfg,
        gt_center_mask,
        groundtruth_probs,
        pred_logits,
        ignore_where_true_mask=ignore_region_is_true_mask,
    )
    return {"loss/supervised/probs_heatmap": confidence_loss}
