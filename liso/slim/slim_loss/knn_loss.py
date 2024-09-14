#!/usr/bin/env python3
from typing import Dict

import torch
from liso.slim.slim_loss.knn_wrapper import NearestPointLoss, compute_flow_loss_a_to_b
from munch import Munch


def compute_knn_loss_components(
    pcl_t0: torch.FloatTensor,
    valid_mask_t0: torch.BoolTensor,
    pcl_t1: torch.FloatTensor,
    valid_mask_t1: torch.BoolTensor,
    *,
    prediction: Dict[str, torch.Tensor],
    loss_cfg: Dict[str, torch.Tensor],
    model_cfg: Dict[str, torch.Tensor],
    bev_extent: Dict[str, torch.Tensor],
):
    assert pcl_t0.ndim == 3
    assert pcl_t0.size(2) == 3
    assert pcl_t1.ndim == 3
    assert pcl_t1.size(2) == 3
    assert prediction.static_flow.ndim == 3
    assert prediction.static_flow.size(2) == 3

    eval_flow_types = {"aggregated"}
    if loss_cfg.artificial_labels.cross_entropy_penalty > 0.0:
        eval_flow_types.add("dynamic")
        if loss_cfg.artificial_labels.use_static_aggr_flow:
            eval_flow_types.add("static_aggr")
        else:
            eval_flow_types.add("static")
    if loss_cfg.knn_on_dynamic_penalty != 0.0:
        assert loss_cfg.knn_on_dynamic_penalty > 0.0
        eval_flow_types.add("dynamic")
    if loss_cfg.knn_on_static_penalty != 0.0:
        assert loss_cfg.knn_on_static_penalty > 0.0
        if model_cfg.use_static_aggr_flow_for_aggr_flow:
            eval_flow_types.add("static_aggr")
        else:
            eval_flow_types.add("static")
    pcl_t0[~valid_mask_t0] = float("nan")
    pcl_t1[~valid_mask_t1] = float("nan")
    eval_flow_types = list(eval_flow_types)
    pcl_t0s = [pcl_t0] * len(eval_flow_types)
    pcl_t1s = [pcl_t1] * len(eval_flow_types)
    flows = []
    for flow_type in eval_flow_types:
        flow_pred = prediction["%s_flow" % flow_type]
        flow_pred[~valid_mask_t0] = float("nan")
        flows.append(flow_pred)

    bs = flows[0].size(0)
    flows, pcl_t0s, pcl_t1s = [
        torch.cat(el, dim=0)
        for el in [
            flows,
            pcl_t0s,
            pcl_t1s,
        ]
    ]
    flow_loss_per_type, knn_results_per_type = compute_flow_loss_a_to_b(
        pcl_t0s,
        pcl_t1s,
        flows,
        loss_function=NearestPointLoss(bev_extent=bev_extent, **loss_cfg.knn_loss),
        nearest_dist_mode=loss_cfg.knn_dist_measure,
    )

    knn_results = {}
    for i, flow_type in enumerate(eval_flow_types):
        knn_results[flow_type] = {
            "loss": flow_loss_per_type[i * bs : (i + 1) * bs, ...],
            "knn": Munch(
                **{
                    k: v[i * bs : (i + 1) * bs, ...]
                    for k, v in knn_results_per_type.items()
                }
            ),
        }
    return knn_results
