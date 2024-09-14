from typing import Dict

import torch


def castf(tensor):
    if tensor.dtype not in {torch.float, torch.double}:
        tensor = tensor.to(torch.float)
    return tensor


def constant_labels(
    *,
    static_knn_results,
    dynamic_knn_results,
    knn_dist_sqr_key: str,
):

    is_static_artificial_label = castf(
        static_knn_results[knn_dist_sqr_key] <= dynamic_knn_results[knn_dist_sqr_key]
    )
    artificial_label_weights = torch.ones_like(is_static_artificial_label)

    return (
        is_static_artificial_label,
        artificial_label_weights,
    )


def compute_artificial_label_loss(
    *,
    # el,
    prediction: Dict[str, torch.Tensor],
    # mask_fw: torch.Tensor,
    # mask_bw: torch.Tensor,
    knn_results: Dict[str, torch.Tensor],
    # final_scale: int,
    loss_cfg: Dict[str, torch.Tensor],
    # summaries: Dict,
):

    assert loss_cfg.artificial_labels.knn_mode == "point"
    knn_dist_sqr_key = "nearest_dist_sqr"

    assert loss_cfg.artificial_labels.weight_mode in {
        "constant",
    }, "only constant mode allowed - this is not supposed to happen"
    static_flow_key = (
        "static_aggr" if loss_cfg.artificial_labels.use_static_aggr_flow else "static"
    )
    is_static_artificial_label, artificial_label_weights = constant_labels(
        static_knn_results=knn_results[static_flow_key]["knn"],
        dynamic_knn_results=knn_results["dynamic"]["knn"],
        knn_dist_sqr_key=knn_dist_sqr_key,
    )

    ce_loss = (
        torch.nn.BCELoss(reduction="none")(
            prediction["staticness"], is_static_artificial_label
        )
        * artificial_label_weights.detach()
    )
    assert ce_loss.ndim == 2, ce_loss.shape  # [B, N]
    return ce_loss
