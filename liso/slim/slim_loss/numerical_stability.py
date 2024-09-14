#!/usr/bin/env python3

import torch
from liso.utils.debug import print_stats


def numerically_stable_lin_comb_exps(
    *, exps, weights, mask=None, dim: int = -1, keepdims: bool = False
):
    if mask is not None:
        min_exp = torch.min(exps, dim=dim, keepdims=True)[0]
        exps = torch.where(mask, exps, min_exp)
        weights = weights * mask

    max_exp = torch.max(exps, dim=dim, keepdims=True)[0]

    if keepdims:
        max_exp_kd = max_exp
    else:
        max_exp_kd = torch.squeeze(max_exp, dim=dim)

    return max_exp_kd, (torch.exp(exps - max_exp) * weights).sum(
        dim=dim, keepdims=keepdims
    )


def normalized_sigmoid_sum(logits, mask=None):
    # sigmoid(x) = exp(-relu(-x)) * sigmoid(abs(x))
    neg_logit_part = -torch.relu(-logits)
    weights = torch.sigmoid(torch.abs(logits))

    denom_exp, denom_lin_comb = numerically_stable_lin_comb_exps(
        exps=neg_logit_part, weights=weights, mask=mask, keepdims=True
    )

    if mask is not None:
        weights = weights * mask
        all_masked = ~mask.any(dim=-1, keepdims=True)
        denom_lin_comb = torch.where(
            all_masked, torch.ones_like(denom_lin_comb), denom_lin_comb
        )
        neg_logit_part = torch.where(mask, neg_logit_part, denom_exp)

    assert (denom_lin_comb >= 0.5).all(), print_stats("denom_lin_comb", denom_lin_comb)

    exp_part = torch.exp(neg_logit_part - denom_exp)
    assert (exp_part <= 1.0).all(), print_stats("exp_part", exp_part)
    assert (weights[mask] >= 0.5).all(), print_stats("weights[mask]", weights[mask])

    result = exp_part * weights / denom_lin_comb
    assert logits.shape == result.shape, (logits.shape, result.shape)

    return result
