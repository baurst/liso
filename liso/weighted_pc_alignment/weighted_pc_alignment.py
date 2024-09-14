#!/usr/bin/env python3

import numpy as np
import torch
from torch_symm_ortho import symmetric_orthogonalization

EPSILON = 1e-7


def print_stats(name, array):
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    if array.size <= 100:
        print(name + ":", array)
        return
    nan_mask = np.isnan(array)
    inf_mask = array == np.inf
    neg_inf_mask = array == -np.inf
    mask = nan_mask | inf_mask | neg_inf_mask
    shape = array.shape
    array = array[~mask]
    if mask.sum() > 0:
        maxi = np.max(array)
        mini = np.min(array)
        mean = np.mean(array)
        std = np.std(array)
    else:
        maxi, mini, mean, std = np.nan, np.nan, np.nan, np.nan
    count_neg = (array < 0.0).sum()
    count_pos = (array > 0.0).sum()
    count_zero = (array == 0.0).sum()
    count_nan = nan_mask.sum()
    count_inf = inf_mask.sum()
    count_neg_inf = neg_inf_mask.sum()
    print(
        "%s%s: %.1e <= %.1e +- %.1e <= %.1e     (<0: %d, =0: %d, >0: %d, NaN: %d, inf: %d, -inf: %d)"
        % (
            name,
            str(shape),
            mini,
            mean,
            std,
            maxi,
            count_neg,
            count_zero,
            count_pos,
            count_nan,
            count_inf,
            count_neg_inf,
        )
    )


def weighted_pc_alignment(
    cloud_t0: torch.Tensor,
    cloud_t1: torch.Tensor,
    weights: torch.Tensor,
    use_epsilon_on_weights=False,
):
    dims = 3
    assert cloud_t0.shape[1:] == (dims,), (cloud_t0.shape, dims)
    assert cloud_t1.shape[1:] == (dims,), (cloud_t1.shape, dims)
    assert len(weights.shape) == 1

    assert (weights >= 0.0).all(), (
        print_stats("weights", weights),
        "negative weights found",
    )

    if use_epsilon_on_weights:
        weights = weights + EPSILON
        count_nonzero_weighted_points = (weights > 0).sum()
        not_enough_points = count_nonzero_weighted_points < 3
    else:
        count_nonzero_weighted_points = (weights > 0).sum()
        not_enough_points = count_nonzero_weighted_points < 3
        if not_enough_points:
            weights = weights + EPSILON

    # m, n = cloud_t0.shape  # m = dims, n num points
    cum_wts = weights.sum(dim=-1)

    X_wtd = cloud_t0 * weights[..., None]
    Y_wtd = cloud_t1 * weights[..., None]

    mx_wtd = X_wtd.sum(dim=0) / cum_wts
    my_wtd = Y_wtd.sum(dim=0) / cum_wts
    Xc = cloud_t0 - mx_wtd[None, :]
    Yc = cloud_t1 - my_wtd[None, :]

    # sx = np.mean(np.sum(Xc * Yc, 0))

    # Sxy_wtd = (
    #     tf.einsum("bnc,bnd->bcd", Yc * weights[..., None], Xc) / cum_wts[:, None, None]
    # )
    Sxy_wtd = (Yc * weights[..., None]).T @ Xc / cum_wts

    try:
        R = symmetric_orthogonalization(Sxy_wtd.to(torch.double))
        mask = weights > 0
    except AssertionError:
        U, S, Vh = torch.linalg.svd(Sxy_wtd.to(torch.double))
        print("Sxy_wtd", Sxy_wtd)
        if (weights > 0).sum() <= 10:
            mask = weights > 0
            print("masked weights", weights[mask])
            print("masked cloud_t0", cloud_t0[mask])
            print("masked cloud_t1", cloud_t1[mask])
            print("masked Xc", Xc[mask])
            print("masked Yc", Yc[mask])
        print_stats("weights", weights)
        print_stats("cloud_t0", cloud_t0)
        print_stats("cloud_t1", cloud_t1)
        print("S", S)
        print("R", U @ Vh)
        raise
    except RuntimeError:
        print("Sxy_wtd", Sxy_wtd)
        print_stats("weights", weights)
        print_stats("cloud_t0", cloud_t0)
        print_stats("cloud_t1", cloud_t1)
        raise
    t = my_wtd.to(torch.double) - R @ mx_wtd.to(torch.double)

    R = torch.cat([R, torch.zeros((1, 3), dtype=R.dtype, device=R.device)], dim=0)
    t = torch.cat([t, torch.ones((1,), dtype=t.dtype, device=t.device)], dim=-1)
    T = torch.cat([R, t[:, None]], dim=-1)

    assert T.dtype == torch.double

    # if T.requires_grad:
    #     from main_utils import backward_anomaly_detection_hook
    #
    #     T.register_hook(backward_anomaly_detection_hook("wpa_T_grad"))
    #     Sxy_wtd.register_hook(backward_anomaly_detection_hook("wpa_Sxy_wtd_grad"))
    #
    #     if weights.requires_grad:
    #         weights.register_hook(backward_anomaly_detection_hook("wpa_weights_grad"))
    #         cum_wts.register_hook(backward_anomaly_detection_hook("wpa_cum_wts_grad"))

    return T, not_enough_points  # R, t
