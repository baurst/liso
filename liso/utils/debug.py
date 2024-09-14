import numpy as np
import torch


def print_stats(name, array):
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    if array.size <= 100:
        print(name + ":", array)
    nan_mask = np.isnan(array)
    inf_mask = array == np.inf
    neg_inf_mask = array == -np.inf
    mask = nan_mask | inf_mask | neg_inf_mask
    shape = array.shape
    array = array[~mask]
    if array.size > 0:
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
        "%s%s: %s <= %s +- %s <= %s     (<0: %s, =0: %s, >0: %s, NaN: %s, inf: %s, -inf: %s)"
        % (
            "name",
            "(shape)",
            "mini",
            "mean",
            "std",
            "maxi",
            "count_neg",
            "count_zero",
            "count_pos",
            "count_nan",
            "count_inf",
            "count_neg_inf",
        )
    )
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
