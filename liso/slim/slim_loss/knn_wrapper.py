#!/usr/bin/env python3

import functools as ft
import typing as t

import torch
from liso.slim.slim_loss.knn_graph import knn_graph
from munch import Munch


def huber_delta(
    *,
    err: torch.FloatTensor = None,
    err_sqr: torch.FloatTensor = None,
    delta: float,
    mode: str = "large_grad_1"
):
    assert mode in {"large_grad_1", "small_err_sqr"}

    if delta == 0.0:
        assert mode == "large_grad_1"
        if err is None:
            assert err_sqr is not None
            nonzero_mask_gradient_safe = ~(err_sqr == 0.0)
            return torch.where(
                nonzero_mask_gradient_safe, err_sqr, torch.ones_like(err_sqr)
            ).sqrt() * nonzero_mask_gradient_safe.to(torch.float)
        else:
            raise NotImplementedError()
    assert delta > 0.0
    if err is None:
        assert err_sqr is not None
    else:
        assert err_sqr is None
        err_sqr = err.square()

    if mode == "large_grad_1":
        delta_tensor = (
            torch.clamp(err_sqr, max=delta**2) / (2.0 * delta)
            + torch.clamp(err_sqr, min=delta**2).sqrt()
            - delta
        )
    elif mode == "small_err_sqr":
        delta_tensor = (
            torch.clamp(err_sqr, max=delta**2)
            + torch.clamp(err_sqr, min=delta**2).sqrt() * (2 * delta)
            - 2 * delta**2
        )
    else:
        raise ValueError("Unknown huber mode %s" % mode)
    return delta_tensor


def squared_sum(delta: torch.FloatTensor, dim: int = -1):
    return delta.square().sum(dim=dim)


class NearestPointLoss:
    def __init__(
        self,
        *args,
        bev_extent: t.Tuple[float, float, float, float],
        L1_delta: float,
        drop_outliers__perc: float,
        fov_mode: str = "ignore_out_fov",
        **kwargs
    ):
        super().__init__()
        assert 0.0 <= drop_outliers__perc < 100.0
        assert fov_mode in {
            "none",
            "ignore_out_fov",
            "use_nearest",
            "mask_close_fov",
        }

        self.bev_extent = bev_extent
        self.drop_outliers__perc = drop_outliers__perc
        self.huber_loss = ft.partial(huber_delta, delta=L1_delta, mode="large_grad_1")
        self.fov_mode = fov_mode

    def __call__(self, *, cloud_b__a, nearest_cloud_b__a, nearest_dist_sqr_b__a):
        fov_dist_minx_cloud_b__a = cloud_b__a[..., 0] - self.bev_extent[0]
        fov_dist_miny_cloud_b__a = cloud_b__a[..., 1] - self.bev_extent[1]
        fov_dist_maxx_cloud_b__a = self.bev_extent[2] - cloud_b__a[..., 0]
        fov_dist_maxy_cloud_b__a = self.bev_extent[3] - cloud_b__a[..., 1]
        min_fov_dist_cloud_b__a = torch.min(
            torch.stack(
                [
                    fov_dist_minx_cloud_b__a,
                    fov_dist_miny_cloud_b__a,
                    fov_dist_maxx_cloud_b__a,
                    fov_dist_maxy_cloud_b__a,
                ],
                dim=-1,
            ),
            dim=-1,
        )[0]

        if self.fov_mode == "ignore_out_fov":
            weights__a = (min_fov_dist_cloud_b__a > 0.0).to(torch.float)
        elif self.fov_mode == "use_nearest":
            nearest_dist_sqr_b__a = torch.min(
                nearest_dist_sqr_b__a, min_fov_dist_cloud_b__a.square()
            )
        elif self.fov_mode == "mask_close_fov":
            weights__a = (min_fov_dist_cloud_b__a > 0.0).to(torch.float) * (
                nearest_dist_sqr_b__a < min_fov_dist_cloud_b__a.square()
            )
        elif self.fov_mode == "none":
            pass
        else:
            raise ValueError("Unknown fov_mode: %s" % self.fov_mode)

        loss = self.huber_loss(err_sqr=nearest_dist_sqr_b__a)

        if self.fov_mode != "none":
            loss = loss * weights__a

        if self.drop_outliers__perc > 0.0:
            keep_quantile = 1.0 - self.drop_outliers__perc / 100.0
            bs = loss.size(0)
            num_elems_per_batch = loss.numel() / bs
            kth = int(round(num_elems_per_batch * keep_quantile))
            loss_threshold = torch.stack(
                [torch.kthvalue(loss[b], kth)[0] for b in range(bs)], dim=0
            )
            loss = torch.where(
                loss
                <= loss_threshold[(slice(None),) + tuple([None] * (loss.ndim - 1))],
                loss,
                torch.zeros_like(loss),
            )

        return loss


@torch.no_grad()
def get_idx_dists_for_knn(
    ref_pts: torch.FloatTensor, query_pts: torch.FloatTensor, num_neighbors: int = 1
):
    assert ref_pts.ndim == 2
    assert query_pts.ndim == 2

    indices = knn_graph(
        query_pts,
        index=ref_pts,
        k=num_neighbors,
        loop=True,
    )

    return indices


def compute_flow_loss_a_to_b(
    cloud_a: torch.FloatTensor,
    cloud_b: torch.FloatTensor,
    flow_a_to_b: torch.FloatTensor,
    loss_function,
    nearest_dist_mode: str = "point",
):
    # #region check shapes and dtypes
    assert nearest_dist_mode in {"point", "plane"}
    assert cloud_a.ndim == 3
    assert cloud_b.ndim == 3
    assert flow_a_to_b.ndim == 3
    assert cloud_a.shape[-1] == 3
    assert cloud_b.shape[-1] == 3
    assert flow_a_to_b.shape[-1] == 3
    # #endregion check shapes and dtypes

    # torch only supports point for now
    assert nearest_dist_mode == "point"

    # notation: X_b__a, where X is name/meaning, b denotes the time idx for the values, but a denotes the set of support points
    # especially: X_b__a has as many entries as cloud_a, not cloud_b

    cloud_b__a = cloud_a + flow_a_to_b
    bs = cloud_b.size(0)
    indices_into_b__a = torch.stack(
        [
            get_idx_dists_for_knn(cloud_b[b], cloud_b__a[b], num_neighbors=1)
            for b in range(bs)
        ],
        dim=0,
    )

    nearest_cloud_b__a = torch.gather(cloud_b, 1, indices_into_b__a.repeat(1, 1, 3))

    nearest_dist_sqr_b__a = squared_sum(nearest_cloud_b__a - cloud_b__a, dim=-1)

    if nearest_dist_mode == "point":
        selected_nearest_cloud_b__a = {
            "points": nearest_cloud_b__a,
            "dists_sqr": nearest_dist_sqr_b__a,
        }
    else:
        raise DeprecationWarning()
        # assert nearest_dist_mode == "plane"
        # selected_nearest_cloud_b__a = {
        #     "points": plumb_line_point_cloud_b__a,
        #     "dists_sqr": plumb_line_point_dist_sqr_b__a,
        # }

    loss = loss_function(
        cloud_b__a=cloud_b__a,
        nearest_cloud_b__a=selected_nearest_cloud_b__a["points"],
        nearest_dist_sqr_b__a=selected_nearest_cloud_b__a["dists_sqr"],
    )

    return (
        loss,
        Munch(
            nearest_dist_sqr=nearest_dist_sqr_b__a,
            nearest_dist=nearest_dist_sqr_b__a.sqrt(),
        ),
    )
