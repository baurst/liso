from typing import Tuple

import numpy as np
import torch
from liso.kabsch.mask_fusing import fuse_masks_screen
from liso.kabsch.shape_utils import Shape
from liso.torch_symm_ortho import symmetric_orthogonalization
from liso.transformations.transformations import compose_matrix
from liso.utils.bev_utils import get_bev_setup_params, get_metric_voxel_center_coords
from liso.utils.torch_transformation import (
    homogenize_pcl,
    numpy_compose_matrix,
    torch_compose_matrix,
)


def map_nan_padding_to_zeros(batched_points, valid_mask):
    batched_points[~valid_mask] = 0.0
    return batched_points


def sigmoid(x, slope=1.0):
    return 1 / (1 + np.exp(-slope * x))


def cauchy(logits: torch.FloatTensor) -> torch.FloatTensor:
    retval = 0.5 + 1 / np.pi * torch.atan(logits)
    return retval


def get_mask_softness_fun(softness_fun):
    return {"cauchy": cauchy, "sigmoid": torch.sigmoid}[softness_fun]


def is_point_in_box_array(pcl_in_obj_coords, length_x, width_y, height_z):
    point_is_in_box_length_mask = np.logical_and(
        -length_x / 2.0 < pcl_in_obj_coords[..., 0],
        pcl_in_obj_coords[..., 0] < length_x / 2.0,
    )
    point_is_in_box_width_mask = np.logical_and(
        -width_y / 2.0 < pcl_in_obj_coords[..., 1],
        pcl_in_obj_coords[..., 1] < width_y / 2.0,
    )
    point_is_in_box_height_mask = np.logical_and(
        -height_z / 2.0 < pcl_in_obj_coords[..., 2],
        pcl_in_obj_coords[..., 2] < height_z / 2.0,
    )

    point_is_in_box_mask = np.logical_and(
        np.logical_and(point_is_in_box_length_mask, point_is_in_box_width_mask),
        point_is_in_box_height_mask,
    )
    return point_is_in_box_mask


def batched_multivariate_gaussian(pos, mu, sigma, normalize=True):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[-1]
    sigma_determinant = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    if normalize:
        N = (np.sqrt((2 * np.pi) ** n * sigma_determinant))[:, :, None, None]
    else:
        N = 1.0

    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum(
        "...k,...kl,...l->...",
        pos - mu[:, :, None, None, :],
        sigma_inv[:, :, None, None, :, :],
        pos - mu[:, :, None, None, :],
    )

    return np.exp(-fac / 2) / N


def batched_render_gaussian_kabsch_mask(
    *,
    box_x,
    box_y,
    box_len,
    box_w,
    box_theta,
    bev_range_x,
    bev_range_y,
    img_shape,
    normalize_gaussian=True,
):
    pos = get_metric_voxel_center_coords(
        bev_range_x=bev_range_x, bev_range_y=bev_range_y, dataset_img_shape=img_shape
    )[..., 0:2]
    mu = np.stack([box_x, box_y], axis=-1)
    # 95% of points should lie in mask
    cov_top_row = np.stack([box_len, np.zeros_like(box_len)], axis=-1)
    cov_bottom_row = np.stack([np.zeros_like(box_w), box_w], axis=-1)
    # see CornerNet: whose center is at the positive location and whose σ is 1/3 of the radius
    # radius -> halve the length/width
    cov = 0.3 * 0.5 * np.stack([cov_top_row, cov_bottom_row], axis=-2)
    rot_mat = numpy_compose_matrix(t_x=box_x, t_y=box_y, theta_z=box_theta)[
        ..., 0:2, 0:2
    ]
    cov = np.einsum(
        "...ij, ...jk", np.einsum("...ij, ...jk", rot_mat, cov), np.linalg.inv(rot_mat)
    )
    # cov = cov @ rot_mat.T

    weight = batched_multivariate_gaussian(
        pos[None, None, ...], mu, cov, normalize=normalize_gaussian
    )
    if not normalize_gaussian:
        # TODO: this seems like a bug: we normalize if normalize_gaussian == False
        weight_max = weight.max(axis=(-1, -2), keepdims=True)
        weight = weight / np.maximum(weight_max, 1e-5)
    return weight


def render_hard_kabsch_mask(
    *,
    box_x,
    box_y,
    box_len,
    box_w,
    box_theta,
    bev_range_x,
    bev_range_y,
    img_shape,
):
    pcl_s = get_metric_voxel_center_coords(
        bev_range_x=bev_range_x, bev_range_y=bev_range_y, dataset_img_shape=img_shape
    )

    s_T_box = compose_matrix(
        translate=np.array([box_x, box_y, 0]), angles=np.array([0.0, 0.0, box_theta])
    )

    pcl_in_obj_coords = np.einsum("ij, knj -> kni", np.linalg.inv(s_T_box), pcl_s)
    weight = 1.0 * is_point_in_box_array(
        pcl_in_obj_coords=pcl_in_obj_coords,
        length_x=box_len,
        width_y=box_w,
        height_z=100.0,
    )

    return weight, pcl_s[..., 0:2]


def render_soft_kabsch_mask_torch(
    *,
    box_x,
    box_y,
    box_z,
    box_len,
    box_w,
    box_h,
    box_theta,
    sigmoid_slope=7.0,
    batched_metric_homog_coords=None,
    softness_fun: torch.sigmoid,
):
    for el in (box_x, box_y, box_len, box_w, box_theta):
        assert len(el.shape) == 2, el.shape  # require batch x slots

    s_T_box = torch_compose_matrix(t_x=box_x, t_y=box_y, t_z=box_z, theta_z=box_theta)
    box_T_s = torch.inverse(s_T_box)

    assert batched_metric_homog_coords.shape[-1] == 4, batched_metric_homog_coords.shape
    assert torch.all(
        (batched_metric_homog_coords[..., -1] == 1.0)
        | (torch.isnan(batched_metric_homog_coords[..., -1]))  # allow nan padded values
    ), "need metric coordinates"
    if len(batched_metric_homog_coords.shape) == 4:
        pcl_in_obj_coords = torch.einsum(
            "bsij, bwhj -> bswhi", box_T_s, batched_metric_homog_coords
        )
    else:
        pcl_in_obj_coords = torch.einsum(
            "bsij, bnj -> bsni", box_T_s, batched_metric_homog_coords
        )

    # add spatial dimss to box_len and box_w:
    weight_probs, weight_logits = get_box_pixel_weights(
        box_len=box_len,
        box_w=box_w,
        box_h=box_h,
        sigmoid_slope=sigmoid_slope,
        softness_fun=softness_fun,
        pcl_in_obj_coords=pcl_in_obj_coords,
    )
    return weight_probs, weight_logits


def get_box_pixel_weights(
    *,
    box_len,
    box_w,
    box_h,
    sigmoid_slope,
    softness_fun,
    pcl_in_obj_coords,
):
    assert box_len.shape == box_w.shape, (box_len.shape, box_w.shape)
    assert box_h.shape == box_w.shape, (box_h.shape, box_w.shape)
    assert box_len.shape == pcl_in_obj_coords.shape[0:2], (
        box_len.shape,
        pcl_in_obj_coords.shape,
    )
    assert pcl_in_obj_coords.shape[-1] == 4, pcl_in_obj_coords.shape
    num_broadcast_dims = len(pcl_in_obj_coords.shape[2:-1])

    logit_x = sigmoid_slope * (
        box_len.reshape(box_len.shape + num_broadcast_dims * (1,)) / 2
        - torch.abs(pcl_in_obj_coords[..., 0])
    )
    logit_y = sigmoid_slope * (
        box_w.reshape(box_w.shape + num_broadcast_dims * (1,)) / 2
        - torch.abs(pcl_in_obj_coords[..., 1])
    )
    logit_z = sigmoid_slope * (
        box_h.reshape(box_h.shape + num_broadcast_dims * (1,)) / 2
        - torch.abs(pcl_in_obj_coords[..., 2])
    )

    weight = softness_fun(logit_x) * softness_fun(logit_y) * softness_fun(logit_z)

    accum_logit = None
    return weight, accum_logit


class KabschDecoder(torch.nn.Module):
    def __init__(self, cfg, img_grid_size=None) -> None:
        super().__init__()
        self.disable_asserts = True
        self.cfg = cfg
        (
            self.bev_range_m_np,
            self.img_grid_size_np,
            self.bev_pixel_per_meter_res_np,
            self.pcl_bev_center_coords_homog_np,
            torch_params,
        ) = get_bev_setup_params(cfg)

        for param_name, param in torch_params.items():
            self.register_parameter(
                param_name,
                torch.nn.Parameter(
                    param,
                    requires_grad=False,
                ),
            )

        self.softness_fun = get_mask_softness_fun(cfg.mask_rendering.softness_fun)

    def forward(
        self,
        shapes: Shape,
        batched_padded_points: torch.FloatTensor = None,
        batched_padded_is_valid_points: torch.BoolTensor = None,
        shape_name=None,
        sigmoid_slope=None,
        obj_dim_scale=1.0,
        softness_func=None,
    ):
        # handle default arguments
        softness_func = (
            softness_func if softness_func is not None else self.softness_fun
        )
        sigmoid_slope = (
            sigmoid_slope
            if sigmoid_slope is not None
            else self.cfg.mask_rendering.pred_sigmoid_slope
        )
        shape_name = shape_name if shape_name is not None else self.cfg.data.shapes.name
        if batched_padded_points is None and batched_padded_is_valid_points is None:
            grid_coords_homog = self.pcl_bev_center_coords_homog[None, ...]
        else:
            # add empty dimension to fake grid
            assert len(batched_padded_points.shape) == 3, batched_padded_points.shape
            pts_no_nan = map_nan_padding_to_zeros(
                batched_padded_points, batched_padded_is_valid_points
            )
            homog_pts = homogenize_pcl(
                pts_no_nan[..., :3], batched_padded_is_valid_points
            )
            assert self.disable_asserts or torch.all(
                torch.isfinite(homog_pts[batched_padded_is_valid_points])
            )
            grid_coords_homog = homog_pts  # [:, None, :, :]
        grid_coords_homog = grid_coords_homog.to(shapes.pos.dtype)
        assert len(shapes.pos.shape) == 3, shapes.pos.shape
        if shape_name == "boxes":
            # we have batches x slots boxes
            (
                per_batch_per_slot_mask_prob,
                per_batch_per_slot_mask_logit,
            ) = render_soft_kabsch_mask_torch(
                box_x=shapes.pos[..., 0],
                box_y=shapes.pos[..., 1],
                box_z=shapes.pos[..., 2],
                box_len=shapes.dims[..., 0] * obj_dim_scale,
                box_w=shapes.dims[..., 1] * obj_dim_scale,
                box_h=shapes.dims[..., 2] * obj_dim_scale,
                box_theta=shapes.rot[..., 0],
                sigmoid_slope=sigmoid_slope,
                softness_fun=softness_func,
                batched_metric_homog_coords=grid_coords_homog,
            )
        else:
            raise NotImplementedError(shape_name)

        if (
            batched_padded_points is not None
            and batched_padded_is_valid_points is not None
        ):
            assert self.disable_asserts or torch.all(
                torch.isfinite(per_batch_per_slot_mask_logit)
            ), torch.where(
                ~torch.isfinite(
                    per_batch_per_slot_mask_logit,
                )
            )
            assert self.disable_asserts or torch.all(
                torch.isfinite(per_batch_per_slot_mask_prob)
            ), torch.where(~torch.isfinite(per_batch_per_slot_mask_prob))
        return per_batch_per_slot_mask_prob, per_batch_per_slot_mask_logit

    def get_kabsch_trafos_from_point_flow(
        self: torch.nn.Module,
        *,
        point_cloud_ta: torch.FloatTensor,
        valid_mask_ta: torch.BoolTensor,
        pointwise_flow_ta_tb: torch.FloatTensor,
        pred_boxes_ta: Shape,
        sigmoid_slope=None,
        obj_dim_scale_buffer=None,
        softness_func=None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        sigmoid_slope = (
            sigmoid_slope
            if sigmoid_slope is not None
            else self.cfg.mask_rendering.pred_sigmoid_slope
        )
        obj_dim_scale_buffer = (
            obj_dim_scale_buffer
            if obj_dim_scale_buffer is not None
            else self.cfg.mask_rendering.obj_dim_scale_buffer
        )
        softness_func = (
            softness_func if softness_func is not None else self.softness_fun
        )

        fg_kabsch_weights, _ = self(
            pred_boxes_ta,
            batched_padded_points=point_cloud_ta,
            batched_padded_is_valid_points=valid_mask_ta,
            obj_dim_scale=(1.0 - obj_dim_scale_buffer),
            softness_func=softness_func,
        )
        assert self.disable_asserts or torch.all(
            torch.isfinite(fg_kabsch_weights),
        ), torch.where(~torch.isfinite(fg_kabsch_weights))
        fg_kabsch_weights_for_bg, _ = self(
            pred_boxes_ta,
            batched_padded_points=point_cloud_ta,
            batched_padded_is_valid_points=valid_mask_ta,
            obj_dim_scale=(1.0 + obj_dim_scale_buffer),
            softness_func=softness_func,
        )
        bg_weights_mask = 1.0 - fuse_masks_screen(
            fg_kabsch_weights_for_bg, dim=1, keepdim=True
        )
        kabsch_weights_for_trafos = torch.cat(
            [fg_kabsch_weights, bg_weights_mask], dim=1
        )
        assert self.disable_asserts or torch.all(
            torch.isfinite(kabsch_weights_for_trafos),
        ), torch.where(~torch.isfinite(kabsch_weights_for_trafos))
        kabsch_trafos, cum_wts = self.per_mask_trafo_from_pointwise_flows(
            point_cloud_ta=point_cloud_ta,
            valid_mask_ta=valid_mask_ta,
            pointwise_flow_ta_tb=pointwise_flow_ta_tb[:, :, 0:2],
            kabsch_weights=kabsch_weights_for_trafos,
        )
        _, num_boxes, _ = pred_boxes_ta.pos.shape
        num_bgs = 1
        fg_kabsch_trafos, bg_kabsch_trafo = torch.split(
            kabsch_trafos, [num_boxes, num_bgs], dim=1
        )
        fg_cum_kabsch_weights, bg_cum_kabsch_weight = torch.split(
            cum_wts, [num_boxes, num_bgs], dim=1
        )
        return (
            fg_kabsch_trafos,
            fg_kabsch_weights,
            fg_cum_kabsch_weights,
            bg_kabsch_trafo,
            bg_cum_kabsch_weight,
        )

    def per_mask_trafo_from_pointwise_flows(
        self,
        point_cloud_ta: torch.FloatTensor,
        valid_mask_ta: torch.BoolTensor,
        pointwise_flow_ta_tb: torch.FloatTensor,
        kabsch_weights: torch.FloatTensor,
    ) -> torch.Tensor:
        assert len(kabsch_weights.shape) == 3, ("expect BxSxN", kabsch_weights.shape)
        assert pointwise_flow_ta_tb.shape[-1] == 2, pointwise_flow_ta_tb.shape
        point_cloud_no_nans_ta = map_nan_padding_to_zeros(
            point_cloud_ta.clone(), valid_mask_ta
        )
        point_cloud_no_nans_ta[..., 2] = 0.0
        pointwise_flow_ta_tb = map_nan_padding_to_zeros(
            pointwise_flow_ta_tb, valid_mask_ta
        )
        flow_3d = torch.cat(
            [pointwise_flow_ta_tb, torch.zeros_like(pointwise_flow_ta_tb[..., :1])],
            dim=-1,
        )

        kabsch_weights = kabsch_weights.swapaxes(1, 2)
        pointwise_flow_ta_tb = map_nan_padding_to_zeros(kabsch_weights, valid_mask_ta)
        kabsch_weights = kabsch_weights.swapaxes(1, 2)
        assert self.disable_asserts or torch.all(
            torch.isfinite(kabsch_weights)
        ), "we cannot handle infinite kabsch weights - map padding nans to zero!"
        assert self.disable_asserts or torch.all(
            torch.logical_and(kabsch_weights >= 0.0, kabsch_weights <= 1.0)
        )

        point_cloud_tb = point_cloud_no_nans_ta + flow_3d

        (
            T,
            cum_wts,
        ) = self.weighted_pc_alignment_for_different_batched_slotted_kabsch_weights(
            point_cloud_ta=point_cloud_no_nans_ta,
            point_cloud_tb=point_cloud_tb,
            batched_slotted_kabsch_weights=kabsch_weights,
        )

        return T, cum_wts

    def weighted_pc_alignment_for_different_batched_slotted_kabsch_weights(
        self,
        *,
        point_cloud_ta: torch.FloatTensor,
        point_cloud_tb: torch.FloatTensor,
        batched_slotted_kabsch_weights: torch.FloatTensor,
    ) -> torch.DoubleTensor:
        eps = 1e-12
        assert (
            len(batched_slotted_kabsch_weights.shape) == 3
        ), batched_slotted_kabsch_weights.shape
        cum_wts = batched_slotted_kabsch_weights.sum(dim=-1)
        if (cum_wts < eps).any():
            # print("Weighted pc alignment epsilon triggered!")
            # print_stats("cum_wts", cum_wts)
            batched_slotted_kabsch_weights = batched_slotted_kabsch_weights.clone()
            batched_slotted_kabsch_weights = torch.where(
                (cum_wts < eps)[..., None],
                torch.tensor(
                    eps,
                    dtype=batched_slotted_kabsch_weights.dtype,
                    device=batched_slotted_kabsch_weights.device,
                ),
                batched_slotted_kabsch_weights,
            )
            cum_wts = batched_slotted_kabsch_weights.sum(dim=-1)
        X_wtd = (
            point_cloud_ta[:, None, :, :] * batched_slotted_kabsch_weights[..., None]
        )
        Y_wtd = (
            point_cloud_tb[:, None, :, :] * batched_slotted_kabsch_weights[..., None]
        )

        mx_wtd = X_wtd.sum(dim=2) / cum_wts[..., None]
        my_wtd = Y_wtd.sum(dim=2) / cum_wts[..., None]
        Xc = point_cloud_ta[:, None, :, :] - mx_wtd[:, :, None, :]
        Yc = point_cloud_tb[:, None, :, :] - my_wtd[:, :, None, :]

        Sxy_wtd = torch.einsum(
            "bsnc,bsnd->bscd", Yc * batched_slotted_kabsch_weights[..., None], Xc
        )

        Sxy_wtd = Sxy_wtd / cum_wts[:, :, None, None]
        if self.cfg.svd_backend == "torch":
            U, S, Vh = torch.linalg.svd(Sxy_wtd.to(torch.double))

            R = torch.einsum("bsoc,bsci->bsoi", U, Vh)
        elif self.cfg.svd_backend == "symm_ortho":
            R = symmetric_orthogonalization(Sxy_wtd.to(torch.double))

        else:
            raise NotImplementedError(f"dont know svd backend {self.cfg.svd_backend}")
        t = my_wtd.to(torch.double) - torch.einsum(
            "bsoc,bsc->bso", R, mx_wtd.to(torch.double)
        )
        R = torch.cat([R, torch.zeros_like(R[:, :, :1, :])], axis=2)
        t = torch.cat([t, torch.ones_like(t[:, :, :1])], axis=-1)
        T = torch.cat([R, t[:, :, :, None]], axis=-1)

        assert T.dtype == torch.double
        assert self.disable_asserts or torch.all(
            torch.isfinite(T),
        ), torch.where(~torch.isfinite(T))
        return T, cum_wts
