from typing import Dict

import numpy as np
import torch
from liso.slim.model.raft_mod import move_channel_to_last_dim
from liso.slim.slim_loss.numerical_stability import normalized_sigmoid_sum
from liso.slim.slim_loss.static_aggregation import (
    batched_grid_data_to_pointwise_data,
    compute_batched_bev_static_aggregated_flow,
)
from munch import Munch
from torch import nn


def homogenize_coors(coors):
    assert coors.shape[-1] == 3
    return torch.cat(
        [
            coors,
            torch.ones(
                list(coors.shape[:-1]) + [1],
                dtype=coors.dtype,
                device=coors.device,
            ),
        ],
        dim=-1,
    )


class HeadDecoder(nn.Module):
    def __init__(self, cfg, name, bev_extent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.name = name
        self.bev_extent = bev_extent

    def concat2network_output(
        self,
        *,
        logits,
        static_flow,
        dynamic_flow,
        weight_logits_for_static_aggregation=None,
    ):
        assert logits.shape[1] == 4
        assert static_flow.shape[1] == 2
        assert dynamic_flow.shape[1] == 2
        assert (weight_logits_for_static_aggregation is None) == (
            not self.cfg.model.predict_weight_for_static_aggregation
        )
        if weight_logits_for_static_aggregation is None:
            concated_output = torch.cat([logits, static_flow, dynamic_flow], axis=1)
        else:
            assert weight_logits_for_static_aggregation.shape[1] == 1
            concated_output = torch.cat(
                [
                    logits,
                    static_flow,
                    dynamic_flow,
                    weight_logits_for_static_aggregation,
                ],
                axis=1,
            )
        concat_output_channels_last = move_channel_to_last_dim(concated_output)
        return concat_output_channels_last

    def apply_output_modification(
        self,
        network_output,
        dynamicness_threshold,
        *,
        pc,
        pointwise_voxel_coordinates_fs,
        pointwise_valid_mask,
        filled_pillar_mask,
        inv_odom,
        per_point_cluster_idxs_gt=None,
        gt_flow_bev=None,
        ohe_gt_stat_dyn_ground_label_bev_map=None,
        dynamic_flow_is_non_rigid_flow=False,
        overwrite_non_filled_pillars_with_default_flow: bool = True,
        overwrite_non_filled_pillars_with_default_logits: bool = True,
    ):
        net_output_device = network_output.device
        flow_dim = 2
        assert 3 == len(filled_pillar_mask.shape) == len(network_output.shape) - 1, (
            filled_pillar_mask.shape,
            network_output.shape,
        )
        assert filled_pillar_mask.shape[-2:] == network_output.shape[-3:-1], (
            filled_pillar_mask.shape,
            network_output.shape,
        )
        filled_pillar_mask = filled_pillar_mask[..., None]

        # network_output_slicing
        network_output_dict = {}
        if self.cfg.model.predict_weight_for_static_aggregation is not False:
            network_output_dict[
                "weight_logits_for_static_aggregation"
            ] = network_output[..., -1]
            network_output = network_output[..., :-1]
        assert network_output.shape[-1] == 4 + 2 * flow_dim
        network_output_dict.update(
            {
                "disappearing_logit": network_output[..., 0:1],
                "static_logit": network_output[..., 1:2],
                "dynamic_logit": network_output[..., 2:3],
                "ground_logit": network_output[..., 3:4],
                "static_flow": network_output[..., 4 : 4 + flow_dim],
                "dynamic_flow": network_output[..., 4 + flow_dim : 4 + 2 * flow_dim],
            }
        )
        final_grid_size = network_output.shape[1:3]

        assert pointwise_voxel_coordinates_fs.shape[-1] == 2

        if "gt_label_based" in self.cfg.model.output_modification.values():
            assert ohe_gt_stat_dyn_ground_label_bev_map is not None
            assert ohe_gt_stat_dyn_ground_label_bev_map.shape[-3:-1] == final_grid_size
            assert ohe_gt_stat_dyn_ground_label_bev_map.shape[-1] == 3
            assert ohe_gt_stat_dyn_ground_label_bev_map.dtype == torch.bool

        # #region precompute gt_static_flow
        bev_extent = np.array(self.bev_extent)
        net_output_shape = final_grid_size
        voxel_center_metric_coordinates = get_voxel_center_coords_m(
            bev_extent, net_output_shape
        )
        homog_metric_voxel_center_coords = torch.from_numpy(
            np.concatenate(
                [
                    voxel_center_metric_coordinates,
                    np.zeros_like(voxel_center_metric_coordinates[..., :1]),
                    np.ones_like(voxel_center_metric_coordinates[..., :1]),
                ],
                axis=-1,
            )
        ).to(inv_odom.device)
        voxel_center_metric_coordinates = torch.from_numpy(
            voxel_center_metric_coordinates
        ).to(inv_odom.device)
        gt_static_flow = torch.einsum(
            "bij,hwj->bhwi",
            inv_odom[:, :2, :]
            - torch.eye(2, m=4, dtype=torch.float64, device=inv_odom.device)[None],
            homog_metric_voxel_center_coords,
        ).to(torch.float32)
        gt_pointwise_static_flow = torch.einsum(
            "bij,bnj->bni",
            inv_odom[:, :3, :]
            - torch.eye(3, m=4, dtype=torch.float64, device=inv_odom.device)[None],
            homogenize_coors(pc[:, :, :3]).to(torch.float64),
        ).to(torch.float32)
        # #endregion precompute gt_static_flow

        (
            network_output_dict,
            static_aggr_trafo,
            not_enough_points,
        ) = artificial_network_output(
            network_output_dict=network_output_dict,
            dynamicness_threshold=dynamicness_threshold,
            cfg=self.cfg,
            ohe_gt_stat_dyn_ground_label_bev_map=ohe_gt_stat_dyn_ground_label_bev_map,
            gt_flow_bev=gt_flow_bev,
            gt_static_flow=gt_static_flow,
            filled_pillar_mask=filled_pillar_mask,
            pc=pc,
            pointwise_voxel_coordinates_fs=pointwise_voxel_coordinates_fs.to(
                net_output_device
            ),
            pointwise_valid_mask=pointwise_valid_mask.to(net_output_device),
            voxel_center_metric_coordinates=voxel_center_metric_coordinates,
            overwrite_non_filled_pillars_with_default_flow=overwrite_non_filled_pillars_with_default_flow,
            overwrite_non_filled_pillars_with_default_logits=overwrite_non_filled_pillars_with_default_logits,
        )

        # with tf.name_scope("slice_output"):
        disappearing_logit = network_output_dict["disappearing_logit"][
            ..., 0
        ]  # TODO is this correct wrt tf?
        disappearing = torch.sigmoid(disappearing_logit)
        class_logits = network_output_dict["class_logits"]
        class_probs = network_output_dict["class_probs"]
        staticness = network_output_dict["staticness"]
        dynamicness = network_output_dict["dynamicness"]
        groundness = network_output_dict["groundness"]
        is_static = network_output_dict["is_static"]
        is_dynamic = network_output_dict["is_dynamic"]
        is_ground = network_output_dict["is_ground"]
        static_flow = network_output_dict["static_flow"]
        static_aggr_flow = network_output_dict["static_aggr_flow"]
        dynamic_flow = network_output_dict["dynamic_flow"]
        dynamic_aggr_flow = network_output_dict.get("dynamic_aggr_flow", None)
        masked_dynamic_aggr_flow = network_output_dict.get(
            "masked_dynamic_aggr_flow", None
        )
        masked_gt_static_flow = network_output_dict["masked_gt_static_flow"]
        masked_static_aggr_flow = network_output_dict["masked_static_aggr_flow"]
        if self.cfg.model.predict_weight_for_static_aggregation is not False:
            masked_weights_for_static_aggregation = network_output_dict[
                "masked_weights_for_static_aggregation"
            ]
        else:
            masked_weights_for_static_aggregation = None
        if flow_dim == 2:
            dynamic_flow = torch.cat(
                [dynamic_flow, torch.zeros_like(dynamic_flow[..., :1])], dim=-1
            )
            static_flow = torch.cat(
                [static_flow, torch.zeros_like(static_flow[..., :1])], dim=-1
            )
            static_aggr_flow = torch.cat(
                [static_aggr_flow, torch.zeros_like(static_aggr_flow[..., :1])],
                dim=-1,
            )
            masked_static_aggr_flow = torch.cat(
                [
                    masked_static_aggr_flow,
                    torch.zeros_like(masked_static_aggr_flow[..., :1]),
                ],
                dim=-1,
            )
            if dynamic_aggr_flow is not None:
                dynamic_aggr_flow = torch.cat(
                    [
                        dynamic_aggr_flow,
                        torch.zeros_like(dynamic_aggr_flow[..., :1]),
                    ],
                    dim=-1,
                )
            if masked_dynamic_aggr_flow is not None:
                masked_dynamic_aggr_flow = torch.cat(
                    [
                        masked_dynamic_aggr_flow,
                        torch.zeros_like(masked_dynamic_aggr_flow[..., :1]),
                    ],
                    dim=-1,
                )

        if self.cfg.model.use_static_aggr_flow_for_aggr_flow:
            static_flow_for_aggr = masked_static_aggr_flow
        else:
            static_flow_for_aggr = static_flow

        assert len(is_static.shape) == 3
        assert len(groundness.shape) == 3
        if dynamic_flow_is_non_rigid_flow:
            aggregated_flow = torch.where(
                torch.tile(
                    is_static[..., None],
                    [1, 1, 1, static_flow_for_aggr.shape[-1]],
                ),
                static_flow_for_aggr,
                (static_flow_for_aggr + dynamic_flow) * (1.0 - groundness[..., None]),
            )
        else:
            aggregated_flow = torch.where(
                torch.tile(
                    is_static[..., None],
                    [1, 1, 1, static_flow_for_aggr.shape[-1]],
                ),
                static_flow_for_aggr,
                dynamic_flow * (1.0 - groundness[..., None]),
            )
        # now we have:
        # disappearing, disappearing_logit
        # class_probs, class_logits, is_static, is_dynamic, is_ground
        # dynamic_flow, static_flow, aggregated_flow
        modified_output_bev_img = Munch(
            disappearing=disappearing,
            disappearing_logit=disappearing_logit,
            class_probs=class_probs,
            class_logits=class_logits,
            staticness=staticness,
            dynamicness=dynamicness,
            groundness=groundness,
            is_static=is_static,
            is_dynamic=is_dynamic,
            is_ground=is_ground,
            dynamic_flow=dynamic_flow,
            static_flow=static_flow,
            aggregated_flow=aggregated_flow,
            static_aggr_flow=static_aggr_flow,
        )
        return (
            modified_output_bev_img,
            network_output_dict,
            gt_flow_bev,
            gt_static_flow,
            gt_pointwise_static_flow,
            masked_gt_static_flow,
            masked_static_aggr_flow,
            masked_weights_for_static_aggregation,
            static_aggr_trafo,
            not_enough_points,
        )

    def apply_flow_to_points(
        self,
        *,
        modified_output_bev_img,
        pointwise_voxel_coordinates_fs,
        pointwise_valid_mask,
    ):
        # with tf.name_scope("apply_flow2points"):
        concat_bool_vals = torch.stack(
            [
                modified_output_bev_img.is_static,
                modified_output_bev_img.is_dynamic,
                modified_output_bev_img.is_ground,
            ],
            dim=-1,
        )
        concat_flt_vals = torch.stack(
            [
                modified_output_bev_img.disappearing,
                modified_output_bev_img.disappearing_logit,
                modified_output_bev_img.staticness,
                modified_output_bev_img.dynamicness,
                modified_output_bev_img.groundness,
            ],
            dim=-1,
        )
        concat_flt_vals = torch.cat(
            [
                concat_flt_vals,
                modified_output_bev_img.class_probs,
                modified_output_bev_img.class_logits,
                modified_output_bev_img.dynamic_flow,
                modified_output_bev_img.static_flow,
                modified_output_bev_img.aggregated_flow,
                modified_output_bev_img.static_aggr_flow,
            ],
            dim=-1,
        )
        num_required_concat_vals = 23
        assert concat_flt_vals.shape[-1] == num_required_concat_vals, (
            concat_flt_vals.shape,
            num_required_concat_vals,
        )

        assert torch.all(
            pointwise_voxel_coordinates_fs >= 0
        ), "negative pixel coordinates found"
        for idx_dim in range(1, 3):
            assert torch.all(
                pointwise_voxel_coordinates_fs[..., idx_dim - 1]
                < concat_bool_vals.shape[idx_dim]
            ), "too large pixel coordinates found"

            assert torch.all(
                pointwise_voxel_coordinates_fs[..., idx_dim - 1]
                < concat_flt_vals.shape[idx_dim]
            ), "too large pixel coordinates found"

        pointwise_concat_bool_vals = batched_grid_data_to_pointwise_data(
            concat_bool_vals,
            pointwise_voxel_coordinates_fs,
            pointwise_valid_mask,
            default_value=False,
        )

        pointwise_concat_flt_vals = batched_grid_data_to_pointwise_data(
            concat_flt_vals,
            pointwise_voxel_coordinates_fs,
            pointwise_valid_mask,
            default_value=0.0,
        )

        assert (
            pointwise_concat_bool_vals.shape[-1] == 3
        ), pointwise_concat_bool_vals.shape
        pointwise_is_static = pointwise_concat_bool_vals[..., 0]
        pointwise_is_dynamic = pointwise_concat_bool_vals[..., 1]
        pointwise_is_ground = pointwise_concat_bool_vals[..., 2]
        assert (
            pointwise_concat_flt_vals.shape[-1] == num_required_concat_vals
        ), pointwise_concat_flt_vals.shape
        pointwise_disappearing = pointwise_concat_flt_vals[..., 0]
        pointwise_disappearing_logit = pointwise_concat_flt_vals[..., 1]
        pointwise_staticness = pointwise_concat_flt_vals[..., 2]
        pointwise_dynamicness = pointwise_concat_flt_vals[..., 3]
        pointwise_groundness = pointwise_concat_flt_vals[..., 4]
        pointwise_class_probs = pointwise_concat_flt_vals[..., 5:8]
        pointwise_class_logits = pointwise_concat_flt_vals[..., 8:11]
        pointwise_dynamic_flow = pointwise_concat_flt_vals[..., 11:14]
        pointwise_static_flow = pointwise_concat_flt_vals[..., 14:17]
        pointwise_aggregated_flow = pointwise_concat_flt_vals[..., 17:20]
        pointwise_static_aggregated_flow = pointwise_concat_flt_vals[..., 20:23]
        retval = Munch(
            disappearing_logit=pointwise_disappearing_logit,
            disappearing=pointwise_disappearing,
            class_logits=pointwise_class_logits,
            class_probs=pointwise_class_probs,
            staticness=pointwise_staticness,
            dynamicness=pointwise_dynamicness,
            groundness=pointwise_groundness,
            is_static=pointwise_is_static,
            is_dynamic=pointwise_is_dynamic,
            is_ground=pointwise_is_ground,
            dynamic_flow=pointwise_dynamic_flow,
            static_flow=pointwise_static_flow,
            aggregated_flow=pointwise_aggregated_flow,
            static_aggr_flow=pointwise_static_aggregated_flow,
        )
        return retval

    def forward(
        self,
        network_output,
        dynamicness_threshold,
        *,
        pc,
        pointwise_voxel_coordinates,
        pointwise_valid_mask,
        filled_pillar_mask,
        odom,
        inv_odom,
        summaries,
        gt_flow_bev=None,
        per_point_cluster_idxs_gt=None,
        ohe_gt_stat_dyn_ground_label_bev_map=None,
        dynamic_flow_is_non_rigid_flow=False,
    ):
        pointwise_voxel_coordinates_fs = torch.div(
            pointwise_voxel_coordinates,
            self.cfg.model.u_net.final_scale,
            rounding_mode="trunc",
        )
        assert pointwise_voxel_coordinates_fs.shape[-1] == 2

        (
            modified_output_bev_img,
            network_output_dict,
            gt_flow_bev,
            _,  # gt_static_flow,
            _,  # gt_pointwise_static_flow,
            _,  # masked_gt_static_flow,
            _,  # masked_static_aggr_flow,
            _,  # masked_weights_for_static_aggregation,
            static_aggr_trafo,
            not_enough_points,
        ) = self.apply_output_modification(
            network_output,
            dynamicness_threshold,
            pc=pc,
            pointwise_voxel_coordinates_fs=pointwise_voxel_coordinates_fs,
            pointwise_valid_mask=pointwise_valid_mask,
            filled_pillar_mask=filled_pillar_mask,
            inv_odom=inv_odom,
            gt_flow_bev=gt_flow_bev,
            ohe_gt_stat_dyn_ground_label_bev_map=ohe_gt_stat_dyn_ground_label_bev_map,
            dynamic_flow_is_non_rigid_flow=dynamic_flow_is_non_rigid_flow,
            per_point_cluster_idxs_gt=per_point_cluster_idxs_gt,
        )

        pointwise_output = self.apply_flow_to_points(
            modified_output_bev_img=modified_output_bev_img,
            pointwise_voxel_coordinates_fs=pointwise_voxel_coordinates_fs,
            pointwise_valid_mask=pointwise_valid_mask,
        )

        retval = Munch(
            **pointwise_output,
            dense_maps=Munch(
                # agg_flow_img=agg_flow_img,
                aggregated_flow=modified_output_bev_img.aggregated_flow,
                static_flow=modified_output_bev_img.static_flow,
            ),
            modified_network_output=Munch(network_output_dict),
        )

        # retval = self.output_decoder_summaries(
        #     network_output_dict=network_output_dict,
        #     pointwise_output=pointwise_output,
        #     modified_output_bev_img=modified_output_bev_img,
        #     summaries=summaries,
        #     gt_static_flow=gt_static_flow,
        #     masked_gt_static_flow=masked_gt_static_flow,
        #     masked_static_aggr_flow=masked_static_aggr_flow,
        #     masked_weights_for_static_aggregation=masked_weights_for_static_aggregation,
        #     gt_pointwise_static_flow=gt_pointwise_static_flow,
        #     pointwise_valid_mask=pointwise_valid_mask,
        #     filled_pillar_mask=filled_pillar_mask,
        #     odom=odom,
        #     static_aggr_trafo=static_aggr_trafo,
        #     not_enough_points=not_enough_points,
        #     gt_flow_bev=gt_flow_bev,
        # )

        retval["static_aggr_trafo"] = static_aggr_trafo
        retval["dynamicness_threshold"] = dynamicness_threshold
        retval["not_enough_points"] = not_enough_points
        return retval


def get_voxel_center_coords_m(bev_extent_m, net_output_shape_pix):
    voxel_center_metric_coordinates = (
        np.stack(
            np.meshgrid(
                np.arange(net_output_shape_pix[0]),
                np.arange(net_output_shape_pix[1]),
                indexing="ij",
            ),
            axis=-1,
        )
        + 0.5
    )
    voxel_center_metric_coordinates /= net_output_shape_pix
    voxel_center_metric_coordinates *= bev_extent_m[2:] - bev_extent_m[:2]
    voxel_center_metric_coordinates += bev_extent_m[:2]
    return voxel_center_metric_coordinates


def artificial_network_output(
    *,
    network_output_dict: Dict[str, torch.Tensor],
    dynamicness_threshold: torch.Tensor,
    cfg,
    ohe_gt_stat_dyn_ground_label_bev_map: torch.Tensor,
    gt_flow_bev: torch.Tensor,
    gt_static_flow: torch.Tensor,
    filled_pillar_mask: torch.Tensor,
    pc: torch.Tensor,
    pointwise_voxel_coordinates_fs: torch.Tensor,
    pointwise_valid_mask: torch.Tensor,
    voxel_center_metric_coordinates: np.array,
    overwrite_non_filled_pillars_with_default_flow: bool = False,
    overwrite_non_filled_pillars_with_default_logits: bool = False,
):
    model_cfg = cfg.model
    out_mod_cfg = model_cfg.output_modification

    assert len(network_output_dict["static_flow"].shape) == len(
        filled_pillar_mask.shape
    ), (
        network_output_dict["static_flow"].shape,
        filled_pillar_mask.shape,
    )

    # test_batched_grid_data_to_pointwise_data(
    #     pc,
    #     pointwise_valid_mask,
    #     pointwise_voxel_coordinates_fs,
    #     model_cfg.point_pillars.nbr_pillars,
    #     bev_extent=cfg.data.bev_extent,
    # )

    network_output_dict = artificial_flow_network_output(
        network_output_dict=network_output_dict,
        model_cfg=model_cfg,
        gt_flow_bev=gt_flow_bev,
        gt_static_flow=gt_static_flow,
    )

    network_output_dict = artificial_logit_network_output(
        network_output_dict=network_output_dict,
        model_cfg=model_cfg,
        ohe_gt_stat_dyn_ground_label_bev_map=ohe_gt_stat_dyn_ground_label_bev_map,
        gt_flow_bev=gt_flow_bev,
        gt_static_flow=gt_static_flow,
    )

    # with tf.name_scope("mask_nonfilled_pillars"):
    default_values_for_nonfilled_pillars = {
        "disappearing_logit": -100.0,
        "static_logit": -100.0 if out_mod_cfg.static_logit is False else 0.0,
        "dynamic_logit": 0.0 if out_mod_cfg.dynamic_logit is True else -100.0,
        "ground_logit": 0.0 if out_mod_cfg.ground_logit is True else -100.0,
        "static_flow": 0.0,
        "dynamic_flow": 0.0,
        "static_aggr_flow": 0.0,
    }

    modification_taboo_keys = []
    if not overwrite_non_filled_pillars_with_default_flow:
        modification_taboo_keys += [
            "static_flow",
            "dynamic_flow",
            "static_aggr_flow",
        ]
    if not overwrite_non_filled_pillars_with_default_logits:
        modification_taboo_keys += [
            "disappearing_logit",
            "static_logit",
            "dynamic_logit",
            "ground_logit",
        ]

    for k in network_output_dict:
        if k == "weight_logits_for_static_aggregation":
            continue
        assert len(network_output_dict[k].shape) == len(filled_pillar_mask.shape), (
            k,
            network_output_dict[k].shape,
            filled_pillar_mask.shape,
        )

        if k in modification_taboo_keys:
            continue

        network_output_dict[k] = torch.where(
            filled_pillar_mask,
            network_output_dict[k],
            default_values_for_nonfilled_pillars[k]
            * torch.ones_like(network_output_dict[k]),
        )

    # with tf.name_scope("construct_class_probs"):
    network_output_dict["class_logits"] = torch.cat(
        [
            network_output_dict["static_logit"],
            network_output_dict["dynamic_logit"],
            network_output_dict["ground_logit"],
        ],
        dim=-1,
    )
    network_output_dict["class_probs"] = torch.nn.functional.softmax(
        network_output_dict["class_logits"], dim=-1
    )
    network_output_dict["staticness"] = network_output_dict["class_probs"][..., 0]
    network_output_dict["dynamicness"] = network_output_dict["class_probs"][..., 1]
    network_output_dict["groundness"] = network_output_dict["class_probs"][..., 2]
    network_output_dict["is_dynamic"] = (
        network_output_dict["dynamicness"] >= dynamicness_threshold
    )
    network_output_dict["is_static"] = (
        network_output_dict["staticness"] >= network_output_dict["groundness"]
    ) & (~network_output_dict["is_dynamic"])
    network_output_dict["is_ground"] = ~(
        network_output_dict["is_static"] | network_output_dict["is_dynamic"]
    )

    # with tf.name_scope("construct_static_aggregation"):
    static_aggr_weight_map = network_output_dict["staticness"] * castf(
        filled_pillar_mask[..., 0]
    )
    if model_cfg.predict_weight_for_static_aggregation is not False:
        mode = model_cfg.predict_weight_for_static_aggregation
        assert mode in {"sigmoid", "softmax"}
        if mode == "softmax":
            network_output_dict["masked_weights_for_static_aggregation"] = torch.where(
                filled_pillar_mask[..., 0],
                network_output_dict["weight_logits_for_static_aggregation"],
                torch.ones_like(
                    network_output_dict["weight_logits_for_static_aggregation"]
                )
                * (
                    torch.min(
                        network_output_dict["weight_logits_for_static_aggregation"]
                    )
                    - 1000.0
                ),
            )
            curshape = network_output_dict[
                "masked_weights_for_static_aggregation"
            ].shape
            assert len(curshape) == 3, curshape
            prodshape = curshape[-1] * curshape[-2]
            network_output_dict[
                "masked_weights_for_static_aggregation"
            ] = torch.reshape(
                torch.nn.functional.softmax(
                    torch.reshape(
                        network_output_dict["masked_weights_for_static_aggregation"],
                        (-1, prodshape),
                    )
                ),
                (-1, *curshape[-2:]),
            )
        else:
            assert mode == "sigmoid"
            grid_size = filled_pillar_mask.shape[-3:-1]
            prod_size = grid_size[0] * grid_size[1]
            network_output_dict[
                "masked_weights_for_static_aggregation"
            ] = torch.reshape(
                normalized_sigmoid_sum(
                    logits=torch.reshape(
                        network_output_dict["weight_logits_for_static_aggregation"],
                        [-1, prod_size],
                    ),
                    mask=torch.reshape(filled_pillar_mask[..., 0], [-1, prod_size]),
                ),
                [-1, *grid_size],
            )
        static_aggr_weight_map = (
            static_aggr_weight_map
            * network_output_dict["masked_weights_for_static_aggregation"]
        )
    (
        network_output_dict["static_aggr_flow"],
        static_aggr_trafo,
        not_enough_points,
    ) = compute_batched_bev_static_aggregated_flow(
        pc,
        pointwise_voxel_coordinates_fs,
        pointwise_valid_mask,
        network_output_dict["static_flow"],
        static_aggr_weight_map,
        voxel_center_metric_coordinates,
        use_eps_for_weighted_pc_alignment=cfg.losses.unsupervised.use_epsilon_for_weighted_pc_alignment,
    )
    network_output_dict["masked_static_aggr_flow"] = torch.where(
        filled_pillar_mask,
        network_output_dict["static_aggr_flow"],
        torch.zeros_like(network_output_dict["static_aggr_flow"]),
    )
    network_output_dict["masked_gt_static_flow"] = torch.where(
        filled_pillar_mask,
        gt_static_flow,
        torch.zeros_like(network_output_dict["masked_static_aggr_flow"]),
    )

    return network_output_dict, static_aggr_trafo, not_enough_points


def scale_gradient(tensor, scaling):
    if scaling == 1.0:
        return tensor
    if scaling == 0.0:
        return tensor.detach()
    assert scaling > 0.0
    return tensor * scaling - tensor.detach() * (scaling - 1.0)


def castf(tensor):
    if tensor.dtype not in {torch.float32, torch.float64}:
        tensor = tensor.float()
    return tensor


def artificial_flow_network_output(
    *,
    network_output_dict: Dict[str, torch.Tensor],
    model_cfg,
    gt_flow_bev: torch.Tensor,
    gt_static_flow: torch.Tensor,
):
    out_mod_cfg = model_cfg.output_modification
    # #region static_flow
    if out_mod_cfg.static_flow == "net":
        pass
    elif out_mod_cfg.static_flow == "gt":
        network_output_dict["static_flow"] = gt_static_flow
    elif out_mod_cfg.static_flow == "zero":
        network_output_dict["static_flow"] = torch.zeros_like(
            network_output_dict["static_flow"]
        )
    else:
        raise ValueError("unknown output mode: %s" % str(out_mod_cfg.static_flow))
    # #endregion static_flow

    # #region dynamic_flow
    if out_mod_cfg.dynamic_flow == "net":
        pass
    elif out_mod_cfg.dynamic_flow == "gt":
        network_output_dict["dynamic_flow"] = gt_flow_bev
        if model_cfg.dynamic_flow_is_non_rigid_flow:
            network_output_dict["dynamic_flow"] = (
                network_output_dict["dynamic_flow"] - network_output_dict["static_flow"]
            )
    elif out_mod_cfg.dynamic_flow == "zero":
        network_output_dict["dynamic_flow"] = torch.zeros_like(
            network_output_dict["dynamic_flow"]
        )
    else:
        raise ValueError("unknown output mode: %s" % str(out_mod_cfg.dynamic_flow))
    network_output_dict["dynamic_flow"] = scale_gradient(
        network_output_dict["dynamic_flow"],
        out_mod_cfg.dynamic_flow_grad_scale,
    )
    # #endregion dynamic_flow
    return network_output_dict


def artificial_logit_network_output(
    *,
    network_output_dict: Dict[str, torch.Tensor],
    model_cfg,
    ohe_gt_stat_dyn_ground_label_bev_map: torch.Tensor,
    gt_flow_bev: torch.Tensor,
    gt_static_flow: torch.Tensor,
):
    out_mod_cfg = model_cfg.output_modification
    ones = torch.ones_like(network_output_dict["static_logit"])

    # #region disappearing_logit
    if out_mod_cfg.disappearing_logit == "net":
        pass
    elif out_mod_cfg.disappearing_logit == "gt":
        raise NotImplementedError()
    elif out_mod_cfg.disappearing_logit is True:
        network_output_dict["disappearing_logit"] = 0 * ones
    elif out_mod_cfg.disappearing_logit is False:
        network_output_dict["disappearing_logit"] = -100 * ones
    else:
        raise ValueError(
            "unknown output mode: %s" % str(out_mod_cfg.disappearing_logit)
        )
    # #endregion disappearing_logit

    # #region static_logit
    if out_mod_cfg.static_logit == "net":
        pass
    elif out_mod_cfg.static_logit == "gt_label_based":
        assert out_mod_cfg.dynamic_logit == "gt_label_based"
        if out_mod_cfg.ground_logit is False:
            # add gt labels to static if ground == off
            gt_stat = (
                ohe_gt_stat_dyn_ground_label_bev_map[..., 0:1]
                | ohe_gt_stat_dyn_ground_label_bev_map[..., 2:3]
            )
            gt_stat_flt = castf(gt_stat)
            network_output_dict["static_logit"] = 100.0 * (gt_stat_flt - 1.0)
        elif out_mod_cfg.ground_logit == "gt_label_based":
            network_output_dict["static_logit"] = 100.0 * (
                castf(ohe_gt_stat_dyn_ground_label_bev_map[..., 0:1]) - 1.0
            )
        else:
            raise AssertionError(
                "when using gt_label for cls then ground_logit must be `gt_label_based` or `off`, not %s"
                % out_mod_cfg.ground_logit
            )
    elif out_mod_cfg.static_logit == "gt_flow_based":
        assert out_mod_cfg.dynamic_logit == "gt_flow_based"
        assert out_mod_cfg.ground_logit is False

        norig_flow = gt_flow_bev - gt_static_flow
        bev_is_static_map = torch.linalg.norm(norig_flow, dim=-1, keepdim=True) <= 0.05
        bev_is_static_map = castf(bev_is_static_map)
        network_output_dict["static_logit"] = 100.0 * (bev_is_static_map - 1.0)
    elif out_mod_cfg.static_logit is True:
        assert out_mod_cfg.dynamic_logit is False
        assert out_mod_cfg.ground_logit is False
        network_output_dict["static_logit"] = (
            torch.max(
                torch.cat(
                    [
                        network_output_dict["dynamic_logit"],
                        network_output_dict["ground_logit"],
                    ],
                    dim=0,
                )
            ).detach()
            + 100.0 * ones
        )
    elif out_mod_cfg.static_logit is False:
        assert (
            out_mod_cfg.dynamic_logit is not False
            or out_mod_cfg.ground_logit is not False
        )
        network_output_dict["static_logit"] = (
            torch.max(
                torch.cat(
                    [
                        network_output_dict["dynamic_logit"],
                        network_output_dict["ground_logit"],
                    ],
                    dim=0,
                )
            ).detach()
            - 100.0 * ones
        )
    else:
        raise ValueError("unknown output mode: %s" % str(out_mod_cfg.static_logit))
    # #endregion static_logit

    # #region dynamic_logit
    if out_mod_cfg.dynamic_logit == "net":
        pass
    elif out_mod_cfg.dynamic_logit == "gt_label_based":
        assert out_mod_cfg.static_logit == "gt_label_based"
        network_output_dict["dynamic_logit"] = 100.0 * (
            castf(ohe_gt_stat_dyn_ground_label_bev_map[..., 1:2]) - 1.0
        )
    elif out_mod_cfg.dynamic_logit == "gt_flow_based":
        network_output_dict["dynamic_logit"] = (
            100.0 - network_output_dict["static_logit"]
        )
    elif out_mod_cfg.dynamic_logit is True:
        # assert out_mod_cfg.static_logit is False
        # assert out_mod_cfg.ground_logit is False
        network_output_dict["dynamic_logit"] = (
            torch.max(
                torch.cat(
                    [
                        network_output_dict["static_logit"],
                        network_output_dict["ground_logit"],
                    ],
                    dim=0,
                )
            ).detach()
            + 100.0 * ones
        )
    elif out_mod_cfg.dynamic_logit is False:
        network_output_dict["dynamic_logit"] = (
            torch.min(
                torch.cat(
                    [
                        network_output_dict["static_logit"],
                        network_output_dict["ground_logit"],
                    ],
                    dim=0,
                )
            ).detach()
            - 100.0 * ones
        )
    else:
        raise ValueError("unknown output mode: %s" % str(out_mod_cfg.dynamic_logit))
    # #endregion dynamic_logit

    # #region ground_logit
    if out_mod_cfg.ground_logit == "net":
        pass
    elif out_mod_cfg.ground_logit == "gt_label_based":
        assert out_mod_cfg.static_logit == "gt_label_based"
        assert out_mod_cfg.dynamic_logit == "gt_label_based"
        network_output_dict["ground_logit"] = 100.0 * (
            castf(ohe_gt_stat_dyn_ground_label_bev_map[..., 2:3]) - 1.0
        )
    elif out_mod_cfg.ground_logit is True:
        assert out_mod_cfg.static_logit is False
        assert out_mod_cfg.dynamic_logit is False
        network_output_dict["ground_logit"] = (
            torch.max(
                torch.cat(
                    [
                        network_output_dict["static_logit"],
                        network_output_dict["dynamic_logit"],
                    ],
                    dim=0,
                )
            ).detach()
            + 100.0 * ones
        )
    elif out_mod_cfg.ground_logit is False:
        network_output_dict["ground_logit"] = (
            torch.min(
                torch.cat(
                    [
                        network_output_dict["static_logit"],
                        network_output_dict["dynamic_logit"],
                    ],
                    dim=0,
                )
            ).detach()
            - 100.0 * ones
        )
    else:
        raise ValueError("unknown output mode: %s" % str(out_mod_cfg.ground_logit))
    # #endregion ground_logit
    return network_output_dict
