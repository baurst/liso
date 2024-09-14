import torch


def modify_pred_pos(
    pred_pos,
    box_pred_cfg,
    data_cfg,
    pillar_center_coors_m: torch.FloatTensor,
):
    if box_pred_cfg.position_representation.method == "global_relative":
        assert box_pred_cfg.activations.pos in ("tanh",)
        bev_dims_m = torch.tensor(data_cfg.bev_range_m, device=pred_pos.device)
        modified_pos = pred_pos * bev_dims_m * 0.6
    elif box_pred_cfg.position_representation.method == "local_relative_offset":
        assert box_pred_cfg.activations.pos in (
            "tanh",
            "none",
        ), box_pred_cfg.activations.pos
        assert len(pred_pos.shape) == 4, pred_pos.shape
        assert pillar_center_coors_m.shape[:-1] == pred_pos.shape[1:3], (
            pillar_center_coors_m.shape,
            pred_pos.shape,
        )

        local_bev_img_shape = pred_pos.shape[1:3]
        local_voxel_resolution = (
            torch.tensor(data_cfg.bev_range_m) / torch.tensor(local_bev_img_shape)
        ).to(pred_pos.device)
        box_xy_offset = local_voxel_resolution * 0.5 * pred_pos[..., :2]
        modified_pos = pillar_center_coors_m[None, ...] + box_xy_offset
        if box_pred_cfg.position_representation.num_box_pos_dims == 3:
            assert pred_pos.shape[-1] == 3, pred_pos.shape
            box_z = box_pred_cfg.position_representation.box_z_pos_prior_min + 0.5 * (
                pred_pos[..., [-1]] + 1.0
            ) * (
                box_pred_cfg.position_representation.box_z_pos_prior_max
                - box_pred_cfg.position_representation.box_z_pos_prior_min
            )
            modified_pos = torch.cat([modified_pos, box_z], dim=-1)
    elif box_pred_cfg.position_representation.method == "global_absolute":
        modified_pos = pred_pos
    else:
        raise NotImplementedError(box_pred_cfg.position_representation.method)
    return modified_pos


def maybe_flatten_anchors_except_for(box_vars_pred, do_not_flatten=("pos",)):
    for box_attr_name, box_attr_val in box_vars_pred.items():
        if box_attr_name in do_not_flatten:
            continue
        elif len(box_attr_val.shape) == 4:
            box_vars_pred[box_attr_name] = torch.flatten(
                box_attr_val, start_dim=1, end_dim=2
            )
    return box_vars_pred


def box_pred_convention_to_gt_convention(
    box_vars_pred,
    box_pred_cfg,
    data_cfg,
    pillar_center_coors_m: torch.FloatTensor,
):
    # box_vars_pred = maybe_flatten_anchors_except_for_pos(box_vars_pred)
    # DIMENSIONS
    if box_pred_cfg.dimensions_representation.method == "predict_aspect_ratio":
        box_scale, box_aspect_ratio_inv = torch.split(box_vars_pred["dims"], 1, dim=-1)
        box_len = (
            box_pred_cfg.dimensions_representation.box_len_prior_min
            + box_scale
            * (
                box_pred_cfg.dimensions_representation.box_len_prior_max
                - box_pred_cfg.dimensions_representation.box_len_prior_min
            )
        )
        box_width = box_len * box_aspect_ratio_inv
        box_vars_pred["dims"] = torch.cat([box_len, box_width], dim=-1)

    elif box_pred_cfg.dimensions_representation.method == "predict_abs_size":
        pass
    elif box_pred_cfg.dimensions_representation.method == "predict_log_size":
        assert box_pred_cfg.activations.dims == "exp", box_pred_cfg.activations.dims
        box_vars_pred["dims"] = torch.exp(box_vars_pred["dims"])
    else:
        raise NotImplementedError(box_pred_cfg.dimensions_representation.method)

    # ROTATION
    if box_pred_cfg.rotation_representation.method == "vector":
        if box_pred_cfg.rotation_representation.norm_vector_len:
            vec_normed = torch.nn.functional.normalize(
                box_vars_pred["rot"], p=2.0, dim=-1
            )
            sin_yaw, cos_yaw = torch.split(vec_normed, 1, dim=-1)
            theta = torch.atan2(sin_yaw, cos_yaw)
        else:
            rot_vec = box_vars_pred["rot"]
            # centermaps:
            # rot[0] == sin(theta) "== y"
            # rot[1] == cos(theta) "== x"
            # theta = atan(y,x)
            sin_yaw, cos_yaw = torch.split(rot_vec, 1, dim=-1)
            theta = torch.atan2(sin_yaw, cos_yaw)
        box_vars_pred["rot"] = theta
    elif box_pred_cfg.rotation_representation.method == "direct":
        pass
    elif box_pred_cfg.rotation_representation.method == "class_bins":
        num_bins = 36
        bin_size = 2 * torch.pi / num_bins
        bin_idx = torch.argmax(box_vars_pred["rot"], dim=-1, keepdim=True)
        box_vars_pred["rot"] = bin_idx * bin_size
    else:
        raise NotImplementedError(box_pred_cfg.rotation_representation.method)

    # POSITION
    box_vars_pred["pos"] = modify_pred_pos(
        box_vars_pred["pos"],
        box_pred_cfg,
        data_cfg,
        pillar_center_coors_m,
    )

    # if len(box_vars_pred["pos"].shape) == 4:
    #     box_vars_pred["pos"] = torch.flatten(
    #         box_vars_pred["pos"], start_dim=1, end_dim=2
    #     )

    return box_vars_pred


def output_modification(
    box_vars_pred,
    box_pred_cfg,
    data_cfg,
    shape_name: str,
    pillar_center_coors_m: torch.FloatTensor,
):
    box_vars_pred = {k: v.clone() for k, v in box_vars_pred.items()}
    if shape_name == "boxes":
        box_vars_pred = box_pred_convention_to_gt_convention(
            box_vars_pred,
            box_pred_cfg,
            data_cfg,
            pillar_center_coors_m,
        )
    else:
        raise NotImplementedError(shape_name)

    return box_vars_pred
