from typing import Any, Dict

import torch
from liso.slim.slim_loss.artificial_labels_pytorch import compute_artificial_label_loss
from liso.slim.slim_loss.knn_loss import compute_knn_loss_components
from liso.utils.torch_transformation import homogenize_pcl


def trafo_distance(delta_trafos, points, mask):
    dim = delta_trafos.shape[-1] - 1
    assert list(delta_trafos.shape)[-2] == dim + 1
    # assert delta_trafos.dtype == tf.float64
    assert list(points.shape)[-1] == dim
    assert len(list(points.shape)) == 3
    # assert points.dtype == tf.float32
    points = points.detach()
    # p_mask = torch.logical_not(tf.reduce_all(tf.math.is_nan(points), axis=-1))
    count = mask.sum(dim=-1)
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    # points_h = tf.where(
    #     tf.tile(p_mask[..., None], [1, 1, 4]), points_h, tf.zeros_like(points_h)
    # )
    points_h[
        ~mask
    ] = 0.0  # all zero homog points are 0.0 after trafo -> do not mess up mean
    delta_points = cast32(
        torch.einsum("b...ij,bkj->b...ki", delta_trafos[..., :3, :], cast64(points_h))
    )
    delta_lengths_sqr = torch.sum(delta_points**2, dim=-1)
    avg_dist_sqr = torch.sum(delta_lengths_sqr, dim=-1) / count
    return avg_dist_sqr


def cast64(stuff):
    # TODO fix this try stuff.to(torch.double)
    return stuff.to(torch.double)
    # return stuff.type(torch.DoubleTensor)


def cast32(stuff):
    return stuff.float()


def weighted_mse_loss(input_tensor, target, weight):
    assert input_tensor.size(-1) == 3
    assert torch.all(torch.isfinite(input_tensor))
    assert torch.all(torch.isfinite(target))
    assert torch.all(torch.isfinite(weight))
    return ((weight.clone() * (input_tensor - target.clone()) ** 2)).mean(dim=-1)


def static_points_loss(
    pc,
    valid_mask,
    flow,
    weights,
    trafo,
):
    assert pc.shape[-1] in [3, 4], pc.shape
    assert flow.shape[-1] == 3, flow.shape
    assert len(list(weights.shape)) == 2
    assert list(trafo.shape)[1:] == [
        4,
        4,
    ], "The returned trafo from weighted_pc_alignment has only shape %s!" % str(
        trafo.shape[1:]
    )
    pc_valid = torch.where(
        valid_mask[..., None], pc, torch.tensor(0.0, dtype=pc.dtype, device=pc.device)
    )
    pc_hom = homogenize_pcl(pc_valid[..., :3])

    assert list(pc_hom.shape)[2:] == [4], pc_hom.shape
    assert list(trafo.shape[1:]) == [4, 4], trafo.shape

    assert torch.all(torch.isfinite(pc_hom))
    pc_trafo_hom = torch.sum(
        cast64(pc_hom[:, :, None, :]) * trafo.detach()[:, None, :, :], dim=-1
    )
    # pc_trafo_hom.values.set_shape(pc_trafo_hom.values.shape.as_list()[:-1] + [4])
    assert list(pc_trafo_hom.shape)[2:] == [4], list(pc_trafo_hom.shape)

    trafo_flow_est = cast32(pc_trafo_hom[..., :3] - cast64(pc_valid[..., :3]))
    assert list(trafo_flow_est.shape)[2:] == [3], list(trafo_flow_est.shape)
    loss = weighted_mse_loss(trafo_flow_est, flow, weights[..., None])
    assert loss.ndim == 2, loss.shape  # [B, N]
    return loss


def symmetric_static_points_loss(
    pc0,
    valid_mask_pc0,
    static_flow_fw,
    static_aggr_trafo_fw,
    staticness_fw,
    pc1=None,
    valid_mask_pc1=None,
    static_aggr_trafo_bw=None,
):
    assert (pc1 is None) == (static_aggr_trafo_bw is None)

    loss0 = static_points_loss(
        pc0, valid_mask_pc0, static_flow_fw, staticness_fw, static_aggr_trafo_fw
    )
    # loss1 = static_points_loss(pc1, static_flow_bw, staticness_bw, static_aggr_trafo_bw)
    # static_flow_loss = 0.5 * (loss0 + loss1)
    static_flow_loss = loss0

    if pc1 is not None:
        for_back_trafo = torch.einsum(
            "boc,bcx->box", static_aggr_trafo_bw, static_aggr_trafo_fw
        )
        for_back_trafo_loss = trafo_distance(
            for_back_trafo - torch.eye(4, device=for_back_trafo.device),
            torch.cat([pc0[..., :3], pc1[..., :3]], axis=1),
            mask=torch.cat([valid_mask_pc0, valid_mask_pc1], axis=1),
        ).mean()
        return static_flow_loss, for_back_trafo_loss
    else:
        return static_flow_loss


def selfsupervisedSlimSingleScaleLoss(
    pc1,
    valid_mask_pc1,
    pc2,
    valid_mask_pc2,
    pred_fw,
    pred_bw,
    moving_thresh_module,
    *,
    loss_cfg,
    model_cfg,
    bev_extent,
    metrics_collector: Dict[str, Any],
    training=True,
):
    # pc1 = pc1.permute(0, 2, 1)
    # pc2 = pc2.permute(0, 2, 1)
    # flow_fw = flow_fw.permute(0, 2, 1)
    # flow_bw = flow_bw.permute(0, 2, 1)
    # if flow_static_aggr_fw is not None:
    #     flow_static_aggr_fw = flow_static_aggr_fw.permute(0, 2, 1)
    #     flow_static_aggr_bw = flow_static_aggr_bw.permute(0, 2, 1)
    total_loss = torch.zeros(1).cuda()
    # predictions = {"fw": prediction_fw, "bw": prediction_bw}
    is_static_flow_penalty_active = loss_cfg.static_flow_penalty_factor != 0.0
    is_fw_bw_static_trafo_penalty_factor_active = (
        loss_cfg.fw_bw_static_trafo_penalty_factor != 0.0
    )
    if is_static_flow_penalty_active or is_fw_bw_static_trafo_penalty_factor_active:
        if loss_cfg.artificial_labels.cross_entropy_penalty > 0.0:
            staticness_fw = pred_fw.staticness.detach()
            staticness_bw = pred_bw.staticness.detach()
        else:
            staticness_fw = pred_fw.staticness
            staticness_bw = pred_bw.staticness
        static_flow_loss_fw, for_back_static_trafo_loss = symmetric_static_points_loss(
            pc0=pc1,
            valid_mask_pc0=valid_mask_pc1,
            static_flow_fw=pred_fw.static_flow,
            static_aggr_trafo_fw=pred_fw.static_aggr_trafo,
            staticness_fw=staticness_fw,
            pc1=pc2,
            valid_mask_pc1=valid_mask_pc2,
            static_aggr_trafo_bw=pred_bw.static_aggr_trafo,
        )
        static_flow_loss_fw = static_flow_loss_fw[valid_mask_pc1].mean()
        static_flow_loss_bw = symmetric_static_points_loss(
            pc0=pc2,
            valid_mask_pc0=valid_mask_pc2,
            static_flow_fw=pred_bw.static_flow,
            static_aggr_trafo_fw=pred_bw.static_aggr_trafo,
            staticness_fw=staticness_bw,
        )[valid_mask_pc2].mean()
        static_flow_loss = 0.5 * (static_flow_loss_fw + static_flow_loss_bw)
        metrics_collector["static_flow_loss"] = static_flow_loss.detach().cpu().numpy()
        metrics_collector["for_back_static_trafo_loss"] = (
            for_back_static_trafo_loss.detach().cpu().numpy()
        )

        if is_static_flow_penalty_active:
            assert loss_cfg.static_flow_penalty_factor > 0.0
            total_loss = (
                total_loss + static_flow_loss * loss_cfg.static_flow_penalty_factor
            )
        if is_fw_bw_static_trafo_penalty_factor_active:
            assert loss_cfg.fw_bw_static_trafo_penalty_factor > 0.0
            total_loss = (
                total_loss
                + for_back_static_trafo_loss
                * loss_cfg.fw_bw_static_trafo_penalty_factor
            )
    knn_results_fw = compute_knn_loss_components(
        pc1[..., :3],
        valid_mask_pc1,
        pc2[..., :3],
        valid_mask_pc2,
        prediction=pred_fw,
        loss_cfg=loss_cfg,
        model_cfg=model_cfg,
        bev_extent=bev_extent,
    )
    knn_results_bw = compute_knn_loss_components(
        pc2[..., :3],
        valid_mask_pc2,
        pc1[..., :3],
        valid_mask_pc1,
        prediction=pred_bw,
        loss_cfg=loss_cfg,
        model_cfg=model_cfg,
        bev_extent=bev_extent,
    )
    assert set(knn_results_fw.keys()) == set(knn_results_bw.keys())

    if loss_cfg.artificial_labels.cross_entropy_penalty > 0.0:
        prediction_fw = {"staticness": pred_fw.staticness}
        prediction_bw = {"staticness": pred_bw.staticness}
        ce_loss_fw = compute_artificial_label_loss(
            prediction=prediction_fw,
            knn_results=knn_results_fw,
            loss_cfg=loss_cfg,
        )[valid_mask_pc1].mean()
        ce_loss_bw = compute_artificial_label_loss(
            prediction=prediction_bw,
            knn_results=knn_results_bw,
            loss_cfg=loss_cfg,
        )[valid_mask_pc2].mean()
        metrics_collector["ce_loss_fw"] = ce_loss_fw.detach().cpu().numpy()
        metrics_collector["ce_loss_bw"] = ce_loss_bw.detach().cpu().numpy()
    else:
        assert loss_cfg.artificial_labels.cross_entropy_penalty == 0.0
        assert loss_cfg.artificial_labels.gauss_widths is None

    if loss_cfg.knn_loss.range_based_weights.weight_slope != 0.0:
        fw_knn_weights = get_range_based_knn_loss_weights(pc1, valid_mask_pc1, loss_cfg)
        bw_knn_weights = get_range_based_knn_loss_weights(pc2, valid_mask_pc2, loss_cfg)

    else:
        fw_knn_weights = torch.ones_like(pc1[..., 0])
        bw_knn_weights = torch.ones_like(pc2[..., 0])

    forward_flow_loss = (fw_knn_weights * knn_results_fw["aggregated"]["loss"])[
        valid_mask_pc1
    ].mean()
    backward_flow_loss = (bw_knn_weights * knn_results_bw["aggregated"]["loss"])[
        valid_mask_pc2
    ].mean()

    flow_loss = 0.5 * (backward_flow_loss + forward_flow_loss)

    if loss_cfg.knn_loss_penalty_factor != 0.0:
        assert loss_cfg.knn_loss_penalty_factor > 0.0
        total_loss = total_loss + flow_loss * loss_cfg.knn_loss_penalty_factor

    if loss_cfg.knn_on_dynamic_penalty != 0.0:
        assert loss_cfg.knn_on_dynamic_penalty > 0.0
        fw_dyn_loss = (fw_knn_weights * knn_results_fw["dynamic"]["loss"])[
            valid_mask_pc1
        ].mean()
        bw_dyn_loss = (bw_knn_weights * knn_results_bw["dynamic"]["loss"])[
            valid_mask_pc2
        ].mean()
        dynamic_flow_loss = 0.5 * (bw_dyn_loss + fw_dyn_loss)
        metrics_collector["dynamic_flow_loss"] = (
            dynamic_flow_loss.detach().cpu().numpy()
        )
        total_loss = total_loss + dynamic_flow_loss * loss_cfg.knn_on_dynamic_penalty

    if loss_cfg.knn_on_static_penalty != 0.0:
        assert loss_cfg.knn_on_static_penalty == 1.0
        assert loss_cfg.knn_on_static_penalty > 0.0
        assert model_cfg.use_static_aggr_flow_for_aggr_flow, (loss_cfg, model_cfg)
        if model_cfg.use_static_aggr_flow_for_aggr_flow:
            static_flow_key = "static_aggr"
        else:
            static_flow_key = "static"

        fw_stat_loss = (fw_knn_weights * knn_results_fw[static_flow_key]["loss"])[
            valid_mask_pc1
        ].mean()
        bw_stat_loss = (bw_knn_weights * knn_results_bw[static_flow_key]["loss"])[
            valid_mask_pc2
        ].mean()
        static_flow_loss = 0.5 * (bw_stat_loss + fw_stat_loss)
        metrics_collector["static_flow_loss"] = static_flow_loss.detach().cpu().numpy()
        total_loss = total_loss + static_flow_loss * loss_cfg.knn_on_static_penalty
    else:
        # TODO: dont think this is necessary
        # assert flow_static_aggr_fw is None
        # assert flow_static_aggr_bw is None
        pass

    assert loss_cfg.opposite_flow_penalty_factor == 0.0

    # #region update dynamicness threshold
    if model_cfg.use_static_aggr_flow_for_aggr_flow:
        # assert staticness_fw is not None
        # assert staticness_bw is not None
        assert model_cfg.use_static_aggr_flow_for_aggr_flow

        predictions_fw = {"dynamicness": pred_fw.dynamicness}
        predictions_bw = {"dynamicness": pred_bw.dynamicness}
        predictions = {"fw": predictions_fw, "bw": predictions_bw}
        valid_masks = {"fw": valid_mask_pc1, "bw": valid_mask_pc2}

        epes_stat_flow = []
        epes_dyn_flow = []
        dynamicness_scores = []
        for flowdir, knn_results_dir in zip(
            ["fw", "bw"], [knn_results_fw, knn_results_bw]
        ):
            epes_stat_flow.append(
                knn_results_dir["static_aggr"]["knn"]["nearest_dist"][
                    valid_masks[flowdir]
                ].flatten(),
            )
            epes_dyn_flow.append(
                knn_results_dir["dynamic"]["knn"]["nearest_dist"][
                    valid_masks[flowdir]
                ].flatten(),
            )
            dynamicness_scores.append(
                predictions[flowdir]["dynamicness"][valid_masks[flowdir]].flatten()
            )

        moving_thresh_module.update(
            epes_stat_flow=torch.cat(epes_stat_flow, dim=0),
            epes_dyn_flow=torch.cat(epes_dyn_flow, dim=0),
            moving_mask=None,
            dynamicness_scores=torch.cat(dynamicness_scores, dim=0),
            training=training,
        )
        metrics_collector["moving_thresh_module.value"] = (
            moving_thresh_module.value().detach().cpu().numpy()
        )
    # #endregion update dynamicness threshold

    if loss_cfg.artificial_labels.cross_entropy_penalty > 0.0:
        total_loss = (
            total_loss
            + 0.5
            * (ce_loss_fw + ce_loss_bw)
            * loss_cfg.artificial_labels.cross_entropy_penalty
        )
    metrics_collector["total_loss"] = total_loss.detach().cpu().numpy()

    return total_loss


@torch.no_grad()
def get_range_based_knn_loss_weights(
    pc1: torch.FloatTensor, valid_mask_pc1: torch.BoolTensor, loss_cfg: Dict[str, float]
):
    range_m_pc1 = torch.linalg.norm(pc1[..., :3], axis=-1)
    fw_weight = (
        loss_cfg.knn_loss.range_based_weights.slope_sign
        * loss_cfg.knn_loss.range_based_weights.weight_slope
    ) * range_m_pc1 + loss_cfg.knn_loss.range_based_weights.weight_at_range_0
    fw_knn_weights = torch.clip(
        fw_weight,
        min=loss_cfg.knn_loss.range_based_weights.min_weight_clip_at,
        max=loss_cfg.knn_loss.range_based_weights.max_weight_clip_at,
    )
    weight_sum = fw_knn_weights[valid_mask_pc1].sum()
    weight_target_sum = torch.count_nonzero(valid_mask_pc1)
    fw_knn_weights = fw_knn_weights * weight_target_sum / weight_sum
    return fw_knn_weights


def squared_sum(delta, axis: int = -1):
    return torch.sum(torch.square(delta), dim=axis)
