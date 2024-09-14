import torch
from liso.utils.torch_differentiable_forward_scatter import masked_scatter_mean_2d
from liso.utils.torch_transformation import homogenize_pcl


def get_bev_dynamic_flow_map_from_pcl_flow_and_odom(
    *,
    pcl_is_valid,
    pcl,
    pillar_coors,
    point_flow,
    odom_ta_tb,
    target_shape,
    return_nonrigid_bev_flow=False,
):
    pcl_homog_no_nans = torch.where(
        pcl_is_valid[..., None],
        homogenize_pcl(pcl[..., :3], pcl_is_valid),
        torch.zeros(1, device=pcl.device),
    )

    flow_t0_t1_no_nans = torch.where(
        pcl_is_valid[..., None],
        point_flow,
        torch.zeros(1, device=point_flow.device),
    )

    flow_stat_aggr_t0_t1 = torch.einsum(
        "bij,bnj->bni",
        torch.linalg.inv(odom_ta_tb)
        - torch.eye(4, device=odom_ta_tb.device, dtype=torch.float64)[None, ...],
        pcl_homog_no_nans.double(),
    )[..., :3].float()

    flow_stat_aggr_t0_t1 = torch.where(
        pcl_is_valid[..., None],
        flow_stat_aggr_t0_t1,
        torch.zeros(1, device=pcl.device),
    )
    nonrigid_flow = flow_t0_t1_no_nans - flow_stat_aggr_t0_t1
    residual_flow_length = torch.linalg.norm(nonrigid_flow, dim=-1, keepdim=True)
    h, w = target_shape
    bev_dynamicness = torch.zeros(
        (pcl_is_valid.shape[0], h, w, 1),
        device=pcl.device,
        dtype=torch.float32,
    )
    per_point_batch_coors = torch.arange(
        pcl_homog_no_nans.shape[0],
        device=pcl_homog_no_nans.device,
    )[:, None].repeat((1,) + pillar_coors.shape[1:-1])
    bev_dynamicness = masked_scatter_mean_2d(
        bev_dynamicness,
        pcl_is_valid,
        per_point_batch_coors,
        pillar_coors[..., 0],
        pillar_coors[..., 1],
        update_values=residual_flow_length,
    )

    if return_nonrigid_bev_flow:
        bev_nonrigid_flow = torch.zeros(
            (pcl_is_valid.shape[0], h, w, 3),
            device=pcl.device,
            dtype=torch.float32,
        )
        bev_nonrigid_flow = masked_scatter_mean_2d(
            bev_nonrigid_flow,
            pcl_is_valid,
            per_point_batch_coors,
            pillar_coors[..., 0],
            pillar_coors[..., 1],
            update_values=nonrigid_flow,
        )
        return bev_dynamicness, bev_nonrigid_flow

    return bev_dynamicness
