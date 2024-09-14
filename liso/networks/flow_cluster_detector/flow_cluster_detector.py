import numpy as np
import torch
from liso.kabsch.kabsch_mask import KabschDecoder
from liso.kabsch.shape_utils import Shape, extract_motion_in_pred_box_coordinates
from liso.utils.bev_flow_utils import get_bev_dynamic_flow_map_from_pcl_flow_and_odom
from liso.utils.bev_utils import get_bev_setup_params
from liso.visu.flow_image import log_flow_image
from skimage import segmentation
from skimage.measure import regionprops
from sklearn.cluster import DBSCAN
from torch.utils.tensorboard.writer import SummaryWriter


class FlowClusterDetector(torch.nn.Module):
    def __init__(
        self,
        cfg,
        min_num_pts_per_box=10,
        max_box_len_m=7.0,
        aspect_ratio_max=4.0,
        min_box_area_m2=0.35,  # area of a pedestrian
        min_box_volume_m3=0.5,  # volume of a pedestrian
    ) -> None:
        super().__init__()
        if (
            hasattr(
                cfg.data.tracking_cfg,
                "flow_cluster_detector_ignore_min_box_size_limits",
            )
            and cfg.data.tracking_cfg.flow_cluster_detector_ignore_min_box_size_limits
        ):
            print("Ignoring min box size limits for flow cluster detector!")
            min_box_area_m2 = 0.0
            min_box_volume_m3 = 0.0
        if (
            hasattr(
                cfg.data.tracking_cfg,
                "flow_cluster_detector_ignore_max_box_size_limits",
            )
            and cfg.data.tracking_cfg.flow_cluster_detector_ignore_max_box_size_limits
        ):
            # min_num_pts_per_box = 0
            print("Ignoring max box size limits for flow cluster detector!")
            aspect_ratio_max = 1000.0
            max_box_len_m = 1000.0

        self.min_box_area_m2 = min_box_area_m2  # area of a pedestrian
        self.min_box_volume_m3 = min_box_volume_m3  # volume of a pedestrian
        self.min_num_pts_per_box = min_num_pts_per_box
        self.aspect_ratio_max = aspect_ratio_max
        self.max_box_len_m = max_box_len_m
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
        self.min_residual_flow_thresh_mps = 1.0  # 0.1m disp for 100ms
        # information should be propagated for maximum half a meter
        self.grid_pts_3d = (
            self.pcl_bev_center_coords_homog[..., 0:2].detach().cpu().numpy()
        )
        self.pillar_bev_coors = np.stack(
            np.meshgrid(
                np.arange(self.grid_pts_3d.shape[0]),
                np.arange(self.grid_pts_3d.shape[1]),
                indexing="ij",
            ),
            axis=-1,
        )
        self.bev_img_grid_size = np.array(self.cfg.data.img_grid_size)

        self.kabsch_decoder = KabschDecoder(cfg)

    def forward(
        self,
        sample_data_ta: torch.FloatTensor,
        writer: SummaryWriter = None,
        writer_prefix: str = "",
        global_step: int = None,
        is_batched=True,
    ) -> Shape:
        pcl = sample_data_ta["pcl_ta"]["pcl"]
        pcl_full_w_ground_for_box_fitting = sample_data_ta["pcl_full_w_ground_ta"]
        pillar_coors = sample_data_ta["pcl_ta"]["pillar_coors"]
        point_flow = sample_data_ta[self.cfg.data.flow_source]["flow_ta_tb"]
        odom_ta_tb = sample_data_ta[self.cfg.data.odom_source]["odom_ta_tb"]

        nonrigid_bev_flow_threshold = (
            sample_data_ta["src_trgt_time_delta_s"] * self.min_residual_flow_thresh_mps
        )
        if is_batched:
            pcl_is_valid = sample_data_ta["pcl_ta"]["pcl_is_valid"]
        else:
            pcl_is_valid = torch.ones_like(pcl[:, 0], dtype=bool)[None, ...]
            pcl = pcl[None]
            pcl_full_w_ground_for_box_fitting = pcl_full_w_ground_for_box_fitting[None]
            pillar_coors = pillar_coors[None]
            point_flow = point_flow[None]
            odom_ta_tb = odom_ta_tb[None]
            nonrigid_bev_flow_threshold = nonrigid_bev_flow_threshold[None]

        (
            bev_dynamicness,
            bev_nonrigid_flow,
        ) = get_bev_dynamic_flow_map_from_pcl_flow_and_odom(
            pcl_is_valid=pcl_is_valid,
            pcl=pcl,
            pillar_coors=pillar_coors,
            point_flow=point_flow,
            odom_ta_tb=odom_ta_tb,
            target_shape=self.bev_img_grid_size,
            return_nonrigid_bev_flow=True,
        )
        bev_dynamicness_u8_cpu_npy = (
            (bev_dynamicness * 255.0).to(torch.uint8).cpu()
        ).numpy()
        valid_mask_cpu_npy = (
            (
                torch.squeeze(bev_dynamicness, dim=-1)
                > nonrigid_bev_flow_threshold[..., None, None]
            ).cpu()
        ).numpy()

        assert len(valid_mask_cpu_npy.shape) == 3, valid_mask_cpu_npy.shape
        # num_clusters = 20

        slic_img_segments = []
        batched_pred_boxes = []

        # flow_rgb = (
        #     pytorch_create_flow_image(bev_nonrigid_flow[..., :2].permute(0, 3, 1, 2))
        #     .permute(0, 2, 3, 1)
        #     .cpu()
        #     .numpy()
        #     * 255
        # ).astype(np.uint8)
        # Image.fromarray(flow_rgb[batch_idx]).save("debug_imgs/nonrigid_flow.png")
        box_target_device = pcl.device
        for batch_idx, valid_mask in enumerate(valid_mask_cpu_npy):
            if np.count_nonzero(valid_mask) > 1:
                flow_similarity_importance = (
                    2.0  # problem: dynamic flow is not smooth, should be Kabsched!
                )
                dynamic_coors = self.grid_pts_3d[valid_mask]
                dynamic_flow = flow_similarity_importance * (
                    bev_nonrigid_flow[batch_idx].detach().cpu().numpy()[valid_mask]
                )
                cluster_coords = np.concatenate([dynamic_coors, dynamic_flow], axis=-1)
                db = DBSCAN(
                    eps=1.0,
                    min_samples=5,
                    metric="euclidean",
                    algorithm="auto",
                    n_jobs=1,
                ).fit(cluster_coords)
                labels = db.labels_
                labels = np.where(labels >= 0, labels + 1, 0)
                bev_labels = np.zeros_like(valid_mask, dtype=labels.dtype)
                pillar_coors = self.pillar_bev_coors[valid_mask]
                bev_labels[pillar_coors[..., 0], pillar_coors[..., 1]] = labels
                slic_img_segments.append(bev_labels)

                clustered_regions = regionprops(bev_labels)
                box_center_pix = np.clip(
                    np.array([el.centroid for el in clustered_regions]).astype(np.int),
                    a_min=0,
                    a_max=min(self.grid_pts_3d.shape[:-1]) - 1,
                )
                rot = torch.from_numpy(
                    np.array([el.orientation for el in clustered_regions])[..., None]
                )
                box_len = np.array([el.axis_major_length for el in clustered_regions])[
                    ..., None
                ]
                box_width = np.array(
                    [el.axis_minor_length for el in clustered_regions]
                )[..., None]
                box_dims = (
                    torch.from_numpy(np.concatenate([box_len, box_width], axis=-1))
                    * 1.0
                    / self.bev_pixel_per_meter_resolution
                )
                if box_center_pix.size > 0:
                    box_center_m = torch.from_numpy(
                        self.grid_pts_3d[box_center_pix[..., 0], box_center_pix[..., 1]]
                    )
                else:
                    box_center_m = torch.zeros_like(box_dims)
                probs = torch.ones_like(rot)
                pred_boxes = Shape(
                    pos=box_center_m,
                    dims=box_dims,
                    rot=rot,
                    probs=probs,
                ).to(box_target_device)
                assert (
                    pred_boxes.dims.shape[-1] == 2
                ), "otherwise box fitting will use bad box size from clustering!"
                (
                    num_pts_in_box,
                    fitted_box_z,
                    fitted_box_height,
                ) = fit_bev_box_z_and_height_using_points_in_box(
                    pcl_full_w_ground_for_box_fitting[batch_idx][:, :3],
                    pred_boxes,
                    box_height=1000.0,
                )
                box_has_enough_points = num_pts_in_box >= self.min_num_pts_per_box
                aspect_ratio = pred_boxes.dims[..., 0] / torch.max(
                    pred_boxes.dims[..., 1],
                    0.001 * torch.ones_like(pred_boxes.dims[..., 1]),
                )
                box_aspect_ratio_is_acceptable = aspect_ratio <= self.aspect_ratio_max
                box_is_not_too_large = pred_boxes.dims[..., 0] <= self.max_box_len_m
                box_footprint_is_not_too_small = (
                    torch.prod(pred_boxes.dims[..., :2], dim=-1) > self.min_box_area_m2
                )
                pred_boxes.dims = torch.cat(
                    [pred_boxes.dims, fitted_box_height[..., None]], axis=-1
                )
                pred_boxes.pos = torch.cat(
                    [pred_boxes.pos, fitted_box_z[..., None]], axis=-1
                )
                box_volume_is_not_too_small = (
                    torch.prod(pred_boxes.dims[..., :3], dim=-1)
                    > self.min_box_volume_m3
                )
                is_valid_box = (
                    box_has_enough_points
                    & box_aspect_ratio_is_acceptable
                    & box_is_not_too_large
                    & box_footprint_is_not_too_small
                    & box_volume_is_not_too_small
                )
                pred_boxes.valid = is_valid_box

                pred_boxes = pred_boxes.drop_padding_boxes()
                batched_pred_boxes.append(pred_boxes)

            else:
                batched_pred_boxes.append(
                    Shape.createEmpty().to_tensor().to(box_target_device)
                )
                slic_img_segments.append(None)
        if (global_step % self.cfg.logging.img_log_interval) == 0:
            visu_superpixels = []
            for batch_idx in range(
                min(len(slic_img_segments), self.cfg.logging.max_log_img_batches)
            ):
                if slic_img_segments[batch_idx] is not None:
                    visu_superpixels.append(
                        segmentation.mark_boundaries(
                            np.tile(
                                bev_dynamicness_u8_cpu_npy[batch_idx],
                                (1, 1, 3),
                            ),
                            slic_img_segments[batch_idx],
                            outline_color=None,
                            mode="outer",
                            background_label=0,
                        )
                    )
            if len(visu_superpixels) > 0:
                visu_superpixels = np.stack(visu_superpixels, axis=0)
                writer.add_images(
                    writer_prefix + "/clustering/superpixels",
                    visu_superpixels,
                    global_step=global_step,
                    dataformats="NHWC",
                )

            log_flow_image(
                cfg=self.cfg,
                writer=writer,
                global_step=global_step,
                flow_2d=bev_nonrigid_flow[:, :, :, :2].permute((0, 3, 1, 2)),
                prefix=writer_prefix + "/clustering/nonrigid_flow",
            )
            for pcl_id in ("pcl_ta", "pcl_tb"):
                black_canvas = torch.zeros(
                    tuple(self.cfg.data.img_grid_size) + (3,),
                    dtype=torch.float32,
                    device=bev_nonrigid_flow.device,
                )
                pillar_coors = sample_data_ta[pcl_id]["pillar_coors"].to(torch.long)
                black_canvas[
                    pillar_coors[0, ..., 0],
                    pillar_coors[0, ..., 1],
                ] = torch.ones((3,), device=black_canvas.device)
                writer.add_image(
                    writer_prefix + f"/clustering/{pcl_id}",
                    (255 * black_canvas.cpu().numpy()).astype(np.uint8),
                    global_step=global_step,
                    dataformats="HWC",
                )

        boxes = Shape.from_list_of_shapes(batched_pred_boxes, numeric_padding_value=0.0)
        # adapt the rotation of the box to the direction of the flow
        (
            fg_kabsch_trafos,
            _,
            _,
            bg_kabsch_trafo,
            _,
        ) = self.kabsch_decoder.get_kabsch_trafos_from_point_flow(
            point_cloud_ta=pcl[..., :3],
            valid_mask_ta=pcl_is_valid,
            pointwise_flow_ta_tb=point_flow,
            pred_boxes_ta=boxes,
        )

        box_translation, _ = extract_motion_in_pred_box_coordinates(
            boxes, fg_kabsch_trafos, bg_kabsch_trafo
        )
        delta_angle = torch.atan2(box_translation[..., [1]], box_translation[..., [0]])
        box_velo = torch.zeros_like(boxes.probs)
        box_velo[..., 0] = torch.linalg.norm(box_translation, dim=-1)
        boxes.rot = boxes.rot + delta_angle
        boxes.velo = box_velo
        if not is_batched:
            boxes = boxes[0]
            assert len(boxes.shape) == 1, boxes.shape
        return boxes


@torch.no_grad()
def fit_bev_box_z_and_height_using_points_in_box(pcl, boxes: Shape, box_height=1000.0):
    assert len(pcl.shape) == 2, pcl.shape
    assert len(boxes.pos.shape) == 2, boxes.print_attr_shapes()
    sensor_T_box = boxes.get_poses()
    homog_pcl = torch.cat([pcl, torch.ones_like(pcl[:, :1])], dim=-1)

    if boxes.dims.shape[-1] == 2:
        box_dims = torch.cat(
            [boxes.dims, box_height * torch.ones_like(boxes.dims[:, :1])], dim=-1
        )
    else:
        assert boxes.dims.shape[-1] == 3, boxes.dims.shape
        box_dims = boxes.dims
    pts_homog_in_box = torch.einsum(
        "kij,nj->nki", torch.linalg.inv(sensor_T_box), homog_pcl.to(torch.double)
    ).to(torch.float)
    assert torch.all(torch.isfinite(pts_homog_in_box))
    pt_is_in_box = torch.all(
        torch.abs(pts_homog_in_box[..., 0:3]) < 0.5 * box_dims[None, ...], dim=-1
    )
    z_coords = pts_homog_in_box[..., 2]
    dummy_box_height = torch.tensor(box_height).to(
        dtype=z_coords.dtype, device=z_coords.device
    )
    z_coords_max = (
        torch.where(pt_is_in_box, z_coords, -dummy_box_height).max(dim=0).values
    )
    z_coords_min = torch.where(pt_is_in_box, z_coords, +dummy_box_height).min(dim=0)
    z_coords_min_vals = z_coords_min.values
    fitted_box_height = z_coords_max - z_coords_min_vals
    fitted_box_height = torch.clip(fitted_box_height, min=1.0, max=2.0)
    z_coords_min_idxs = z_coords_min.indices
    # take the lowest point in box (sensor coordinates) and add half box_height
    fitted_box_z = homog_pcl[..., 2][z_coords_min_idxs] + 0.5 * fitted_box_height
    num_pts_in_box = torch.sum(pt_is_in_box, dim=0)
    assert torch.all(torch.isfinite(fitted_box_z))
    # fitted_box_z = torch.where(
    #     num_pts_in_box > 0, fitted_box_z, torch.tensor(np.nan).to(fitted_box_z.device)
    # )
    # fitted_box_height = torch.where(
    #     num_pts_in_box > 0,
    #     fitted_box_height,
    #     torch.tensor(np.nan).to(fitted_box_height.device),
    # )
    return num_pts_in_box, fitted_box_z, fitted_box_height
