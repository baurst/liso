import numpy as np
import torch
from liso.networks.pcl_to_feature_grid.pcl_to_feature_grid import (
    PointsPillarFeatureNetWrapper,
)
from liso.slim.model.extractor import SmallEncoder

# from liso.slim.model.point_pillars import PointPillarsLayer
from liso.slim.model.raft_code.corr import CorrBlock
from liso.slim.model.raft_code.utils import initialize_flow, upflow_n, uplogits_n
from liso.slim.model.update import SmallUpdateBlock
from torch import nn


def move_channel_to_last_dim(tensor):
    return tensor.permute(0, 2, 3, 1)


class RAFT(nn.Module):
    def __init__(self, cfg, head_decoder_fw, head_decoder_bw, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.slim_cfg = cfg.SLIM
        bev_pc_range_half = 0.5 * np.array(self.cfg.data.bev_range_m)  # SYMMETRIC_BEV
        bev_pc_range = np.concatenate([-bev_pc_range_half, bev_pc_range_half], axis=0)
        self.head_decoder_fw = head_decoder_fw
        self.head_decoder_bw = head_decoder_bw

        self.iters = self.slim_cfg.model.num_iters

        self.bev_rows_res_meters_per_fs_pixel = (
            (bev_pc_range[2] - bev_pc_range[0])
            / self.cfg.data.img_grid_size[0]
            * self.slim_cfg.model.u_net.final_scale
        )

        self.bev_cols_res_meters_per_fs_pixel = (
            (bev_pc_range[3] - bev_pc_range[1])
            / self.cfg.data.img_grid_size[1]
            * self.slim_cfg.model.u_net.final_scale
        )
        assert (
            self.bev_rows_res_meters_per_fs_pixel
            == self.bev_cols_res_meters_per_fs_pixel
        ), "resolutions are different, but I am unsure if they are applied correctly in this case cause of interpretion switch in raft"
        self.pp_layer = PointsPillarFeatureNetWrapper(cfg)
        # self.pp_layer = PointPillarsLayer(
        #     self.cfg.model.point_pillars,
        #     bn_kwargs=dict(self.cfg.layers.batch_normalization),
        # )

        self.hidden_dim = hdim = 96
        self.context_dim = cdim = 64
        feat_for_corr_dim = 128

        if self.slim_cfg.model.corr_cfg.module == "all":
            assert (
                self.slim_cfg.model.feature_downsampling_factor == 8
            ), "you cannot use default CorrBlock without default resolution"
        else:
            raise ValueError(
                "Don't know what to do with correlation module specification {0}".format(
                    self.slim_cfg.model.corr_cfg.module
                )
            )

        self.fnet = SmallEncoder(
            output_dim=feat_for_corr_dim,
            norm_fn=self.slim_cfg.model.raft_fnet_norm,
            dropout=self.slim_cfg.model.dropout_rate,
        )
        self.cnet = SmallEncoder(
            output_dim=hdim + cdim,
            norm_fn="none",
            dropout=self.slim_cfg.model.dropout_rate,
        )
        self.update_block = SmallUpdateBlock(
            cfg=self.slim_cfg,
            filters=hdim,
        )

    def forward(
        self,
        pcl_t0,
        pcl_t1,
    ):
        img_t0, bev_occupancy_map_t0 = self.pp_layer(  # bev_enc_0
            pcl_t0,
        )

        img_t1, bev_occupancy_map_t1 = self.pp_layer(  # bev_enc_1
            pcl_t1,
        )

        aux_outputs = {
            "t0": {"bev_net_input_dbg": bev_occupancy_map_t0},
            "t1": {"bev_net_input_dbg": bev_occupancy_map_t1},
        }

        # feature extractor -> (bs, nch, h/8, w/8)x2
        fmap_t0 = self.fnet(img_t0)

        fmap_t1 = self.fnet(img_t1)  # nch: 128

        assert self.slim_cfg.model.flow_maps_archi in [
            "single",
            "vanilla",
        ], "David did only check this branch because he thinks others are unused"

        retvals_fw = self.predict_single_flow_map_and_classes(
            img_t0,
            fmap_t0,
            fmap_t1,
            self.head_decoder_fw,  # training
        )
        retvals_bw = self.predict_single_flow_map_and_classes(
            img_t1,
            fmap_t1,
            fmap_t0,
            self.head_decoder_bw,  # training
        )
        return retvals_fw, retvals_bw, aux_outputs

    def predict_single_flow_map_and_classes(
        self,
        img_t0,
        fmap_t0,
        fmap_t1,
        decoder,  # training
    ):
        assert (
            img_t0.shape[1] == self.slim_cfg.model.point_pillars.nbr_point_feats
        ), img_t0.shape
        # coordiantes have behavior [:, :, w, h] = [h, w] where img_t0 shape is [B, C, H, W]
        # pixel_coords_t0[0,...,10,20] -> tensor([20., 10.], device='cuda:0')
        pixel_coords_t0 = initialize_flow(
            img_t0, downscale_factor=self.slim_cfg.model.feature_downsampling_factor
        )
        pixel_coords_t1 = initialize_flow(
            img_t0, downscale_factor=self.slim_cfg.model.feature_downsampling_factor
        )

        b, _, h, w = pixel_coords_t0.shape
        if self.slim_cfg.model.flow_maps_archi == "vanilla":
            logits = None
        else:
            logits = torch.zeros(
                (b, 4, h, w), dtype=torch.float32, device=img_t0.device
            )
        if self.slim_cfg.model.predict_weight_for_static_aggregation is not False:
            assert self.slim_cfg.model.flow_maps_archi != "vanilla"
            weight_logits_for_static_aggregation = torch.zeros(
                (b, 1, h, w), dtype=torch.float32, device=img_t0.device
            )
        else:
            weight_logits_for_static_aggregation = None

        # setup correlation values
        if self.slim_cfg.model.corr_cfg.module == "all":
            correlation = CorrBlock(
                fmap_t0,
                fmap_t1,
                num_levels=self.slim_cfg.model.corr_cfg.num_levels,
                radius=self.slim_cfg.model.corr_cfg.search_radius,
            )
        else:
            raise ValueError("Wrong corr module selected")

        # context network
        cnet = self.cnet(img_t0)
        net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        iters = (
            self.slim_cfg.model.num_iters
        )  # if training else self.slim_cfg.model.num_pred_iters

        flow_resolution_adapter = torch.tensor(
            [
                self.bev_rows_res_meters_per_fs_pixel,  # x-resolution of bev map
                self.bev_cols_res_meters_per_fs_pixel,  # y-resolution of bev map
            ],
            device=inp.device,
            dtype=inp.dtype,
        )[None, ..., None, None]
        intermediate_flow_predictions = []
        for _i in range(iters):
            pixel_coords_t1 = pixel_coords_t1.detach()
            if self.slim_cfg.model.flow_maps_archi != "vanilla":
                logits = logits.detach()
            if self.slim_cfg.model.predict_weight_for_static_aggregation is not False:
                weight_logits_for_static_aggregation = (
                    weight_logits_for_static_aggregation.detach()
                )
            corr = correlation(pixel_coords_t1)

            flow = pixel_coords_t1 - pixel_coords_t0
            (
                net,
                delta_flow,
                delta_logits,
                delta_weight_logits_for_static_aggr,
            ) = self.update_block(
                net, inp, corr, flow, logits, weight_logits_for_static_aggregation
            )

            pixel_coords_t1 = pixel_coords_t1 + delta_flow
            if self.slim_cfg.model.flow_maps_archi != "vanilla":
                logits = logits + delta_logits
            if self.slim_cfg.model.predict_weight_for_static_aggregation is not False:
                weight_logits_for_static_aggregation += (
                    delta_weight_logits_for_static_aggr
                )

            upsampled_flow_usfl_convention = change_flow_convention_from_raft2usfl(
                upflow_n(
                    pixel_coords_t1 - pixel_coords_t0,
                    n=self.slim_cfg.model.feature_downsampling_factor,
                ),
                resolution_adapter=flow_resolution_adapter,
            )

            if self.slim_cfg.model.flow_maps_archi == "vanilla":
                assert logits is None, logits.shape
                upsampled_logits = torch.zeros(
                    (
                        b,
                        4,
                        h * self.slim_cfg.model.feature_downsampling_factor,
                        w * self.slim_cfg.model.feature_downsampling_factor,
                    ),
                    dtype=torch.float32,
                    device=img_t0.device,
                )
            else:
                upsampled_logits = uplogits_n(
                    logits,
                    n=self.slim_cfg.model.feature_downsampling_factor,
                )

            if weight_logits_for_static_aggregation is not None:
                upsampled_weight_logits_for_static_aggregation = uplogits_n(
                    weight_logits_for_static_aggregation,
                    n=self.slim_cfg.model.feature_downsampling_factor,
                )
            else:
                upsampled_weight_logits_for_static_aggregation = None

            intermediate_flow_predictions.append(
                decoder.concat2network_output(
                    logits=upsampled_logits,
                    static_flow=upsampled_flow_usfl_convention,
                    dynamic_flow=upsampled_flow_usfl_convention,
                    weight_logits_for_static_aggregation=upsampled_weight_logits_for_static_aggregation,
                )
            )

        return intermediate_flow_predictions


def change_flow_convention_from_raft2usfl(
    flow: torch.FloatTensor, resolution_adapter: torch.FloatTensor
):
    flow_meters = torch.flip(flow, dims=[1]) * resolution_adapter
    return flow_meters


def main():
    import numpy as np

    img_t0 = torch.from_numpy(np.random.normal(size=(1, 64, 640, 640)))

    coords_t0 = initialize_flow(img_t0, downscale_factor=8)

    print(coords_t0[0, ..., 10, 20])

    coords_usfl_conv = torch.flip(coords_t0, dims=[1])

    print(coords_usfl_conv[0, ..., 10, 20])
    print(coords_usfl_conv[0, ..., 0, 79])
    print(coords_usfl_conv[0, ..., 79, 0])


if __name__ == "__main__":
    main()
