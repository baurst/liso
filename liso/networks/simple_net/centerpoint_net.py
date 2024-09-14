import torch
from liso.networks.centerpoint.center_head import CenterHead
from liso.networks.centerpoint.rpn import RPN
from liso.networks.pcl_to_feature_grid.pcl_to_feature_grid import (
    PointsPillarFeatureNetWrapper,
)
from liso.networks.simple_net.simple_net_utils import get_num_dims_per_box_attr


def is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)


class CenterPointStyleNet(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        channel_reduction_factor = cfg.network.centerpoint.setdefault(
            "channel_reduction_factor", 1
        )
        assert is_power_of_two(channel_reduction_factor), channel_reduction_factor
        rpn_conf = {
            "layer_nums": [3, 5],
            "ds_layer_strides": [2, 2],
            "ds_num_filters": [
                cfg.network.centerpoint.hid_dim // channel_reduction_factor,
                128 // channel_reduction_factor,
            ],
            "us_layer_strides": [0.5, 1],
            "us_num_filters": [
                128 // channel_reduction_factor,
                128 // channel_reduction_factor,
            ],
        }
        if cfg.network.centerpoint.reduce_receptive_field == 2:
            rpn_conf["ds_layer_strides"] = [1, 1]  # wont work
        elif cfg.network.centerpoint.reduce_receptive_field == 1:
            rpn_conf["ds_layer_strides"] = [1, 2]
        elif cfg.network.centerpoint.reduce_receptive_field == 0:
            pass
        else:
            raise NotImplementedError(cfg.network.centerpoint.reduce_receptive_field)

        if self.cfg.network.centerpoint.use_baseline_parameters:
            rpn_conf["layer_nums"].append(5)
            rpn_conf["ds_layer_strides"].append(2)
            rpn_conf["ds_num_filters"].append(256 // channel_reduction_factor)
            rpn_conf["us_layer_strides"].append(2)
            rpn_conf["us_num_filters"].append(128 // channel_reduction_factor)
            head_conf = {
                "stride": 1,
                "in_channels": sum(rpn_conf["us_num_filters"]),
            }
        else:
            head_conf = {
                "stride": 2,
                "in_channels": sum(rpn_conf["us_num_filters"]),
            }
        self.pfn = PointsPillarFeatureNetWrapper(cfg)
        self.rpn = RPN(
            **rpn_conf,
            num_input_features=cfg.network.centerpoint.hid_dim
            // channel_reduction_factor,
            norm_cfg=self.cfg.network.centerpoint.batch_norm.kwargs,
        )
        assert self.cfg.box_prediction.rotation_representation.method in (
            "vector",
            "class_bins",
        ), self.cfg.box_prediction.rotation_representation.method
        dims_per_box_attr = get_num_dims_per_box_attr(cfg)
        common_heads = {
            k: (v, 2)  # (output_channel, num_conv)
            for k, v in dims_per_box_attr.items()
        }
        self.center_head = CenterHead(
            **head_conf,
            common_heads=common_heads,
            norm_cfg=self.cfg.network.centerpoint.batch_norm.kwargs,
        )

        if self.cfg.loss.supervised.centermaps.active:
            assert (
                self.cfg.box_prediction.activations.probs == "none"
            ), self.cfg.box_prediction.activations.probs
            assert self.cfg.box_prediction.dimensions_representation.method in (
                "predict_abs_size",
                "predict_log_size",
            ), self.cfg.box_prediction.dimensions_representation.method
            if (
                self.cfg.box_prediction.dimensions_representation.method
                == "predict_log_size"
            ):
                assert (
                    self.cfg.box_prediction.activations.dims == "exp"
                ), self.cfg.box_prediction.activations.dims
            else:
                assert self.cfg.box_prediction.activations.dims in (
                    "softplus",
                    "sigmoid",
                ), self.cfg.box_prediction.activations.dims

    def forward(self, img_t0, pcls):
        if img_t0 is not None:
            assert (
                len(img_t0.shape) == 4
            ), f"need batched gray image but got shape {img_t0.shape}"
        bev_enc, bev_occupancy_map = self.pfn(pcl_t0=pcls, img_t0=img_t0)
        aux_outputs = {"bev_net_input_dbg": bev_occupancy_map}
        rpn = self.rpn(bev_enc)
        pred_dict = self.center_head(rpn)
        channels_last_dict = {k: v.permute(0, 2, 3, 1) for k, v in pred_dict.items()}
        return channels_last_dict, aux_outputs
