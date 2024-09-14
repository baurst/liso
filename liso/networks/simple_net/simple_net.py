from functools import lru_cache
from logging import Logger, getLogger
from typing import Dict, Tuple

import numpy as np
import torch
from liso.datasets.torch_dataset_commons import get_centermaps_output_grid_size
from liso.kabsch.output_modification import (
    maybe_flatten_anchors_except_for,
    output_modification,
)
from liso.kabsch.shape_utils import Shape
from liso.networks.flow_cluster_detector.flow_cluster_detector import (
    FlowClusterDetector,
)
from liso.networks.simple_net.centerpoint_net import CenterPointStyleNet
from liso.networks.simple_net.point_rcnn import PointRCNNWrapper
from liso.networks.simple_net.pointpillars import PointPillarsWrapper
from liso.networks.simple_net.simple_net_utils import allowed_activations
from liso.networks.simple_net.transfusion_net import TransfusionStyleNet
from liso.utils.bev_utils import get_metric_voxel_center_coords


@lru_cache(10)
def warn_once(logger: Logger, msg: str):
    logger.warning(msg)


class BoxLearner(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.activations = {
            k: allowed_activations[v]
            for k, v in self.cfg.box_prediction.activations.items()
        }

        if cfg.network.name == "centerpoint":
            self.model = CenterPointStyleNet(cfg)
        elif cfg.network.name == "transfusion":
            self.model = TransfusionStyleNet(cfg)
        elif cfg.network.name == "pointpillars":
            self.model = PointPillarsWrapper(cfg)
        elif cfg.network.name == "pointrcnn":
            self.model = PointRCNNWrapper(cfg)
        elif cfg.network.name == "echo_gt":
            self.model = None
        else:
            raise NotImplementedError(cfg.network.name)

        pred_bev_maps_shape = get_centermaps_output_grid_size(
            self.cfg, np.array(self.cfg.data.img_grid_size)
        )
        if pred_bev_maps_shape is not None:
            pred_bev_maps_shape = torch.tensor(pred_bev_maps_shape)
            self.pillar_center_coors_m = torch.nn.parameter.Parameter(
                torch.from_numpy(
                    get_metric_voxel_center_coords(
                        bev_range_x=self.cfg.data.bev_range_m[0],
                        bev_range_y=self.cfg.data.bev_range_m[1],
                        dataset_img_shape=pred_bev_maps_shape,
                    ).astype(np.float32)[..., 0:2]
                ),
                requires_grad=False,
            )

    def forward(
        self,
        img_t0,
        pcls_t0,
        gt_boxes=None,
        centermaps_gt=None,
        train=True,
    ) -> Tuple[Shape, Dict]:
        if self.cfg.network.name == "echo_gt":
            warn_once(getLogger(), "WARNING: NETWORK ECHOES GROUNDTRUTH")
            assert gt_boxes is not None

            return gt_boxes.clone(), gt_boxes.clone().__dict__, None
        if img_t0 is not None:
            assert (
                len(img_t0.shape) == 4
            ), f"need batched gray image but got shape {img_t0.shape}"
        if isinstance(self.model, (PointRCNNWrapper, PointPillarsWrapper)):
            if train:
                loss_dict = self.model(pcls=pcls_t0, gt_boxes=gt_boxes, train=train)
                return loss_dict
            else:
                pred_boxes = self.model(pcls=pcls_t0, gt_boxes=gt_boxes, train=False)
                return (pred_boxes, None, None, None)
        else:
            raw_box_vars, aux_outputs = self.model(img_t0, pcls_t0)

        decoded_box_maps, activated_box_maps = self.apply_all_output_modifications(
            raw_box_vars=raw_box_vars, gt_boxes=gt_boxes, centermaps_gt=centermaps_gt
        )

        flat_boxes = maybe_flatten_anchors_except_for(
            {k: v.clone() for k, v in decoded_box_maps.items()}, ()
        )
        return (
            Shape(**flat_boxes),
            decoded_box_maps,
            activated_box_maps,
            aux_outputs,
        )

    def apply_all_output_modifications(
        self,
        *,
        raw_box_vars: Dict[str, torch.FloatTensor],
        gt_boxes=None,
        centermaps_gt=None,
    ):
        shape_vars_activated = {
            param_name: self.activations[param_name](box_tensor)
            for param_name, box_tensor in raw_box_vars.items()
        }

        # for box_attr_name, box_attr_val in shape_vars_activated.items():
        #     if (
        #         box_attr_val is not None
        #         and self.cfg.box_prediction.gradient_scaling[box_attr_name] != 1
        #     ):
        #         grad_scale = self.cfg.box_prediction.gradient_scaling[box_attr_name]
        #         assert grad_scale >= 0.0, grad_scale
        #         shape_vars_activated[
        #             box_attr_name
        #         ] = box_attr_val * grad_scale - box_attr_val.detach() * (grad_scale - 1)
        #     else:
        #         shape_vars_activated[box_attr_name] = box_attr_val

        # box_vars_grad_scaled = {k: v.clone() for k, v in shape_vars_activated.items()}
        box_vars_for_mod = {k: v.clone() for k, v in shape_vars_activated.items()}
        box_vars_decoded = output_modification(
            box_vars_for_mod,
            self.cfg.box_prediction,
            self.cfg.data,
            self.cfg.data.shapes.name,
            self.pillar_center_coors_m,
        )
        need_gt_replacement = np.any(
            ["gt" in v for _, v in self.cfg.box_prediction.output_modification.items()]
        )
        if need_gt_replacement:
            raise AssertionError("shouldnt be used nymore")

        return box_vars_decoded, shape_vars_activated


def select_network(cfg, device):
    if cfg.network.name in (
        "transfusion",
        "slot",
        "centerpoint",
        "echo_gt",
        "pointrcnn",
        "pointpillars",
    ):
        box_predictor = BoxLearner(
            cfg,
        ).to(device)
    elif cfg.network.name == "flow_cluster_detector":
        box_predictor = FlowClusterDetector(cfg)
    else:
        raise NotImplementedError(cfg.network.name)
    return box_predictor
