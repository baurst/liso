from collections import OrderedDict
from pathlib import Path
from typing import Dict

import torch
from config_helper.config import dumb_load_yaml_to_omegaconf

allowed_activations = {
    "none": lambda x: x,
    "softplus": torch.nn.functional.softplus,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "exp": torch.exp,
}


def get_num_dims_per_box_attr(cfg):
    num_box_rot_attributes = {
        "direct": 1,
        "vector": 2,
        "none": 0,
        "class_bins": 36,
    }[cfg.box_prediction.rotation_representation.method]
    num_box_dim_attributes = {
        "predict_aspect_ratio": 2,
        "predict_abs_size": 3,
        "predict_log_size": 3,
    }[cfg.box_prediction.dimensions_representation.method]
    dims_per_box_attr = OrderedDict(
        zip(
            ("pos", "dims", "rot", "probs"),
            (
                cfg.box_prediction.position_representation.num_box_pos_dims,
                num_box_dim_attributes,
                num_box_rot_attributes,
                1,
            ),
        ),
    )
    return dims_per_box_attr


def load_checkpoint_check_sanity(
    path_to_checkpoint: str,
    cfg: Dict,
    box_predictor,
):
    old_cfg_path = Path(path_to_checkpoint).parent.parent.joinpath("config.yml")

    old_cfg = dumb_load_yaml_to_omegaconf(old_cfg_path)
    must_be_equal = (
        "rotation_representation",
        "position_representation",
        "activations",
        "dimensions_representation",
    )
    for mbe in must_be_equal:
        for el in cfg.box_prediction[mbe]:
            if el in old_cfg.box_prediction[mbe]:
                if mbe == "activations" and el == "probs":
                    assert old_cfg.box_prediction[mbe][el] == "none"
                else:
                    assert (
                        cfg.box_prediction[mbe][el] == old_cfg.box_prediction[mbe][el]
                    ), f"critical diff detected for key {mbe}/{el} \n Was: {old_cfg.box_prediction[mbe][el]}\n Need: {cfg.box_prediction[mbe][el]}"

    checkpoint_content = torch.load(path_to_checkpoint)
    if "network" in checkpoint_content:
        # new method
        box_predictor.load_state_dict(checkpoint_content["network"])
    else:
        # legacy for old checkpoints, where we only had network checkpoints
        box_predictor.load_state_dict(checkpoint_content)
    print(f"Loaded model weights from {path_to_checkpoint}")
    return box_predictor
