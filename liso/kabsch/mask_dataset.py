from typing import Tuple

import torch
from liso.kabsch.shape_utils import Shape


def recursive_device_mover(maybe_tensor, target_device):
    if torch.is_tensor(maybe_tensor) or isinstance(maybe_tensor, Shape):
        return maybe_tensor.to(target_device)
    elif isinstance(maybe_tensor, dict):
        return {
            k: recursive_device_mover(v, target_device) for k, v in maybe_tensor.items()
        }
    elif isinstance(maybe_tensor, tuple):
        return (recursive_device_mover(v, target_device) for v in maybe_tensor)
    elif isinstance(maybe_tensor, list):
        return [recursive_device_mover(v, target_device) for v in maybe_tensor]
    else:
        raise NotImplementedError(f"Unknown type {type(maybe_tensor)}")


def recursive_device_mover_for_specific_keys(
    maybe_tensor, target_device, keys_to_move: Tuple[str], key: str
):
    if (torch.is_tensor(maybe_tensor) or isinstance(maybe_tensor, Shape)) and (
        key not in keys_to_move  # and key != ""
    ):
        return maybe_tensor
    elif torch.is_tensor(maybe_tensor) or isinstance(maybe_tensor, Shape):
        return maybe_tensor.to(target_device)
    elif isinstance(maybe_tensor, dict):
        return {
            k: recursive_device_mover_for_specific_keys(
                v, target_device, keys_to_move, k
            )
            for k, v in maybe_tensor.items()
        }
    elif isinstance(maybe_tensor, tuple):
        return (
            recursive_device_mover_for_specific_keys(
                v, target_device, keys_to_move, key
            )
            for v in maybe_tensor
        )
    elif isinstance(maybe_tensor, list):
        return [
            recursive_device_mover_for_specific_keys(
                v, target_device, keys_to_move, key
            )
            for v in maybe_tensor
        ]
    else:
        raise NotImplementedError(f"Unknown type {type(maybe_tensor)}")


class RecursiveDeviceMover(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.dummy_device_indicator_param = torch.nn.Parameter(torch.empty(0))
        self.needed_on_gpu = [
            "pcl_full_ta",
            "boxes",
            "boxes_nusc",
            "mined",
            "centermaps_probs",
            "centermaps_dims",
            "centermaps_pos",
            "centermaps_rot",
            "centermaps_velo",
            "centermaps_center_bool_mask",
            "ignore_region_is_true_mask",
        ]

    def forward(
        self,
        data_input,
        need_sample_data_t0=True,
        need_sample_data_t1=True,
        need_augm_sample_data_t0=True,
    ):
        (
            dataset_element_t0,
            dataset_element_t1,
            augmented_dataset_element_t0,
            meta_data,
        ) = data_input
        if need_sample_data_t0:
            dataset_element_t0 = recursive_device_mover_for_specific_keys(
                dataset_element_t0,
                self.dummy_device_indicator_param.device,
                keys_to_move=self.needed_on_gpu,
                key="",
            )
        # mem_bytes = get_bytes(dataset_element_t0)
        # mb = mem_bytes * 1e-6
        # print("Moved ", {mb}, " mb.")
        if need_sample_data_t1:
            dataset_element_t1 = recursive_device_mover_for_specific_keys(
                dataset_element_t1,
                self.dummy_device_indicator_param.device,
                keys_to_move=self.needed_on_gpu,
                key="",
            )
        if need_augm_sample_data_t0:
            augmented_dataset_element_t0 = recursive_device_mover_for_specific_keys(
                augmented_dataset_element_t0,
                self.dummy_device_indicator_param.device,
                keys_to_move=self.needed_on_gpu,
                key="",
            )

        return (
            dataset_element_t0,
            dataset_element_t1,
            augmented_dataset_element_t0,
            meta_data,
        )
