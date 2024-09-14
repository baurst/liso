import torch

# for ravel_multi_index see https://github.com/francois-rozet/torchist/blob/5cf8dc62fa73109318433aa0c1c5ac9e265961e4/torchist/__init__.py#L18


def ravel_multi_index(coords, shape):
    r"""Converts a tensor of coordinate vectors into a tensor of flat indices.
    This is a `torch` implementation of `numpy.ravel_multi_index`.
    Args:
        coords: A tensor of coordinate vectors, (*, D).
        shape: The source shape.
    Returns:
        The raveled indices, (*,).
    """

    shape = coords.new_tensor(shape + (1,))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return (coords * coefs).sum(dim=-1)


def scatter_add_2d(
    target_tensor: torch.FloatTensor,
    batch_idxs: torch.IntTensor,
    row_idxs: torch.IntTensor,
    col_idxs: torch.IntTensor,
    update_values: torch.FloatTensor,
):
    assert target_tensor.shape[-1] == update_values.shape[-1], (
        target_tensor.shape,
        update_values.shape,
    )
    num_updates, num_channels = update_values.shape
    (batch_size, num_target_rows, num_target_cols, _) = target_tensor.shape
    channel_idxs = (
        torch.arange(num_channels, device=batch_idxs.device)[None, :]
        .repeat((num_updates, 1))
        .view((-1, 1))
    )
    target_tensor_flat = target_tensor.view(-1)
    spatial_target_idxs = torch.stack(
        [batch_idxs, row_idxs, col_idxs], dim=-1
    ).repeat_interleave(num_channels, dim=0)
    target_w_channel_idxs = torch.cat([spatial_target_idxs, channel_idxs], dim=-1)
    target_idxs_flat = ravel_multi_index(
        target_w_channel_idxs, shape=target_tensor.shape
    )
    target_tensor_flat.scatter_add_(0, target_idxs_flat, update_values.view((-1)))
    target_tensor = target_tensor_flat.view(
        (batch_size, num_target_rows, num_target_cols, num_channels)
    )

    return target_tensor


def masked_scatter_mean_2d(
    target_tensor: torch.FloatTensor,
    valid_mask: torch.BoolTensor,
    batch_idxs: torch.IntTensor,
    row_idxs: torch.IntTensor,
    col_idxs: torch.IntTensor,
    update_values: torch.FloatTensor,
):
    masked_batch_idxs = batch_idxs[valid_mask]
    masked_row_idxs = row_idxs[valid_mask]
    masked_col_idxs = col_idxs[valid_mask]
    masked_update_values = update_values[valid_mask]
    target_tensor = scatter_add_2d(
        target_tensor,
        masked_batch_idxs,
        masked_row_idxs,
        masked_col_idxs,
        masked_update_values,
    )

    count_tensor = scatter_add_2d(
        torch.zeros_like(target_tensor[..., :1], dtype=torch.int),
        masked_batch_idxs,
        masked_row_idxs,
        masked_col_idxs,
        torch.ones_like(masked_update_values[..., :1], dtype=torch.int),
    )

    target_tensor = torch.where(
        count_tensor > 1, target_tensor / count_tensor, target_tensor
    )
    return target_tensor
