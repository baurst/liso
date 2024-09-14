import numpy as np


def scatter_add_nd_numpy(indices, updates, shape):
    assert shape[-1] == updates.shape[-1]
    assert len(shape) == len(indices.shape) + 1  # broadcast along last dim
    target = np.zeros(shape, dtype=updates.dtype)
    indices = tuple(indices.T)
    np.add.at(target, indices, updates)
    return target


def scatter_mean_nd_numpy(indices, updates, shape):
    target = scatter_add_nd_numpy(indices, updates, shape)

    counter_updates = np.ones(updates.shape[:-1], dtype=np.uint32)[..., None]
    counter_target_shape = shape[:-1] + (1,)
    num_samples_per_target_cell = scatter_add_nd_numpy(
        indices, counter_updates, counter_target_shape
    )

    mean_target = target / np.where(
        num_samples_per_target_cell == 0, 1, num_samples_per_target_cell
    )

    return mean_target
