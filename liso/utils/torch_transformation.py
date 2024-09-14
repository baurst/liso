import numpy as np
import torch


def torch_decompose_matrix(matrix):
    if matrix.dtype == torch.float32:
        raise UserWarning(
            "You are decomposing a matrix with 32bit prec.", "This might  be unstable"
        )
    translation = matrix[..., :3, 3].clone()
    theta_z = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])
    return translation, theta_z[..., None]


def torch_compose_matrix(t_x, t_y, theta_z, t_z=None):
    """this is the torch equivalent to the gohlke function compose_matrix
    Return transformation matrix from sequence of transformations."""
    num_batches, num_slots = t_x.shape
    assert t_x.shape == t_y.shape
    if t_z is None:
        t_z = torch.zeros_like(t_x)
    M = (
        torch.eye(4, dtype=t_x.dtype, device=t_x.device)
        .reshape(1, 4, 4)
        .repeat(num_batches, num_slots, 1, 1)
    )
    T = (
        torch.eye(4, dtype=t_x.dtype, device=t_x.device)
        .reshape(1, 4, 4)
        .repeat(num_batches, num_slots, 1, 1)
    )
    T[:, :, 0, 3] = t_x
    T[:, :, 1, 3] = t_y
    T[:, :, 2, 3] = t_z
    M = torch.einsum("bsij,bsjk->bsik", M, T)  # dot product

    si, sj, sk = 0.0, 0.0, torch.sin(theta_z)
    ci, cj, ck = 1.0, 1.0, torch.cos(theta_z)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    R = (
        torch.eye(4, device=t_x.device, dtype=t_x.dtype)
        .reshape(1, 4, 4)
        .repeat(num_batches, num_slots, 1, 1)
    )

    R[:, :, 0, 0] = cj * ck
    R[:, :, 0, 1] = sj * sc - cs
    R[:, :, 0, 2] = sj * cc + ss
    R[:, :, 1, 0] = cj * sk
    R[:, :, 1, 1] = sj * ss + cc
    R[:, :, 1, 2] = sj * cs - sc
    R[:, :, 2, 0] = -sj
    R[:, :, 2, 1] = cj * si
    R[:, :, 2, 2] = cj * ci

    M = torch.einsum("bsij,bsjk->bsik", M, R)  # dot product

    M = M / M[:, :, 3:, 3:]  # no idea why this is necessary, but gohlke uses it so...

    return M


def numpy_compose_matrix(t_x, t_y, theta_z, t_z=None):
    """this is the numpy equivalent to the gohlke function compose_matrix
    Return transformation matrix from sequence of transformations."""
    num_batches, num_slots = t_x.shape
    assert t_x.shape == t_y.shape
    if t_z is None:
        t_z = np.zeros_like(t_x)
    M = np.tile(
        np.eye(4, dtype=t_x.dtype)[None, None, ...], (num_batches, num_slots, 1, 1)
    )
    T = np.tile(
        np.eye(4, dtype=t_x.dtype)[None, None, ...], (num_batches, num_slots, 1, 1)
    )
    T[:, :, 0, 3] = t_x
    T[:, :, 1, 3] = t_y
    T[:, :, 2, 3] = t_z
    M = np.einsum("bsij,bsjk->bsik", M, T)  # dot product

    si, sj, sk = 0.0, 0.0, np.sin(theta_z)
    ci, cj, ck = 1.0, 1.0, np.cos(theta_z)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    R = np.tile(
        np.eye(4, dtype=t_x.dtype)[None, None, ...], (num_batches, num_slots, 1, 1)
    )

    R[:, :, 0, 0] = cj * ck
    R[:, :, 0, 1] = sj * sc - cs
    R[:, :, 0, 2] = sj * cc + ss
    R[:, :, 1, 0] = cj * sk
    R[:, :, 1, 1] = sj * ss + cc
    R[:, :, 1, 2] = sj * cs - sc
    R[:, :, 2, 0] = -sj
    R[:, :, 2, 1] = cj * si
    R[:, :, 2, 2] = cj * ci

    M = np.einsum("bsij,bsjk->bsik", M, R)  # dot product

    M = M / M[:, :, 3:, 3:]  # no idea why this is necessary, but gohlke uses it so...

    return M


def homogenize_pcl(
    pcl: torch.FloatTensor, is_not_padding: torch.BoolTensor = None
) -> torch.FloatTensor:
    if isinstance(pcl, torch.Tensor):
        if pcl.shape[-1] == 4:
            if is_not_padding is None:
                assert torch.all(pcl[..., -1] == 1.0)
            else:
                assert torch.all(torch.isfinite(pcl[is_not_padding]))
                assert torch.all(pcl[is_not_padding][..., -1] == 1.0)
            pcl_homog = pcl
        else:
            pcl_homog = torch.cat([pcl, torch.ones_like(pcl[..., :1])], dim=-1)
    elif isinstance(pcl, np.ndarray):
        if pcl.shape[-1] == 4:
            if is_not_padding is None:
                assert np.all(pcl[..., -1] == 1.0)
            else:
                assert np.all(np.isfinite(pcl[is_not_padding]))
                assert np.all(pcl[is_not_padding][..., -1] == 1.0)
            pcl_homog = pcl
        else:
            pcl_homog = np.concatenate([pcl, np.ones_like(pcl[..., :1])], axis=-1)
    else:
        raise NotImplementedError(type(pcl))
    return pcl_homog


def homogenize_flow(
    flow: torch.FloatTensor, is_not_padding: torch.BoolTensor = None
) -> torch.FloatTensor:
    if isinstance(flow, torch.Tensor):
        if flow.shape[-1] == 4:
            if is_not_padding is None:
                assert torch.all(flow[..., -1] == 0.0)
            else:
                assert torch.all(torch.isfinite(flow[is_not_padding]))
                assert torch.all(flow[is_not_padding][..., -1] == 0.0)
        else:
            flow = torch.cat([flow, torch.zeros_like(flow[..., :1])], dim=-1)
    elif isinstance(flow, np.ndarray):
        if flow.shape[-1] == 4:
            if is_not_padding is None:
                assert np.all(flow[..., -1] == 0.0)
            else:
                assert np.all(np.isfinite(flow[is_not_padding]))
                assert np.all(flow[is_not_padding][..., -1] == 0.0)
        else:
            flow = np.concatenate([flow, np.zeros_like(flow[..., :1])], axis=-1)
    else:
        raise NotImplementedError(type(flow))
    return flow
