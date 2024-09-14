import torch
import torch.nn.functional as F


def upflow_n(flow, n=8, mode="bilinear"):
    new_size = (n * flow.shape[2], n * flow.shape[3])
    return n * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def uplogits_n(logits, n=8, mode="bilinear"):
    new_size = (n * logits.shape[2], n * logits.shape[3])
    return F.interpolate(logits, size=new_size, mode=mode, align_corners=True)


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(
        torch.arange(ht, device=device), torch.arange(wd, device=device), indexing="ij"
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


# adapted from core/raft.py
def initialize_flow(img, downscale_factor=8):
    N, _, H, W = img.shape
    coords0 = coords_grid(
        N, H // downscale_factor, W // downscale_factor, device=img.device
    )
    # optical flow computed as difference: flow = coords1 - coords0
    return coords0
