import torch


def fuse_masks_screen(masks, *, dim, keepdim=False):
    total_occ = 1.0 - torch.prod(1.0 - masks, dim=dim, keepdim=keepdim)
    return total_occ
