import torch
import torchvision
from liso.visu.color_conversion import hsv2rgb
from liso.visu.utils import limit_visu_image_batches


def log_flow_image(
    writer, cfg, global_step, flow_2d, prefix="DATA/FLOW_T0_T1/", suffix=""
):
    assert flow_2d.shape[1] == 2, flow_2d.shape
    assert flow_2d.shape[3] > 3, flow_2d.shape  # need BCHW
    assert flow_2d.shape[2] > 3, flow_2d.shape  # need BCHW
    flow_img = pytorch_create_flow_image(flow_2d)

    writer.add_image(
        prefix + suffix,
        torchvision.utils.make_grid(
            limit_visu_image_batches(
                flow_img,
                max_batches=cfg.logging.max_log_img_batches,
            ),
            padding=2,
            pad_value=128,
        ),
        global_step=global_step,
    )


def pytorch_create_flow_image(flow):
    assert flow.shape[1] == 2
    assert len(flow.shape) == 4, flow.shape
    mag = torch.sqrt(torch.sum(flow**2, dim=1))
    max_mag = torch.amax(mag, dim=(1, 2))
    ang = torch.atan2(flow[:, 1, :, :], flow[:, 0, :, :])
    hue = torch.where(ang >= 0, ang, ang + 2.0 * torch.pi) / (2.0 * torch.pi)
    sat = torch.ones_like(hue)
    val = torch.where(
        mag / max_mag[..., None, None] < 1.0,
        mag / max_mag[..., None, None],
        torch.ones_like(mag),
    )
    hsv_img = torch.stack([hue, sat, val], dim=1)
    rgb_img = hsv2rgb(hsv_img)
    # return rgb_img, unguarded_max_mag
    return rgb_img
