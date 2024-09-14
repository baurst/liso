import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from PIL import Image


def limit_visu_image_batches(img, max_batches=8):
    if img.shape[0] > max_batches:
        return img[:max_batches, ...]
    else:
        return img


def apply_cmap(dL_dpos, normalize=True, input_has_channel_dim=False):
    # assert len(dL_dpos.shape) == 2, dL_dpos.shape
    dl_dx = dL_dpos.detach().cpu().numpy()
    if normalize:
        dl_dx = (dl_dx - np.min(dl_dx)) / np.ptp(dl_dx)

    magma = cm.get_cmap("magma")
    if input_has_channel_dim:
        assert dL_dpos.shape[-1] == 1, dL_dpos.shape
        dl_dx = dl_dx[..., 0]
    dl_dx_img = magma(dl_dx)
    return dl_dx_img


def plot_to_np_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = np.array(
        Image.frombytes(
            "RGB", figure.canvas.get_width_height(), figure.canvas.tostring_rgb()
        )
    )
    # Add the batch dimension
    return image
