"""
Module: files
---------------------------

Contains the codes for interfacing with data files.
One goal here is to separate the Python code from
the JAX code. Thus most of the necessary outward
facing code, which is necessarily in Python, is here.

Functions
---------
- `plot_mrc`:
    Plot MRC image data using Matplotlib with optional scaling and scalebar.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib_scalebar.scalebar as sb
import numpy as np
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Float
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cryoblob.types import MRC_Image, scalar_int


@beartype
def plot_mrc(
    mrc_image: MRC_Image,
    image_size: Optional[Tuple[scalar_int, scalar_int]] = (15, 15),
    cmap: Optional[str] = "magma",
    mode: Optional[str] = "plain",
) -> None:
    """
    Description
    -----------
    Plot an MRC image using Matplotlib with an optional scaling mode and scalebar.

    Parameters
    ----------
    - `mrc_image` (MRC_Image):
        The PyTree structure containing image data and voxel metadata.
    - `image_size` (Tuple[scalar_int, scalar_int], optional)
        Size of the plotted figure (width, height) in inches.
        Default is (15, 15).
    - `cmap` (str, optional):
        The Matplotlib colormap to use.
        Default is "viridis".
    - `mode` (str, optional):
        Mode of visualization:
        - "plain": Plot image data without modifications.
        - "log": Plot logarithmically scaled image data.
        - "exp": Plot exponentially scaled image data.
        Default is "plain".

    Returns
    -------
    None
        Displays the plot.

    Examples
    --------
    >>> plot_mrc(mrc_image, image_size=(10, 10), cmap="viridis", mode="log")
    """
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=image_size)
    normalized_image: Float[Array, "H W"] = (
        mrc_image.image_data - mrc_image.data_min
    ) / (mrc_image.data_max - mrc_image.data_min)
    image_to_plot: Float[Array, "H W"]
    if mode == "log":
        image_to_plot = jnp.log(1 + normalized_image)
    elif mode == "exp":
        image_to_plot = jnp.exp(normalized_image)
    elif mode == "plain":
        image_to_plot = normalized_image
    else:
        raise ValueError("Invalid mode. Choose from 'plain', 'log', or 'exp'.")
    voxel_size_x: float = float(mrc_image.voxel_size[2])
    scalebar: sb.ScaleBar = sb.ScaleBar(
        10 * voxel_size_x,
        units="nm",
        location="lower right",
        box_alpha=0.5,
        color="white",
        frameon=False,
    )
    ax.imshow(np.asarray(image_to_plot), cmap=cmap, origin="lower")
    ax.add_artist(scalebar)
    ax.axis("off")
    fig.tight_layout()
    plt.show()
