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
from cryoblob.types import MRC_Image, scalar_int
from jaxtyping import Array, Float
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle


@beartype
def plot_mrc(
    mrc_image: MRC_Image,
    image_size: Optional[Tuple[scalar_int, scalar_int]] = (15, 15),
    cmap: Optional[str] = "magma",
    mode: Optional[str] = "plain",
    blobs: Optional[Float[Array, "n 3"]] = None,
    blob_color: Optional[str] = "cyan",
) -> None:
    """
    Description
    -----------
    Plot an MRC image using Matplotlib with an optional scaling mode and scalebar,
    optionally overlaying detected blobs.

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
    - `blobs` (Float[Array, "n 3"], optional):
        Blob list as returned by `blob_list_log` / `blob_list_log_watershed`,
        with columns (Y, X, size) in physical units (i.e. already multiplied by
        the voxel size). When provided, each blob is drawn as a circle at its
        detected position and radius. The physical coordinates are converted
        back to pixel coordinates using `mrc_image.voxel_size`, so the overlay
        aligns with the displayed image. Default is None (no overlay).
    - `blob_color` (str, optional):
        Matplotlib color for the blob circles. Default is "cyan".

    Returns
    -------
    None
        Displays the plot.

    Examples
    --------
    >>> plot_mrc(mrc_image, image_size=(10, 10), cmap="viridis", mode="log")
    >>> blobs = blob_list_log(mrc_image, min_blob_size=8, max_blob_size=110)
    >>> plot_mrc(mrc_image, blobs=blobs)
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
    if blobs is not None:
        blobs_np: np.ndarray = np.asarray(blobs)
        if blobs_np.size > 0:
            voxel_y: float = float(mrc_image.voxel_size[1])
            voxel_x: float = float(mrc_image.voxel_size[2])
            # Invert the physical scaling applied inside blob_list_log:
            # column 0 (Y) was multiplied by voxel_x, column 1 (X) by voxel_y,
            # and the size by sqrt(voxel_y * voxel_x).
            y_pix: np.ndarray = blobs_np[:, 0] / voxel_x
            x_pix: np.ndarray = blobs_np[:, 1] / voxel_y
            r_pix: np.ndarray = blobs_np[:, 2] / np.sqrt(voxel_y * voxel_x)
            for xc, yc, rc in zip(x_pix, y_pix, r_pix):
                ax.add_patch(
                    Circle(
                        (xc, yc),
                        radius=max(float(rc), 1.0),
                        edgecolor=blob_color,
                        facecolor="none",
                        linewidth=1.0,
                    )
                )
    ax.axis("off")
    fig.tight_layout()
    plt.show()
