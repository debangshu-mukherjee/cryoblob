import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Optional, TypeAlias, Union
from jax import lax
from jaxtyping import Array, Float, Int, Num, jaxtyped

import cryoblob
from cryoblob.types import *

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=typechecker)
def preprocessing(
    image_orig: Float[Array, "y x"],
    return_params: bool | None = False,
    exponential: bool | None = True,
    logarizer: bool | None = False,
    gblur: int | None = 2,
    background: int | None = 0,
    apply_filter: int | None = 0,
) -> Float[Array, "y x"]:
    """
    Description
    -----------
    Pre-processing of low SNR images to
    improve contrast of blobs.

    Parameters
    ----------
    - `image_orig` (Float[Array, "y x"]):
        An input image represented as a 2D JAX array.
    - `return_params` (bool, optional):
        A boolean indicating whether to return the processing parameters.
        Default is False.
    - `exponential` (bool, optional):
        A boolean indicating whether to apply an exponential function to the image.
        Default is True.
    - `logarizer` (bool, optional):
        A boolean indicating whether to apply a log function to the image.
        Default is False.
    - `gblur` (int, optional):
        The standard deviation of the Gaussian filter.
        Default is 2.
    - `background` (int, optional):
        The standard deviation of the Gaussian filter for background subtraction.
        Default is 0.
    - `apply_filter` (int, optional):
        If greater than 1, a Wiener filter is applied to the image.

    Returns
    -------
    - `image_proc` (Float[Array, "y x"]):
        The pre-processed image
    """
    processing_params: dict = {
        "exponential": exponential,
        "logarizer": logarizer,
        "gblur": gblur,
        "background": background,
        "apply_filter": apply_filter,
    }

    image_proc: Float[Array, "y x"]
    if jnp.amax(image_orig) == jnp.amin(image_orig):
        image_proc = jnp.zeros(image_orig, dtype=jnp.float64)
    else:
        image_proc = (image_orig - jnp.amin(image_orig)) / (
            jnp.amax(image_orig) - jnp.amin(image_orig)
        )
    if exponential:
        image_proc = jnp.exp(image_proc)
    if logarizer:
        image_proc = jnp.log(image_proc)
    if gblur > 0:
        image_proc = cryoblob.apply_gaussian_blur(image_proc, sigma=gblur)
    if background > 0:
        image_proc = image_proc - cryoblob.apply_gaussian_blur(
            image_proc, sigma=background
        )
    if apply_filter > 0:
        image_proc = cryoblob.wiener(image_proc, kernel_size=apply_filter)
    if return_params:
        return image_proc, processing_params
    else:
        return image_proc


def blob_list(
    image: Float[Array, "a b"] | Float[Array, "a b c"],
    min_blob_size: Optional[scalar_num] = 10,
    max_blob_size: Optional[scalar_num] = 100,
    blob_step: Optional[scalar_num] = 2,
    downscale: Optional[scalar_num] = 2,
    std_threshold: Optional[scalar_num] = 6,
) -> Float[Array, "a 3"]:
    """
    Description
    -----------
    Detects blobs in an input image using the Laplacian of Gaussian (LoG) method.
    If input is 3D, sums along the last axis before processing.

    Args
    ----
    - `image` (Float[Array, "a b"] | Float[Array, "a b c"]):
        A 2D or 3D array representing the input image.
        If 3D, will be summed along the last axis.
    - `min_blob_size` (Num[Array, ""], optional):
        The minimum size of the blobs to be detected.
        Defaults to 10.
    - `max_blob_size` (Num[Array, ""], optional):
        The maximum size of the blobs to be detected.
        Defaults to 100.
    - `blob_step` (Num[Array, ""], optional):
        The step size for iterating over different blob sizes.
        Defaults to 2.
    - `downscale` (Num[Array, ""], optional):
        The factor by which the image is downscaled before blob detection.
        Defaults to 2.
    - `std_threshold` (Num[Array, ""], optional):
        The threshold for blob detection based on standard deviation.
        Defaults to 6.

    Returns
    -------
    - `scaled_coords` (Float[Array, "labels 3"]):
        A 2D array containing the coordinates of the detected blobs. Each row
        represents the coordinates of a blob, with the first two columns
        representing the x and y coordinates, and the last column
        representing the size of the blob.

    Notes
    -----
    For 3D inputs, the function:
    1. Sums along the last axis to create a 2D projection
    2. Processes the 2D projection for blob detection
    3. Returns coordinates in the 2D projection space
    """
    # Handle 3D input by summing along last axis
    if image.ndim == 3:
        image = jnp.sum(image, axis=-1)

    peak_range: Float[Array, "c"] = jnp.arange(
        start=min_blob_size, stop=max_blob_size, step=blob_step
    )
    scaled_image: Float[Array, "e f"] = cryoblob.fast_resizer(image, (1 / downscale))

    if jnp.amin(x=jnp.asarray(jnp.shape(scaled_image))) < 20:
        raise ValueError("Image is too small for blob detection")

    vectorized_log = jax.vmap(
        cryoblob.laplacian_gaussian,
        in_axes=(
            None,
            0,
        ),  # First arg (image) is broadcasted, second arg (sigma) is mapped
    )
    results_3D: Float[Array, "e f r"] = vectorized_log(
        scaled_image, peak_range
    ).transpose(1, 2, 0)

    max_filtered: Float[Array, "e f r"] = lax.reduce_window(
        results_3D,
        init_value=-jnp.inf,
        computation_fn=lax.max,
        window_dimensions=(4, 4, 4),  # 3D window size
        window_strides=(1, 1, 1),  # stride of 1 in each dimension
        padding="SAME",  # to match scipy's behavior
    )

    image_thresh: Float[Array, "e f r"] = jnp.mean(max_filtered) + (
        std_threshold * jnp.std(max_filtered)
    )
    coords = cryoblob.find_particle_coords(results_3D, max_filtered, image_thresh)
    scaled_coords: Float[Array, "labels 3"] = jnp.concatenate(
        [
            downscale * coords[:, 0:2],  # x, y coordinates
            downscale * ((blob_step * coords[:, -1:]) + min_blob_size),  # z coordinate
        ],
        axis=1,
    )
    return scaled_coords
