"""
Module: blobs
---------------------------

Codes for actually detecting the blobs. The image
processing and data I/O files are kept separately.
This just deals with preprocessing data and counting
blobs.

Functions
---------
- `find_connected_components`:
    Pure JAX implementation of 3D connected components labeling.
- `center_of_mass_3d`:
    Calculate center of mass for each labeled region in a 3D image.
- `find_particle_coords`:
    Find particle coordinates using connected components and center of mass.
- `preprocessing`:
    Pre-processes low SNR images to improve contrast of blobs.
- `blob_list_log`:
    Detects blobs in an input image using the Laplacian of Gaussian (LoG) method.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from cryoblob.image import (
    apply_gaussian_blur,
    image_resizer,
    laplacian_of_gaussian,
    wiener,
)
from cryoblob.types import MRC_Image, scalar_float, scalar_int, scalar_num
from jax import lax
from jaxtyping import Array, Bool, Float, Integer, jaxtyped


@jaxtyped(typechecker=beartype)
def find_connected_components(
    binary_image: Bool[Array, "x y z"], connectivity: Optional[scalar_int] = 6
) -> Tuple[Integer[Array, "x y z"], scalar_int]:
    """
    Description
    -----------
    Pure JAX implementation of 3D connected components labeling.
    Uses a two-pass algorithm.

    Parameters
    ----------
    - `binary_image` (Bool[Array, "x y z"]):
        Binary image where True/1 indicates foreground
    - `connectivity` (int, optional):
        Either 6 (face-connected) or 26 (fully-connected).
        Default is 6

    Returns
    -------
    - `labels` (Integer[Array, "x y z"]):
        Array where each connected component has unique integer label
    - `num_labels` (int):
        Number of connected components found
    """
    # Eager, JAX-version-safe connected components. The original pure-JAX
    # label-propagation used Python control flow on traced arrays and dynamic-shape
    # ops (jnp.argwhere/jnp.unique) inside lax.while_loop, which modern JAX rejects.
    # blob_list_log runs eagerly, so use standard SciPy labeling with matching
    # 6-/26-connectivity.
    import numpy as _np
    from scipy import ndimage as _ndi

    arr = _np.asarray(binary_image) > 0
    rank = arr.ndim
    conn = 1 if int(connectivity) == 6 else rank
    structure = _ndi.generate_binary_structure(rank, conn)
    labeled, num = _ndi.label(arr, structure=structure)
    sequential_labels = jnp.asarray(labeled, dtype=jnp.int32)
    num_labels: scalar_int = int(num)
    return sequential_labels, num_labels


@jaxtyped(typechecker=beartype)
def center_of_mass_3d(
    image: Float[Array, "x y z"],
    labels: Integer[Array, "x y z"],
    num_labels: scalar_int,
) -> Float[Array, "n 3"]:
    """
    Description
    -----------
    Calculate center of mass for each labeled region in a 3D image.

    Parameters
    ----------
    - `image` (Float[Array, "x y z"]):
        3D image array
    - `labels` (Integer[Array, "x y z"]):
        Integer array of labels
    - `num_labels` (int):
        Number of labels (excluding background)

    Returns
    -------
    - `centroids` (Float[Array, "n 3"]):
        Array of centroid coordinates for each label
    """

    # Eager, memory-light intensity-weighted centroids. The original vmapped over
    # every label, each materializing a full (x,y,z) volume -> O(num_labels * volume)
    # memory (OOM at low downscale on large images). scipy.ndimage.center_of_mass
    # computes the identical sum(image*coord)/sum(image) per label. blob_list_log
    # runs eagerly, so this is a drop-in.
    import numpy as _np
    from scipy import ndimage as _ndi

    n = int(num_labels)
    if n <= 0:
        return jnp.zeros((0, 3), dtype=jnp.float32)
    img = _np.asarray(image)
    lbl = _np.asarray(labels)
    coms = _ndi.center_of_mass(img, labels=lbl, index=list(range(1, n + 1)))
    centroids: Float[Array, "n 3"] = jnp.asarray(
        _np.asarray(coms, dtype=_np.float32).reshape(n, 3), dtype=jnp.float32
    )
    return centroids


@jaxtyped(typechecker=beartype)
def find_particle_coords(
    results_3D: Float[Array, "x y z"],
    max_filtered: Float[Array, "x y z"],
    image_thresh: scalar_float,
) -> Float[Array, "n 3"]:
    """
    Description
    -----------
    Find particle coordinates using connected components and center of mass.
    Pure JAX implementation.

    Parameters
    ----------
    - `results_3D` (Float[Array, "x y z"]):
        3D array of filter responses
    - `max_filtered` (Float[Array, "x y z"]):
        Maximum filtered array
    - `image_thresh` (scalar_float):
        Threshold for peak detection

    Returns
    -------
    - `coords` (Float[Array, "n 3"]):
        Array of particle coordinates
    """
    binary: Bool[Array, "x y z"] = max_filtered > image_thresh
    labels: Integer[Array, "x y z"]
    num_labels: scalar_int
    labels, num_labels = find_connected_components(binary)
    coords = center_of_mass_3d(results_3D, labels, num_labels)
    labels, num_labels = find_connected_components(binary)
    coords: Float[Array, "n 3"] = center_of_mass_3d(results_3D, labels, num_labels)
    return coords


@jaxtyped(typechecker=beartype)
def preprocessing(
    image_orig: Float[Array, "y x"],
    return_params: Optional[bool] = False,
    exponential: Optional[bool] = True,
    logarizer: Optional[bool] = False,
    gblur: Optional[int] = 2,
    background: Optional[int] = 0,
    apply_filter: Optional[int] = 0,
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
        image_proc = apply_gaussian_blur(image_proc, sigma=float(gblur))
    if background > 0:
        image_proc = image_proc - apply_gaussian_blur(image_proc, sigma=float(background))
    if apply_filter > 0:
        image_proc = wiener(image_proc, kernel_size=apply_filter)
    if return_params:
        return image_proc, processing_params
    else:
        return image_proc


@jaxtyped(typechecker=beartype)
def blob_list_log(
    mrc_image: MRC_Image,
    min_blob_size: Optional[scalar_num] = 5,
    max_blob_size: Optional[scalar_num] = 20,
    blob_step: Optional[scalar_num] = 1,
    downscale: Optional[scalar_num] = 4,
    std_threshold: Optional[scalar_num] = 6,
) -> Float[Array, "n 3"]:
    """
    Description
    -----------
    Detect blobs of varying sizes in an MRC image using the Laplacian of Gaussian (LoG) method.

    Parameters
    ----------
    - `mrc_image` (MRC_Image):
        The PyTree containing the image data and metadata.
    - `min_blob_size` (scalar_num, optional):
        Minimum blob size to detect. Defaults to 10.
    - `max_blob_size` (scalar_num, optional):
        Maximum blob size to detect. Defaults to 100.
    - `blob_step` (scalar_num, optional):
        Step size between consecutive blob scales. Defaults to 2.
    - `downscale` (scalar_num, optional):
        Factor by which the image is downscaled before detection.
        Defaults to 4.
    - `std_threshold` (scalar_num, optional):
        Threshold in standard deviations for blob detection. Defaults to 6.

    Returns
    -------
    - `scaled_coords` (Float[Array, "n 3"]):
        Array of blob coordinates and sizes, shape [n, 3].
        Columns represent (Y, X, Blob size in pixels).
    """
    image: Float[Array, "H W"] = (mrc_image.image_data).astype(jnp.float32)
    voxel_size: Float[Array, " 3"] = mrc_image.voxel_size
    peak_range: Float[Array, " scales"] = jnp.arange(
        min_blob_size, max_blob_size, blob_step
    )
    scaled_image: Float[Array, "h w"] = image_resizer(image, downscale)

    def apply_log(img, sigma):
        return laplacian_of_gaussian(img, standard_deviation=sigma)

    results_3D: Float[Array, "scales h w"] = jax.vmap(apply_log, in_axes=(None, 0))(
        scaled_image, peak_range
    )
    results_3D = results_3D.transpose(1, 2, 0)
    max_filtered: Float[Array, "h w scales"] = jax.lax.reduce_window(
        operand=results_3D,
        init_value=-jnp.inf,
        computation=jax.lax.max,
        window_dimensions=(4, 4, 4),
        window_strides=(1, 1, 1),
        padding="SAME",
    )
    mean_val: scalar_float = jnp.mean(max_filtered)
    std_val: scalar_float = jnp.std(max_filtered)
    image_thresh: scalar_float = mean_val + std_threshold * std_val
    coords: Float[Array, "n 3"] = find_particle_coords(
        results_3D, max_filtered, image_thresh
    )
    scaled_coords: Float[Array, "n 3"] = jnp.concatenate(
        [
            downscale * coords[:, :2] * voxel_size[1:][::-1],
            (coords[:, 2:] * blob_step + min_blob_size)
            * jnp.sqrt(voxel_size[1] * voxel_size[2]),
        ],
        axis=1,
    )
    return scaled_coords
