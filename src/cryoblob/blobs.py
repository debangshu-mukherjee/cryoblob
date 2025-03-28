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
- `blob_list`:
    Detects blobs in an input image using the Laplacian of Gaussian (LoG) method.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jax import lax
from jaxtyping import Array, Float, Bool, jaxtyped

import cryoblob as cb
from cryoblob.types import *

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def find_connected_components(
    binary_image: Bool[Array, "x y z"], connectivity: int | None = 6
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
    shape = binary_image.shape

    # Initialize labels with sequential numbers for non-zero pixels
    initial_labels = jnp.where(
        binary_image > 0, jnp.arange(1, binary_image.size + 1).reshape(shape), 0
    )

    def get_neighbors(pos, labels):
        x, y, z = pos
        neighbors = []

        if connectivity == 6:
            # 6-connectivity: face neighbors
            offsets = [
                (-1, 0, 0),
                (1, 0, 0),
                (0, -1, 0),
                (0, 1, 0),
                (0, 0, -1),
                (0, 0, 1),
            ]
        else:
            # 26-connectivity: include diagonal neighbors
            offsets = [
                (dx, dy, dz)
                for dx in [-1, 0, 1]
                for dy in [-1, 0, 1]
                for dz in [-1, 0, 1]
                if not (dx == dy == dz == 0)
            ]

        for dx, dy, dz in offsets:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                neighbors.append(labels[nx, ny, nz])

        return jnp.array(neighbors)

    def update_label(old_label, new_label, labels):
        return jnp.where(labels == old_label, new_label, labels)

    def merge_components(labels):
        positions = jnp.argwhere(binary_image > 0)

        def scan_fn(labels, pos):
            neighbors = get_neighbors(pos, labels)
            valid_neighbors = neighbors[neighbors > 0]
            if len(valid_neighbors) > 0:
                min_label = jnp.min(valid_neighbors)
                current_label = labels[tuple(pos)]
                if current_label > min_label:
                    labels = update_label(current_label, min_label, labels)
            return labels, None

        labels, _ = lax.scan(scan_fn, labels, positions)
        return labels

    # Iterative merging until convergence
    def cond_fn(state):
        prev_labels, curr_labels, _ = state
        return jnp.any(prev_labels != curr_labels)

    def body_fn(state):
        _, curr_labels, i = state
        new_labels = merge_components(curr_labels)
        return curr_labels, new_labels, i + 1

    # Run until convergence
    final_labels, _, _ = lax.while_loop(
        cond_fn, body_fn, (initial_labels, initial_labels, 0)
    )

    # Relabel components to be sequential
    unique_labels = jnp.unique(final_labels)
    num_labels: scalar_int = len(unique_labels) - 1  # subtract 1 for background

    # Create mapping for sequential labels
    label_map = jnp.zeros(jnp.max(unique_labels) + 1, dtype=jnp.int32)
    label_map = label_map.at[unique_labels].set(jnp.arange(len(unique_labels)))

    # Apply mapping
    sequential_labels = label_map[final_labels]

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
    def centroid(label_idx: scalar_int) -> Float[Array, "3"]:
        mask: Bool[Array, "x y z"] = (labels == label_idx)
        masked_image: Float[Array, "x y z"] = jnp.where(mask, image, 0.0)
        total_mass: scalar_float = jnp.sum(masked_image)
        coords: Float[Array, "3"] = jnp.array([
            jnp.sum(masked_image * jnp.arange(image.shape[0])[:, None, None]) / total_mass,
            jnp.sum(masked_image * jnp.arange(image.shape[1])[None, :, None]) / total_mass,
            jnp.sum(masked_image * jnp.arange(image.shape[2])[None, None, :]) / total_mass
        ])

        return coords
    centroids: Float[Array, "n 3"] = jax.vmap(centroid)(jnp.arange(1, num_labels + 1))
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
    labels, num_labels = cb.find_connected_components(binary)
    coords = cb.center_of_mass_3d(results_3D, labels, num_labels)
    labels, num_labels = cb.find_connected_components(binary)
    coords: Float[Array, "n 3"] = cb.center_of_mass_3d(results_3D, labels, num_labels)
    return coords

@jaxtyped(typechecker=beartype)
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
        image_proc = cb.apply_gaussian_blur(image_proc, sigma=gblur)
    if background > 0:
        image_proc = image_proc - cb.apply_gaussian_blur(image_proc, sigma=background)
    if apply_filter > 0:
        image_proc = cb.wiener(image_proc, kernel_size=apply_filter)
    if return_params:
        return image_proc, processing_params
    else:
        return image_proc


@jaxtyped(typechecker=beartype)
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
    scaled_image: Float[Array, "e f"] = cb.fast_resizer(image, (1 / downscale))

    if jnp.amin(x=jnp.asarray(jnp.shape(scaled_image))) < 20:
        raise ValueError("Image is too small for blob detection")

    vectorized_log = jax.vmap(
        cb.laplacian_gaussian,
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
    coords = cb.find_particle_coords(results_3D, max_filtered, image_thresh)
    scaled_coords: Float[Array, "labels 3"] = jnp.concatenate(
        [
            downscale * coords[:, 0:2],  # x, y coordinates
            downscale * ((blob_step * coords[:, -1:]) + min_blob_size),  # z coordinate
        ],
        axis=1,
    )
    return scaled_coords
