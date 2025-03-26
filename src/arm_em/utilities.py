import json
import os
from functools import partial
from importlib.resources import files

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import List, Literal, Tuple, TypeAlias, Union
from jax import lax
from jax.scipy import signal
from jaxtyping import Array, Bool, Float, Integer, Num, Real, jaxtyped

import arm_em

scalar_float: TypeAlias = Union[float, Float[Array, ""]]
scalar_int: TypeAlias = Union[int, Integer[Array, ""]]
scalar_num: TypeAlias = Union[int, float, Num[Array, ""]]

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=typechecker)
@jax.jit
def image_resizer(
    orig_image: Union[Real[Array, "y x"], Real[Array, "y x c"]],
    new_sampling: Union[Real[Array, ""], Real[Array, "2"]],
) -> Float[Array, "a b"]:
    """
    Description
    -----------
    Resize an image using a fast resizing algorithm implemented in JAX.
    If a 3D stack is provided, the function will sum along the last dimension.

    Parameters
    ----------
    - `orig_image` (Union[Real[Array, "y x"], Real[Array, "y x c"]]):
        The original image to be resized. It should be a 2D JAX array or 3D stack.
    - `new_sampling` (Union[Real[Array, ""], Real[Array, "2"]]):
        The new sampling rate for resizing the image. It can be a single
        float value or a tuple of two float values representing the sampling
        rates for the x and y axes respectively.

    Returns
    -------
    - `resampled_image` (Float[Array, "a b"]):
        The resized image.
    """
    image: Float[Array, "y x"] = jnp.where(
        jnp.ndim(orig_image) == 3, jnp.sum(orig_image, axis=-1), orig_image
    ).astype(jnp.float32)
    sampling_array: Float[Array, "2"] = jnp.broadcast_to(
        jnp.atleast_1d(new_sampling), (2,)
    ).astype(jnp.float32)
    in_y, in_x = image.shape
    new_y_len: scalar_int = jnp.round(in_y / sampling_array[0]).astype(jnp.int32)
    new_x_len: scalar_int = jnp.round(in_x / sampling_array[1]).astype(jnp.int32)
    resized_x: Float[Array, "y new_x"] = arm_em.resize_x(image, new_x_len)
    swapped: Float[Array, "new_x y"] = jnp.swapaxes(resized_x, 0, 1)
    resized_xy: Float[Array, "new_x new_y"] = arm_em.resize_x(swapped, new_y_len)
    resampled_image: Float[Array, "new_y new_x"] = jnp.swapaxes(resized_xy, 0, 1)
    return resampled_image


@jaxtyped(typechecker=typechecker)
def resize_x(
    x_image: Num[Array, "y x"], new_x_len: scalar_int
) -> Float[Array, "y new_x"]:
    """
    Description
    -----------
    Resize image along y-axis by independently resampling each column.
    Uses `lax.scan` over the y-dimension, then `vmap` over x-dimension.

    Parameters
    ----------
    - `x_image` (Num[Array, "y x"]):
        Image to resize (y by x)
    - `new_x_len` (scalar_int):
        Target number of columns

    Returns
    -------
    - `resized` (Float[Array, "y new_x"]):
        Resized image (new_y by x)
    """
    orig_x_len: int = x_image.shape[1]

    def resize_column(col: Float[Array, "x"]) -> Float[Array, "new_x"]:
        """
        Resize a single 1D column using cumulative area-based resampling.
        """

        def scan_body(
            carry: Tuple[Integer[Array, ""], Float[Array, ""]], nn: Integer[Array, ""]
        ) -> Tuple[Tuple[Integer[Array, ""], Float[Array, ""]], Float[Array, ""]]:
            m: Integer[Array, ""] = carry[0]
            carry_val: Float[Array, ""] = carry[1]

            def while_cond(
                state: Tuple[Integer[Array, ""], Float[Array, ""], None]
            ) -> Bool[Array, ""]:
                m_state: Integer[Array, ""] = state[0]
                return ((m_state * new_x_len) - (nn * orig_x_len)) < orig_x_len

            def while_body(
                state: Tuple[Integer[Array, ""], Float[Array, ""], None]
            ) -> Tuple[Integer[Array, ""], Float[Array, ""], None]:
                m_state, data_sum, _ = state
                new_sum = data_sum + col[m_state]
                return (m_state + 1, new_sum, None)

            init_state = (m, jnp.array(0.0, dtype=col.dtype), None)
            final_m, data_sum, _ = lax.while_loop(while_cond, while_body, init_state)

            fraction: Float[Array, ""] = final_m - (nn + 1) * orig_x_len / new_x_len
            last_contribution: Float[Array, ""] = fraction * col[final_m - 1]
            adjusted_sum: Float[Array, ""] = data_sum - last_contribution
            result: Float[Array, ""] = (adjusted_sum * new_x_len) / orig_x_len

            return (final_m, last_contribution), result

        init_carry = (jnp.array(0), jnp.array(0.0, dtype=col.dtype))
        _, resized_col = lax.scan(scan_body, init_carry, jnp.arange(new_x_len))
        return resized_col

    resized: Float[Array, "y new_x"] = jax.vmap(resize_column)(x_image)
    return resized


def file_params() -> Tuple[str, dict]:
    """
    Run this at the beginning to generate the dict
    This gives both the absolute and relative paths
    on how the files are organized.

    Returns:
        - `main_directory` (str):
            the main directory where the package is located.
        - `folder_structure` (dict):
            where the files and data are stored, as read
            from the organization.json file.
    """
    pkg_directory: str = os.path.dirname(__file__)
    listring: List = pkg_directory.split("/")[1:-2]
    listring.append("")
    listring.insert(0, "")
    main_directory: str = "/".join(listring)
    folder_structure: dict = json.load(
        open(files("arm_em.params").joinpath("organization.json"))
    )
    return (main_directory, folder_structure)


@jaxtyped(typechecker=typechecker)
def laplacian_gaussian(
    image: Real[Array, "y x"],
    standard_deviation: scalar_num | None = 3,
    hist_stretch: bool | None = True,
    sampling: scalar_num | None = 1,
    normalized: bool | None = True,
) -> Float[Array, "y x"]:
    """
    Description
    -----------
    Applies Laplacian of Gaussian (LoG) filtering to an
    input image.

    Parameters
    ----------
    - `image` (Real[Array, "y x"]):
        An input image represented as a 2D array.
    - `standard_deviation` (int, optional):
        The standard deviation of the Gaussian filter.
        Default is 3.
    - `hist_stretch` (bool, optional):
        A boolean indicating whether to perform histogram
        stretching on the image.
        Default is True.
    - `sampling` (float, optional):
        The downsampling factor for the image.
        Default is 1.
    - `normalized` (bool, optional):
        A boolean indicating whether to normalize the filtered
        image by the standard deviation.
        Default is True.

    Returns
    -------
        - `filtered` (NDArray[Shape["*, *"], Float]):
            The laplacian of gaussian filtered image.

    Flow
    ----
    - If sampling is not 1, the image is resized.
    - If hist_stretch is True, the image is histogram stretched.
    - A Laplacian of Gaussian kernel is created.
    - The image is filtered using the kernel.
    - If normalized is True, the image is normalized by the standard deviation.
    """
    sampled_image: Float[Array, "y x"]
    if sampling != 1:
        sampled_image = arm_em.fast_resizer(image, sampling)
    else:
        sampled_image = jnp.asarray(image, dtype=jnp.float64)
    if hist_stretch:
        sampled_image = arm_em.equalize_hist(sampled_image)
    log_kernel: Float[Array, "3 3"] = arm_em.laplacian_kernel(
        mode="gaussian", size=3, sigma=standard_deviation
    )
    filtered: Float[Array, "y x"] = signal.convolve2d(image, log_kernel, mode="same")
    if normalized:
        filtered = filtered * standard_deviation
    return filtered


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
        image_proc = arm_em.apply_gaussian_blur(image_proc, sigma=gblur)
    if background > 0:
        image_proc = image_proc - arm_em.apply_gaussian_blur(
            image_proc, sigma=background
        )
    if apply_filter > 0:
        image_proc = arm_em.wiener(image_proc, kernel_size=apply_filter)
    if return_params:
        return image_proc, processing_params
    else:
        return image_proc


@jaxtyped(typechecker=typechecker)
def gaussian_kernel(size: int, sigma: float) -> Float[Array, "size size"]:
    """
    Description
    -----------
    Create a 2D Gaussian kernel.

    Parameters
    ----------
    - `size` (int):
        The size of the kernel (will be size x size)
    - sigma (float):
        The standard deviation of the Gaussian distribution

    Returns
    -------
    - `normalized_gaussian` (Float[Array, "size size"]):
        A 2D Gaussian kernel normalized to sum to 1
    """
    # Create a grid of coordinates
    x: Float[Array, "size"] = jnp.arange(-(size // 2), size // 2 + 1)
    y: Float[Array, "size"] = jnp.arange(-(size // 2), size // 2 + 1)
    X: Float[Array, "size size"]
    Y: Float[Array, "size size"]
    X, Y = jnp.meshgrid(x, y)

    # Calculate the 2D gaussian
    gaussian: Float[Array, "size size"] = jnp.exp(-(X**2 + Y**2) / (2 * sigma**2))

    # Normalize the gaussian
    normalized_gaussian: Float[Array, "size size"] = gaussian / jnp.sum(gaussian)
    return normalized_gaussian


@jaxtyped(typechecker=typechecker)
@jax.jit
def apply_gaussian_blur(
    image: Real[Array, "y x"],
    sigma: float | None = 1.0,
    kernel_size: int | None = 5,
    mode: Literal["full", "valid", "same"] | None = "same",
) -> Float[Array, "yp xp"]:
    """
    Description
    -----------
    Apply Gaussian blur to an image using JAX's scipy signal processing.

    Parameters
    ----------
    - `image` (Real[Array, "y x"]):
        Input image to blur
    - `sigma` (float, optional):
        Standard deviation of the Gaussian kernel. Default is 1.0
    - `kernel_size` (int, optional):
        Size of the Gaussian kernel. Default is 5
    - `mode` (str, optional):
        The type of convolution:
        - 'full': output is full discrete linear convolution
        - 'valid': output consists only of elements computed without padding
        - 'same': output is same size as input, centered
        Default is 'same'

    Returns
    -------
    - `blurred` (Float[Array, "yp xp"]):
        The blurred image
    """
    # Create the Gaussian kernel
    kernel: Float[Array, "kernel_size kernel_size"] = arm_em.gaussian_kernel(
        kernel_size, sigma
    )

    # Apply convolution
    blurred: Float[Array, "y x"] = signal.convolve2d(image, kernel, mode=mode)
    return blurred


@jaxtyped(typechecker=typechecker)
def laplacian_kernel(
    mode: Literal["basic", "diagonal", "gaussian"] = "basic",
    size: int | None = 3,
    sigma: float | None = 1.0,
) -> Float[Array, "size size"]:
    """
    Description
    -----------
    Create a Laplacian kernel for edge detection.

    Parameters
    ----------
    - `mode` (str):
        The type of Laplacian kernel to create:
        - "basic": Standard 4-connected Laplacian
        - "diagonal": 8-connected Laplacian including diagonals
        - "gaussian": Laplacian of Gaussian (LoG)
    - `size` (int, optional):
        The size of the kernel (will be size x size).
        Required for 'gaussian' mode. Default is 3
    - `sigma` (float, optional):
        The standard deviation for Gaussian mode.
        Only used when mode='gaussian'. Default is 1.0

    Returns
    -------
    - `kernel` (Float[Array, "size size"]):
        The Laplacian kernel array

    Notes
    -----
    The kernels are:
    - Basic: [[ 0,  1,  0],
             [ 1, -4,  1],
             [ 0,  1,  0]]

    - Diagonal: [[ 1,  1,  1],
                [ 1, -8,  1],
                [ 1,  1,  1]]

    - Gaussian: Laplacian of Gaussian with specified size/sigma
    """
    if mode == "basic":
        kernel: Float[Array, "3 3"] = jnp.array(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=jnp.float32
        )
        return kernel

    elif mode == "diagonal":
        kernel: Float[Array, "3 3"] = jnp.array(
            [[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=jnp.float32
        )
        return kernel

    elif mode == "gaussian":
        # Create coordinate grid
        x: Float[Array, "size"] = jnp.arange(-(size // 2), size // 2 + 1)
        y: Float[Array, "size"] = jnp.arange(-(size // 2), size // 2 + 1)
        X: Float[Array, "size size"]
        Y: Float[Array, "size size"]
        X, Y = jnp.meshgrid(x, y)

        # Calculate squared radius
        R2: Float[Array, "size size"] = X**2 + Y**2

        # Calculate Laplacian of Gaussian
        # LoG(x,y) = -1/(pi*sigma^4) * [1 - (x^2 + y^2)/(2*sigma^2)] * exp(-(x^2 + y^2)/(2*sigma^2))
        gaussian: Float[Array, "size size"] = jnp.exp(-R2 / (2 * sigma**2))
        kernel: Float[Array, "size size"] = (
            -1.0 / (jnp.pi * sigma**4) * (1 - R2 / (2 * sigma**2)) * gaussian
        )

        return kernel

    else:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of: 'basic', 'diagonal', 'gaussian'"
        )


@jaxtyped(typechecker=typechecker)
def histogram(
    image: Real[Array, "h w"],
    bins: int | None = 256,
    range_limits: Tuple[float, float] | None = None,
) -> Integer[Array, "bins"]:
    """
    Description
    -----------
    Calculate the histogram of an image.

    Parameters
    ----------
    - `image` (Real[Array, "h w"]):
        Input image
    - `bins` (int, optional):
        Number of bins for the histogram. Default is 256
    - `range_limits` (Tuple[float, float], optional):
        The lower and upper range of the bins.
        Default is (image.min(), image.max())

    Returns
    -------
    - `hist` (Integer[Array, "bins"]):
        The histogram of the image
    """
    if range_limits is None:
        range_limits = (float(image.min()), float(image.max()))

    hist: Integer[Array, "bins"] = jnp.histogram(image, bins=bins, range=range_limits)[
        0
    ]
    return hist


@jaxtyped(typechecker=typechecker)
@jax.jit
def equalize_hist(
    image: Real[Array, "h w"], nbins: int = 256, mask: Real[Array, "h w"] | None = None
) -> Float[Array, "h w"]:
    """
    Description
    -----------
    Perform histogram equalization on an image using JAX.

    Parameters
    ----------
    - `image` (Real[Array, "h w"]):
        Input image to equalize
    - `nbins` (int, optional):
        Number of bins for histogram. Default is 256
    - `mask` (Real[Array, "h w"], optional):
        Optional mask to use for selective equalization.
        Only pixels where mask is True will be included in hist calculation.
        Default is None (use all pixels)

    Returns
    -------
    - `equalized` (Float[Array, "h w"]):
        Histogram equalized image

    Notes
    -----
    This implementation follows scikit-image's equalize_hist approach:
    1. Compute histogram
    2. Calculate cumulative distribution
    3. Normalize and map values
    """
    # Check if all values are the same
    if jnp.all(image == image.ravel()[0]):
        return jnp.full_like(image, image.ravel()[0], dtype=jnp.float32)

    # Normalize image to [0, 1] range
    img_range: float = image.max() - image.min()
    normalized: Float[Array, "h w"] = (image - image.min()) / img_range

    # Handle mask if provided
    flat_normalized: Real[Array, "p"]
    if mask is not None:
        flat_mask = mask.ravel()
        flat_normalized = jnp.compress(flat_mask, flat_normalized)
    else:
        flat_normalized = normalized.ravel()

    # Calculate histogram
    hist: Integer[Array, "bins"] = arm_em.histogram(
        flat_normalized, bins=nbins, range_limits=(0.0, 1.0)
    )

    # Calculate cumulative distribution
    cdf: Integer[Array, "bins"] = jnp.cumsum(hist)

    # Normalize CDF
    cdf: Float[Array, "bins"] = cdf / cdf[-1]

    # Map values using linear interpolation
    def map_values(x: Float[Array, ""]) -> Float[Array, ""]:
        # Find bin index
        bin_idx = jnp.clip(jnp.floor(x * (nbins - 1)).astype(jnp.int32), 0, nbins - 2)

        # Get surrounding CDF values
        cdf_left = cdf[bin_idx]
        cdf_right = cdf[bin_idx + 1]

        # Calculate fractional position within bin
        f = (x * (nbins - 1)) - bin_idx

        # Linear interpolation
        return (1 - f) * cdf_left + f * cdf_right

    # Vectorize mapping function
    vmap_values = jax.vmap(map_values)

    # Apply mapping
    equalized: Real[Array, "h w"] = vmap_values(normalized.ravel()).reshape(image.shape)

    return equalized


@jaxtyped(typechecker=typechecker)
@jax.jit
def equalize_adapthist(
    image: Real[Array, "h w"],
    kernel_size: int = 8,
    clip_limit: float = 0.01,
    nbins: int = 256,
) -> Float[Array, "h w"]:
    """
    Description
    -----------
    Perform Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Parameters
    ----------
    - `image` (Real[Array, "h w"]):
        Input image
    - `kernel_size` (int, optional):
        Size of local region for histogram equalization. Default is 8
    - `clip_limit` (float, optional):
        Clipping limit for histogram. Higher values give stronger contrast.
        Default is 0.01
    - `nbins` (int, optional):
        Number of bins for histogram. Default is 256

    Returns
    -------
    - `equalized` (Float[Array, "h w"]):
        CLAHE equalized image
    """
    # Normalize image to [0, 1]
    img_min = image.min()
    img_max = image.max()
    normalized = (image - img_min) / (img_max - img_min)

    # Calculate grid size
    h, w = normalized.shape
    grid_h = (h + kernel_size - 1) // kernel_size
    grid_w = (w + kernel_size - 1) // kernel_size

    def process_block(block: Float[Array, "kh kw"]) -> Float[Array, "kh kw"]:
        # Calculate histogram
        hist = histogram(block, bins=nbins, range_limits=(0.0, 1.0))

        # Clip histogram
        if clip_limit > 0:
            excess = jnp.sum(jnp.maximum(hist - clip_limit * block.size / nbins, 0))
            gain = excess / (nbins * block.size)
            hist = jnp.minimum(hist, clip_limit * block.size / nbins) + gain

        # Calculate CDF
        cdf = jnp.cumsum(hist)
        cdf = cdf / cdf[-1]

        # Map values
        def map_value(x):
            bin_idx = jnp.clip(
                jnp.floor(x * (nbins - 1)).astype(jnp.int32), 0, nbins - 2
            )
            cdf_left = cdf[bin_idx]
            cdf_right = cdf[bin_idx + 1]
            f = (x * (nbins - 1)) - bin_idx
            return (1 - f) * cdf_left + f * cdf_right

        vmap_value = jax.vmap(map_value)
        return vmap_value(block.ravel()).reshape(block.shape)

    # Process each block
    def process_grid(i, j):
        start_h = i * kernel_size
        start_w = j * kernel_size

        # Calculate block size (handling edge cases)
        block_h = jnp.minimum(kernel_size, h - start_h)
        block_w = jnp.minimum(kernel_size, w - start_w)

        # Use dynamic_slice instead of regular slicing
        block = jax.lax.dynamic_slice(
            normalized, (start_h, start_w), (block_h, block_w)
        )

        return process_block(block)

    # Create output array
    equalized = jnp.zeros_like(normalized)

    # Apply CLAHE to each block
    for i in range(grid_h):
        for j in range(grid_w):
            start_h = i * kernel_size
            start_w = j * kernel_size
            block_h = min(kernel_size, h - start_h)
            block_w = min(kernel_size, w - start_w)

            processed_block = process_grid(i, j)

            # Use dynamic_update_slice for setting values
            equalized = jax.lax.dynamic_update_slice(
                equalized, processed_block, (start_h, start_w)
            )

    return equalized


@jaxtyped(typechecker=typechecker)
@jax.jit
def wiener(
    img: Float[Array, "h w"],
    kernel_size: Union[int, Tuple[int, int]] | None = 3,
    noise: float | None = None,
) -> Float[Array, "h w"]:
    """
    Description
    -----------
    JAX implementation of Wiener filter for noise reduction.
    This is similar to scipy.signal.wiener.

    Parameters
    ----------
    - `img` (Float[Array, "h w"]):
        The input image to be filtered
    - `kernel_size` (int or tuple, optional):
        The size of the sliding window for local statistics.
        If tuple, represents (height, width).
        Default is 3
    - `noise` (float, optional):
        The noise power. If None, uses the average of the
        local variance.
        Default is None

    Returns
    -------
    - `filtered` (Float[Array, "h w"]):
        The filtered output with the same shape as input

    Notes
    -----
    The Wiener filter is optimal in terms of the mean square error.
    It estimates the local mean and variance around each pixel.
    """
    # Handle kernel size input
    if isinstance(kernel_size, int):
        kernel_size: Integer[Array, "2"] = jnp.asarray((kernel_size, kernel_size))

    # Create uniform kernel for local means
    kernel: Float[Array, "ksize ksize"] = jnp.ones(
        shape=kernel_size, dtype=jnp.float64
    ) / (kernel_size[0] * kernel_size[1])

    # Calculate local mean using convolution
    local_mean: Float[Array, "h w"] = signal.convolve2d(img, kernel, mode="same")

    # Calculate local variance using convolution
    local_var: Float[Array, "h w"] = signal.convolve2d(
        jnp.square(img), kernel, mode="same"
    ) - jnp.square(local_mean)

    # Ensure variance is positive
    local_var = jnp.maximum(local_var, 0)

    if noise is None:
        # Estimate noise as the average of the local variances
        noise = jnp.mean(local_var)

    # Apply Wiener filter
    filtered: Float[Array, "h w"] = local_mean + (
        (local_var - noise) / jnp.maximum(local_var, noise)
    ) * (img - local_mean)

    return filtered


@jaxtyped(typechecker=typechecker)
def find_connected_components(
    binary_image: Bool[Array, "x y z"], connectivity: int | None = 6
) -> Tuple[Integer[Array, "x y z"], int]:
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
    num_labels = len(unique_labels) - 1  # subtract 1 for background

    # Create mapping for sequential labels
    label_map = jnp.zeros(jnp.max(unique_labels) + 1, dtype=jnp.int32)
    label_map = label_map.at[unique_labels].set(jnp.arange(len(unique_labels)))

    # Apply mapping
    sequential_labels = label_map[final_labels]

    return sequential_labels, num_labels


def center_of_mass_3d(
    image: Float[Array, "x y z"], labels: Integer[Array, "x y z"], num_labels: int
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

    def compute_centroid(label_idx):
        mask = labels == label_idx
        masked_image: Float[Array, "p q r"] = jnp.where(mask, image, 0)
        total_mass: Float[Array, ""] = jnp.sum(masked_image)

        x_coords: Num[Array, "x 1 1"] = jnp.arange(image.shape[0])[:, None, None]
        y_coords: Num[Array, "1 y 1"] = jnp.arange(image.shape[1])[None, :, None]
        z_coords: Num[Array, "1 1 z"] = jnp.arange(image.shape[2])[None, None, :]

        center_x: Float[Array, ""] = jnp.sum(masked_image * x_coords) / total_mass
        center_y: Float[Array, ""] = jnp.sum(masked_image * y_coords) / total_mass
        center_z: Float[Array, ""] = jnp.sum(masked_image * z_coords) / total_mass

        return jnp.array([center_x, center_y, center_z])

    # Vectorize over labels
    centroids: Float[Array, "n 3"] = jax.vmap(compute_centroid)(
        jnp.arange(1, num_labels + 1)
    )
    return centroids


@jaxtyped(typechecker=typechecker)
@jax.jit
def find_particle_coords(
    results_3D: Float[Array, "x y z"],
    max_filtered: Float[Array, "x y z"],
    image_thresh: float,
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
    - `image_thresh` (float):
        Threshold for peak detection

    Returns
    -------
    - `coords` (Float[Array, "n 3"]):
        Array of particle coordinates
    """
    # Create binary image of peaks
    binary: Bool[Array, "x y z"] = max_filtered > image_thresh

    # Find connected components
    labels, num_labels = arm_em.find_connected_components(binary)

    # Calculate center of mass for each component
    coords = arm_em.center_of_mass_3d(results_3D, labels, num_labels)

    return coords
