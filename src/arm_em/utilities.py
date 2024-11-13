import json
import os
from functools import partial
from importlib.resources import files
from typing import List, Literal, Tuple, Union

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jax import lax
from jax.scipy import signal
from jaxtyping import Array, Float, Integer, Real, jaxtyped

import arm_em

number = Union[int, float]

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=typechecker)
def fast_resizer(
    orig_image: Real[Array, "y x"],
    new_sampling: Union[number, Tuple[number, number], Real[Array, "1 2"]],
) -> Float[Array, "a b"]:
    """
    Description
    -----------
    Resize an image using a fast resizing algorithm implemented in JAX.

    Parameters
    ----------
    - `orig_image` (Real[Array, "y x"]):
        The original image to be resized. It should be a 2D JAX array.
    - `new_sampling` (Union[number, Tuple[number, number], Real[Array, "1 2"]]):
        The new sampling rate for resizing the image. It can be a single
        float value or a tuple of two float values representing the sampling
        rates for the x and y axes respectively.

    Returns
    -------
    - `resampled_image` (Float[Array, "a b"]):
        The resized image.
    """

    def resize_y(y_image: Real[Array, "y x"], new_y_len: int) -> Real[Array, "y b"]:
        """
        Resize the image along the y-axis.

        Args:
            y_image (ArrayLike):
                The image to be resized along the y-axis.
            new_y_len (int):
                The new length of the y-axis.

        Returns:
            ArrayLike:
                The resized image along the y-axis.
        """
        orig_y_len = y_image.shape[0]

        def scan_body(carry, nn):
            m, carry_array = carry

            def while_cond(state):
                m, _, _ = state
                return ((m * new_y_len) - (nn * orig_y_len)) < orig_y_len

            def while_body(state):
                m, data_sum, _ = state
                return (m + 1, data_sum + y_image[m, :], None)

            # Initialize data_sum with carry_array
            init_state = (m, jnp.copy(carry_array), None)

            # Run the while loop using lax.while_loop
            final_m, data_sum, _ = lax.while_loop(while_cond, while_body, init_state)

            # Calculate new carry array
            new_carry_array = (final_m - (nn + 1) * orig_y_len / new_y_len) * y_image[
                final_m - 1, :
            ]

            # Subtract carry array from data sum
            data_sum = data_sum - new_carry_array

            # Calculate final result for this row
            result = (data_sum * new_y_len) / orig_y_len

            return (final_m, new_carry_array), result

        # Initialize carry state
        init_carry = (0, jnp.zeros(y_image.shape[1], dtype=y_image.dtype))

        # Use scan to iterate over rows
        _, results = lax.scan(scan_body, init_carry, jnp.arange(new_y_len))

        return results

    # Convert new_sampling to JAX array
    if not hasattr(new_sampling, "__len__"):
        new_sampling = jnp.array([new_sampling, new_sampling], dtype=jnp.float32)
    else:
        new_sampling = jnp.asarray(new_sampling, dtype=jnp.float32)

    # Convert image to float
    float_image = jnp.asarray(orig_image, dtype=jnp.float32)

    # Resize in y direction
    resampled_in_y = resize_y(
        float_image, int(jnp.round(float_image.shape[0] / new_sampling[0]))
    )

    # Resize in x direction by transposing, resizing, and transposing back
    swapped_image = jnp.swapaxes(resampled_in_y, 0, 1)
    resampled_swapped = resize_y(
        swapped_image, int(jnp.round(swapped_image.shape[0] / new_sampling[1]))
    )
    resampled_image = jnp.swapaxes(resampled_swapped, 0, 1)

    return resampled_image


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


def laplacian_gaussian(
    image: Real[Array, "y x"],
    standard_deviation: number | None = 3,
    hist_stretch: bool | None = True,
    sampling: number | None = 1,
    normalized: bool | None = True,
) -> Float[Array, "y x"]:
    """
    Description
    -----------
    Applies Laplacian of Gaussian (LoG) filtering to an
    input image.

    Parameters
    ----------
    - `image` (cupy.ndarray):
        An input image represented as a 2D CuPy array.
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

    Flow:
    - Convert the input image to a CuPy array and ensure it is of type float64.
    - If downsampling is applied, zoom the image accordingly.
    - If histogram stretching is enabled, perform histogram equalization on the sampled image.
    - Apply Gaussian filtering to the sampled image using the specified standard deviation.
    - Define the Laplacian filter kernel.
    - Convolve the Gaussian filtered image with the Laplacian kernel.
    - If normalization is enabled, scale the filtered image by the standard deviation.
    - Return the filtered image.
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
        image_proc = xsig.wiener(image_proc, mysize=apply_filter)
    if return_params:
        return image_proc, processing_params
    else:
        return image_proc


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
