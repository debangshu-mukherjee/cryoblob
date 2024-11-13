import json
import os
from functools import partial
from importlib.resources import files
from typing import Optional, Tuple, Union, Literal

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, Integer, Real

import arm_em

number = Union[int, float]


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
    Applies Laplacian of Gaussian (LoG) filtering to an input image.

    Parameters
    ----------
    - `image` (cupy.ndarray):
        An input image represented as a 2D CuPy array.
    - `standard_deviation` (int, optional):
        The standard deviation of the Gaussian filter. Default is 3.
    - `hist_stretch` (bool, optional):
        A boolean indicating whether to perform histogram stretching on the image. Default is True.
    - `sampling` (float, optional):
        The downsampling factor for the image. Default is 1.
    - `normalized` (bool, optional):
        A boolean indicating whether to normalize the filtered image by the standard deviation. Default is True.

    Returns:
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
        sampled_image = xexpose.equalize_hist(sampled_image)
    log_kernel: Float[Array, "3 3"] = arm_em.laplacian_kernel(mode="gaussian", size=3, sigma=standard_deviation)
    filtered: Float[Array, "y x"] = arm_em.conv2d(
        image=image,
        kernel=log_kernel,
        padding="SAME",
        padding_mode="reflect"
    )
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
        image_proc = xnd.gaussian_filter(image_proc, gblur)
    if background > 0:
        image_proc = image_proc - xnd.gaussian_filter(image_proc, background)
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
    kernel_size: int | None = 5,
    sigma: float | None = 1.0,
    padding_mode: str | None = "reflect",
) -> Float[Array, "yp xp"]:
    """
    Description
    -----------
    Apply Gaussian blur to an image using JAX.

    Parameters
    ----------
    - `image` (Real[Array, "y x"]):
        Input image to blur
    - `kernel_size` (int, optional):
        Size of the Gaussian kernel. Default is 5
    - `sigma` (float, optional):
        Standard deviation of the Gaussian kernel. Default is 1.0
    - `padding_mode` (str, optional):
        Padding mode ('reflect', 'constant', or 'edge')
        Default is 'reflect'

    Returns
    -------
    - `gauss_image` (Real[Array, "y x"]):
        Image after applying Gaussian blur
        
    Flow
    ----
    - Create the Gaussian kernel
    - Pad the image
    - Apply convolution
    - Return the blurred image
    - Unpad the image to original size
    """
    # Create the Gaussian kernel
    gauss_kernel: Float[Array, "kernel_size kernel_size"] = arm_em.gaussian_kernel(
        kernel_size, sigma
    )

    @partial(jax.jit, static_argnames=["newshape"])
    def _centered(arr, newshape):
        assert len(newshape) == arr.ndim
        startind = [(s1 - s2) // 2 for s1, s2 in zip(arr.shape, newshape)]
        return jax.lax.dynamic_slice(arr, startind, newshape)

    # Pad the image
    pad_width: int = kernel_size // 2
    padded_image: Real[Array, "yp xp"] = jnp.pad(
        image, ((pad_width, pad_width), (pad_width, pad_width)), padding_mode
    )
    padded_gauss_image: Real[Array, "yp xp"] = arm_em.conv2d(
        image=padded_image, kernel=gauss_kernel, padding="VALID"
    )
    gauss_image: Real[Array, "y x"] = _centered(padded_gauss_image, image.shape)
    return gauss_image


def conv2d(
    image: Real[Array, "ysize xsize"],
    kernel: Integer[Array, "ksize ksize"],
    stride: Union[int, Tuple[int, int]] | None = 1,
    padding: str | None = "SAME",
    padding_mode: str | None = "reflect",
) -> Real[Array, "ypsize xpsize"]:
    """
    Description
    -----------
    Perform 2D convolution on an image using JAX.

    Parameters
    ----------
    - `image` (Real[Array, "ysize xsize"]):
        Input image with shape (height, width)
        Image dimension can be two or 3d.
    - `kernel` (Real[Array, "ksize ksize"]):
        Convolution kernel with shape (kernel_height, kernel_width)
    - `stride` (int or tuple of ints, optional):
        Stride for the convolution. Default is 1
    - `padding` (str, optional):
        Padding mode ('SAME' or 'VALID'). Default is 'SAME'
    - `padding_mode` (str, optional):
        Padding mode ('reflect', 'constant', or 'edge'). Default is 'reflect'

    Returns
    -------
    - `output` (Real[Array, "ypsize xpsize"]):
        Convolved image with shape (out_height, out_width)

    Flow
    ----
    - Handle stride input
    - Handle different input shapes
    - Calculate padding if SAME
    - Extract windows using lax.conv_general_dilated_patches
    - Reshape kernel for broadcasting
    - Apply convolution
    - Remove extra dimensions if input was 2D
    """
    # Handle stride input
    if isinstance(stride, int):
        stride = (stride, stride)
    stride: Integer[Array, "2"] = jnp.asarray(stride)

    # Get shapes
    kernel_h, kernel_w = kernel.shape

    # Handle different input shapes
    squeeze_output: bool
    if image.ndim == 2:
        image = image[None, None, :, :]  # Add batch and channel dimensions
        squeeze_output = True
    elif image.ndim == 3:
        image = image[:, None, :, :]  # Add channel dimension
        squeeze_output = False
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    # Calculate padding if SAME
    if padding == "SAME":
        h_pad = jnp.maximum(kernel_h - stride[0], 0)
        w_pad = jnp.maximum(kernel_w - stride[1], 0)
        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top
        pad_left = w_pad // 2
        pad_right = w_pad - pad_left

        pad_vertical: Tuple[int, int] = (int(pad_top), int(pad_bottom))
        pad_horizontal: Tuple[int, int] = (int(pad_left), int(pad_right))

        # Apply padding
        image = jnp.pad(
            image, pad_width=(pad_vertical, pad_horizontal), mode=padding_mode
        )

    # Extract windows using lax.conv_general_dilated_patches
    windows = lax.conv_general_dilated_patches(
        image,
        filter_shape=(kernel_h, kernel_w),
        window_strides=(int(stride[0]), int(stride[1])),
        padding="VALID",
    )

    # Reshape kernel for broadcasting
    kernel_flat = kernel.reshape(-1)

    # Apply convolution
    output = jnp.sum(windows * kernel_flat, axis=-1)

    # Remove extra dimensions if input was 2D
    if squeeze_output:
        output = output[0, 0]  # Remove batch and channel dims
    elif image.ndim == 3:
        output = output[:, 0]  # Remove channel dim only

    return output

def laplacian_kernel(
    mode: Literal["basic", "diagonal", "gaussian"] = "basic",
    size: int | None = 3,
    sigma: float | None = 1.0
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
        kernel: Float[Array, "3 3"] = jnp.array([
            [0,  1,  0],
            [1, -4,  1],
            [0,  1,  0]
        ], dtype=jnp.float32)
        return kernel
    
    elif mode == "diagonal":
        kernel: Float[Array, "3 3"] = jnp.array([
            [1,  1,  1],
            [1, -8,  1],
            [1,  1,  1]
        ], dtype=jnp.float32)
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
        kernel: Float[Array, "size size"] = -1.0 / (jnp.pi * sigma**4) * (
            1 - R2 / (2 * sigma**2)
        ) * gaussian
        
        return kernel
    
    else:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of: 'basic', 'diagonal', 'gaussian'"
        )

def apply_laplacian(
    image: Real[Array, "y x"],
    mode: Literal["basic", "diagonal", "gaussian"] = "basic",
    size: int | None = 3,
    sigma: float | None = 1.0,
    padding_mode: str | None = "reflect"
) -> Float[Array, "y x"]:
    """
    Description
    -----------
    Apply Laplacian operator to an image for edge detection.

    Parameters
    ----------
    - `image` (Real[Array, "y x"]):
        Input image to process
    - `mode` (str, optional):
        Type of Laplacian kernel to use. Default is "basic"
    - `size` (int, optional):
        Size of kernel for gaussian mode. Default is 3
    - `sigma` (float, optional):
        Sigma for gaussian mode. Default is 1.0
    - `padding_mode` (str, optional):
        Padding mode for convolution. Default is "reflect"

    Returns
    -------
    - `edges` (Float[Array, "y x"]):
        The detected edges in the image

    Notes
    -----
    The Laplacian operator is used for edge detection and highlights 
    regions of rapid intensity change in the image.
    """
    # Create Laplacian kernel
    kernel: Float[Array, "size size"] = laplacian_kernel(
        mode=mode, size=size, sigma=sigma
    )
    
    # Apply convolution
    edges: Float[Array, "y x"] = conv2d(
        image=image,
        kernel=kernel,
        padding="SAME",
        padding_mode=padding_mode
    )
    
    return edges