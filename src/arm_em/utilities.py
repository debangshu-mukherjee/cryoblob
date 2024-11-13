import json
import os
from importlib.resources import files
from typing import Any, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, Real

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
    image: NDArray[Shape["*, *"], Float],
    standard_deviation: Optional[Union[Int, Float]] = 3,
    hist_stretch: Optional[Bool] = True,
    sampling: Optional[Union[Float, Int]] = 1,
    normalized: Optional[Bool] = True,
) -> NDArray[Shape["*, *"], Float]:
    """
    Applies Laplacian of Gaussian (LoG) filtering to an input image.

    Args:
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
    image: NDArray[Shape["*, *"], Float] = xp.asarray(image.astype(xp.float64))
    if sampling != 1:
        sampled_image: NDArray[Shape["*, *"], Float] = xnd.zoom(image, sampling)
    else:
        sampled_image: NDArray[Shape["*, *"], Float] = xp.copy(image)
    if hist_stretch:
        sampled_image = xexpose.equalize_hist(sampled_image)
    gauss_image: NDArray[Shape["*, *"], Float] = xnd.gaussian_filter(
        sampled_image, standard_deviation
    )
    laplacian: NDArray[Shape["3, 3"], Float] = xp.asarray(
        (
            (0.0, 1.0, 0.0),
            (1.0, -4.0, 1.0),
            (0.0, 1.0, 0.0),
        ),
        dtype=np.float64,
    )
    filtered: NDArray[Shape["*, *"], Float] = xsig.convolve2d(
        gauss_image, laplacian, mode="same", boundary="symm", fillvalue=0
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


def pad_image(
    image: Real[Array, "y x"], pad_width: int, mode: str | None = "reflect"
) -> Real[Array, "yp xp"]:
    """
    Description
    -----------
    Pad an image with the specified mode.

    Parameters
    ----------
    - `image` (Real[Array, "y x"]):
        Input image to pad
    - `pad_width` (int):
        Number of pixels to pad on each side
    - `mode` (str, optional):
        Padding mode ('reflect', 'constant', or 'edge')
        Default is 'reflect'

    Returns
    -------
    - `padded_image` (Real[Array, "yp xp"]):
        Padded image
    """
    padded_image: Real[Array, "yp xp"] = jnp.pad(
        image, ((pad_width, pad_width), (pad_width, pad_width)), mode=mode
    )
    return padded_image


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

    Returns:
        Blurred image
    """
    # Create the Gaussian kernel
    kernel: Float[Array, "kernel_size kernel_size"] = arm_em.gaussian_kernel(
        kernel_size, sigma
    )

    # Pad the image
    pad_width: int = kernel_size // 2
    padded_image = arm_em.pad_image(image, pad_width, padding_mode)

    # Apply the convolution
    return conv2d(padded_image, kernel)


@jax.jit
def conv2d(
    image: ArrayLike,
    kernel: ArrayLike,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: str = 'SAME',
    padding_mode: str = 'reflect'
) -> ArrayLike:
    """
    Perform 2D convolution on an image using JAX.
    
    Args:
        image: Input image with shape (height, width) or (batch, height, width)
        kernel: Convolution kernel with shape (kernel_height, kernel_width)
        stride: Integer or tuple of integers for stride in (height, width)
        padding: Either 'SAME' or 'VALID'
        padding_mode: Padding mode for 'SAME' padding ('reflect', 'constant', or 'edge')
        
    Returns:
        Convolved image with appropriate shape based on padding and stride
    """
    # Handle stride input
    if isinstance(stride, int):
        stride = (stride, stride)
    
    # Get shapes
    kernel_h, kernel_w = kernel.shape
    
    # Handle different input shapes
    if image.ndim == 2:
        image = image[None, None, :, :]  # Add batch and channel dimensions
        squeeze_output = True
    elif image.ndim == 3:
        image = image[:, None, :, :]  # Add channel dimension
        squeeze_output = False
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    # Calculate padding if SAME
    if padding == 'SAME':
        h_pad = max(kernel_h - stride[0], 0)
        w_pad = max(kernel_w - stride[1], 0)
        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top
        pad_left = w_pad // 2
        pad_right = w_pad - pad_left
        
        # Apply padding
        image = jnp.pad(
            image,
            ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode=padding_mode
        )
    
    # Extract windows using lax.conv_general_dilated_patches
    windows = lax.conv_general_dilated_patches(
        image,
        filter_shape=(kernel_h, kernel_w),
        window_strides=stride,
        padding='VALID'
    )
    
    # Reshape kernel for broadcasting
    kernel_flat = kernel.reshape(-1)
    
    # Apply convolution
    output = jnp.sum(
        windows * kernel_flat,
        axis=-1
    )
    
    # Remove extra dimensions if input was 2D
    if squeeze_output:
        output = output[0, 0]  # Remove batch and channel dims
    elif image.ndim == 3:
        output = output[:, 0]  # Remove channel dim only
    
    return output

@jax.jit
def conv2d_multi_channel(
    image: ArrayLike,
    kernels: ArrayLike,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: str = 'SAME',
    padding_mode: str = 'reflect'
) -> ArrayLike:
    """
    Perform 2D convolution on a multi-channel image using multiple kernels.
    
    Args:
        image: Input image with shape (height, width, in_channels) or (batch, height, width, in_channels)
        kernels: Convolution kernels with shape (out_channels, kernel_height, kernel_width, in_channels)
        stride: Integer or tuple of integers for stride in (height, width)
        padding: Either 'SAME' or 'VALID'
        padding_mode: Padding mode for 'SAME' padding ('reflect', 'constant', or 'edge')
        
    Returns:
        Convolved image with shape (..., out_channels)
    """
    # Handle stride input
    if isinstance(stride, int):
        stride = (stride, stride)
    
    # Handle different input shapes
    if image.ndim == 3:  # (H, W, C)
        image = image[None, ...]  # Add batch dimension
        squeeze_output = True
    elif image.ndim == 4:  # (B, H, W, C)
        squeeze_output = False
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    # Move channels to second dimension for convolution
    image = jnp.transpose(image, (0, 3, 1, 2))  # (B, C, H, W)
    
    # Get shapes
    batch_size, in_channels, height, width = image.shape
    out_channels, kernel_h, kernel_w, _ = kernels.shape
    
    # Calculate padding if SAME
    if padding == 'SAME':
        h_pad = max(kernel_h - stride[0], 0)
        w_pad = max(kernel_w - stride[1], 0)
        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top
        pad_left = w_pad // 2
        pad_right = w_pad - pad_left
        
        # Apply padding
        image = jnp.pad(
            image,
            ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode=padding_mode
        )
    
    # Reshape kernels for batch matmul
    kernels = kernels.reshape(out_channels, -1, in_channels)  # (out_C, K*K, in_C)
    
    # Extract windows
    windows = lax.conv_general_dilated_patches(
        image,
        filter_shape=(kernel_h, kernel_w),
        window_strides=stride,
        padding='VALID'
    )  # (B, C, out_H, out_W, K*K)
    
    # Reshape windows for batch matmul
    out_h, out_w = windows.shape[2:4]
    windows = windows.transpose(0, 2, 3, 1, 4)  # (B, out_H, out_W, C, K*K)
    windows = windows.reshape(-1, in_channels, kernel_h * kernel_w)  # (B*out_H*out_W, C, K*K)
    
    # Perform convolution using batch matmul
    output = jnp.matmul(windows, kernels.transpose(0, 2, 1))  # (B*out_H*out_W, C, out_C)
    
    # Reshape output
    output = output.reshape(batch_size, out_h, out_w, out_channels)
    
    # Remove batch dimension if input was 3D
    if squeeze_output:
        output = output[0]
    
    return output

