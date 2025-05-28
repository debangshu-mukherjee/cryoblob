"""
Module: adapt
---------------------------

Contains adaptive image processing methods
that take advantage of JAX's automatic
differentiation capabilities.

Functions
---------
- `adaptive_wiener`:
    Adaptive Wiener filter that optimizes the noise estimate using gradient descent.
- `adaptive_threshold`:
    Adaptively optimizes thresholding parameters using gradient descent
    to produce a differentiably thresholded image.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple, Union
from jax import lax
from jaxtyping import Array, Float, jaxtyped

from cryoblob.image import wiener
from cryoblob.types import scalar_float, scalar_int

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=beartype)
def adaptive_wiener(
    img: Float[Array, "h w"],
    target: Float[Array, "h w"],
    kernel_size: Optional[Union[scalar_int, Tuple[int, int]]] = 3,
    initial_noise: Optional[scalar_float] = 0.1,
    learning_rate: Optional[scalar_float] = 0.01,
    iterations: Optional[scalar_int] = 100,
) -> Tuple[Float[Array, "h w"], scalar_float]:
    """
    Adaptive Wiener filter that optimizes the noise estimate using gradient descent.

    Parameters
    ----------
    - `img` (Float[Array, "h w"]):
        Noisy input image.
    - `target` (Float[Array, "h w"]):
        A target image or reference image used for optimization.
    - `kernel_size` (scalar_int | Tuple[int, int], optional):
        Window size for Wiener filter. Default is 3.
    - `initial_noise` (scalar_float, optional):
        Initial guess for noise parameter. Default is 0.1.
    - `learning_rate` (scalar_float, optional):
        Learning rate for optimization. Default is 0.01.
    - `iterations` (scalar_int, optional):
        Number of optimization steps. Default is 100.

    Returns
    -------
    - `filtered_img` (Float[Array, "h w"]):
        Wiener filtered image with optimized noise parameter.
    - `optimized_noise` (scalar_float):
        The optimized noise parameter.
    """

    def wiener_loss_fn(
        noise: scalar_float,
        img: Float[Array, "h w"],
        target: Float[Array, "h w"],
        kernel_size: Union[int, Tuple[int, int]],
    ) -> scalar_float:
        filtered_img: Float[Array, "h w"] = wiener(img, kernel_size, noise)
        loss: scalar_float = jnp.mean((filtered_img - target) ** 2)
        return loss

    def step(noise: scalar_float, _) -> Tuple[scalar_float, None]:
        noise_grad = jax.grad(wiener_loss_fn)(noise, img, target, kernel_size)
        noise_updated = noise - learning_rate * noise_grad
        noise_updated = jnp.clip(noise_updated, 1e-8, 1.0)
        return noise_updated, None

    optimized_noise, _ = lax.scan(step, initial_noise, None, length=iterations)
    filtered_img = wiener(img, kernel_size, optimized_noise)

    return filtered_img, optimized_noise


@jaxtyped(typechecker=beartype)
def adaptive_threshold(
    img: Float[Array, "h w"],
    target: Float[Array, "h w"],
    initial_threshold: Optional[scalar_float] = 0.5,
    initial_slope: Optional[scalar_float] = 10.0,
    learning_rate: Optional[scalar_float] = 0.01,
    iterations: Optional[scalar_int] = 100,
) -> Tuple[Float[Array, "h w"], scalar_float, scalar_float]:
    """
    Description
    -----------
    Adaptively optimizes thresholding parameters using gradient descent
    to produce a differentiably thresholded image.

    Parameters
    ----------
    - `img` (Float[Array, "h w"]):
        The input image to threshold.
    - `target` (Float[Array, "h w"]):
        A reference binary image for supervised parameter optimization.
    - `initial_threshold` (scalar_float, optional):
        Initial guess for the threshold parameter.
        Default is 0.5.
    - `initial_slope` (scalar_float, optional):
        Initial guess for the slope controlling sigmoid steepness.
        Default is 10.0.
    - `learning_rate` (scalar_float, optional):
        The learning rate used during gradient optimization.
        Default is 0.01.
    - `iterations` (scalar_int, optional):
        Number of iterations for gradient optimization.
        Default is 100.

    Returns
    -------
    - `thresholded_img` (Float[Array, "h w"]):
        The image after differentiable thresholding using optimized parameters.
    - `optimized_threshold` (scalar_float):
        The optimized threshold parameter.
    - `optimized_slope` (scalar_float):
        The optimized slope parameter.

    Flow
    ----
    - `sigmoid_threshold`:
        Applies a sigmoid function to the input image.
    - `threshold_loss_fn`:
        Computes the loss between the thresholded image and the target.
    - `step`:
        Performs a single optimization step.
    - `optimized_params`:
        Optimizes threshold and slope parameters.
    - `thresholded_img`:
        Applies the optimized thresholding parameters to the
        input image.
    """

    def sigmoid_threshold(
        img: Float[Array, "h w"],
        threshold: scalar_float,
        slope: scalar_float,
    ) -> Float[Array, "h w"]:
        return jax.nn.sigmoid(slope * (img - threshold))

    def threshold_loss_fn(
        params: Float[Array, "2"],
        img: Float[Array, "h w"],
        target: Float[Array, "h w"],
    ) -> scalar_float:
        threshold, slope = params
        thresh_img = sigmoid_threshold(img, threshold, slope)
        return jnp.mean((thresh_img - target) ** 2)

    def step(params: Float[Array, "2"], _: None) -> Tuple[Float[Array, "2"], None]:
        grads = jax.grad(threshold_loss_fn)(params, img, target)
        updated_params = params - learning_rate * grads
        updated_params = updated_params.at[1].set(
            jnp.clip(updated_params[1], 1.0, 50.0)
        )
        return updated_params, None

    initial_params: Float[Array, "2"] = jnp.array([initial_threshold, initial_slope])
    optimized_params, _ = lax.scan(step, initial_params, None, length=iterations)

    optimized_threshold, optimized_slope = optimized_params
    thresholded_img = sigmoid_threshold(img, optimized_threshold, optimized_slope)

    return thresholded_img, optimized_threshold, optimized_slope
