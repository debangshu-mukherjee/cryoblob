import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from beartype.typing import Optional, Tuple, Union
from jax import lax
from jaxtyping import Array, Float, jaxtyped

import arm_em
from arm_em.types import *

jax.config.update("jax_enable_x64", True)


@jaxtyped(typechecker=typechecker)
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
        filtered_img = arm_em.wiener(img, kernel_size, noise)
        loss = jnp.mean((filtered_img - target) ** 2)
        return loss

    def step(noise: scalar_float, _) -> Tuple[scalar_float, None]:
        noise_grad = jax.grad(wiener_loss_fn)(noise, img, target, kernel_size)
        noise_updated = noise - learning_rate * noise_grad
        noise_updated = jnp.clip(noise_updated, 1e-8, 1.0)
        return noise_updated, None

    optimized_noise, _ = lax.scan(step, initial_noise, None, length=iterations)
    filtered_img = arm_em.wiener(img, kernel_size, optimized_noise)

    return filtered_img, optimized_noise
