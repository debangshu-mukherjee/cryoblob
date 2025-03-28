import chex
import jax
import jax.numpy as jnp
import pytest

from cryoblob.adapt import *
from cryoblob.types import *


class test_adaptive_wiener:
    @pytest.mark.parametrize("kernel_size", [3, (5, 5)])
    @pytest.mark.parametrize("initial_noise", [0.01, 0.1])
    @pytest.mark.parametrize("learning_rate", [0.005, 0.01])
    @chex.variants(with_jit=True, without_jit=True, device=["cpu", "gpu"])
    def test_adaptive_wiener(self, kernel_size, initial_noise, learning_rate, variant):
        img = jnp.ones((32, 32)) + 0.1 * jax.random.normal(
            jax.random.PRNGKey(0), (32, 32)
        )
        target = jnp.ones((32, 32))

        adaptive_fn = variant(adaptive_wiener)

        filtered_img, optimized_noise = adaptive_fn(
            img,
            target,
            kernel_size=kernel_size,
            initial_noise=initial_noise,
            learning_rate=learning_rate,
            iterations=10,
        )

        chex.assert_shape(filtered_img, (32, 32))
        chex.assert_type(optimized_noise, scalar_float)
        chex.assert_scalar_in_range(optimized_noise, 1e-8, 1.0)


class test_adaptive_threshold:
    @pytest.mark.parametrize("initial_threshold", [0.3, 0.7])
    @pytest.mark.parametrize("initial_slope", [5.0, 15.0])
    @pytest.mark.parametrize("learning_rate", [0.001, 0.01])
    @chex.variants(with_jit=True, without_jit=True, device=["cpu", "gpu"])
    def test_adaptive_threshold(
        self, initial_threshold, initial_slope, learning_rate, variant
    ):
        img = jnp.linspace(0, 1, 32 * 32).reshape(32, 32)
        target = jnp.where(img > 0.5, 1.0, 0.0)

        adaptive_fn = variant(adaptive_threshold)

        thresh_img, optimized_thresh, optimized_slope = adaptive_fn(
            img,
            target,
            initial_threshold=initial_threshold,
            initial_slope=initial_slope,
            learning_rate=learning_rate,
            iterations=10,
        )

        chex.assert_shape(thresh_img, (32, 32))
        chex.assert_type(optimized_thresh, scalar_float)
        chex.assert_type(optimized_slope, scalar_float)
        chex.assert_scalar_in_range(optimized_thresh, 0.0, 1.0)
        chex.assert_scalar_in_range(optimized_slope, 1.0, 50.0)
