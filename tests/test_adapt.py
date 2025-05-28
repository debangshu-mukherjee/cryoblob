import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax import random

jax.config.update("jax_enable_x64", True)

from cryoblob.adapt import *
from cryoblob.types import *

if __name__ == "__main__":
    pytest.main([__file__])


class test_adaptive_wiener(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.rng = random.PRNGKey(0)
        self.img = jnp.ones((32, 32)) + 0.1 * random.normal(self.rng, (32, 32))
        self.target = jnp.ones((32, 32))

    @chex.all_variants
    @parameterized.parameters(
        {"kernel_size": 3, "initial_noise": 0.01, "learning_rate": 0.005},
        {"kernel_size": (5, 5), "initial_noise": 0.1, "learning_rate": 0.01},
    )
    def test_output_shapes(self, kernel_size, initial_noise, learning_rate):
        adaptive_fn = self.variant(adaptive_wiener)
        filtered_img, optimized_noise = adaptive_fn(
            self.img,
            self.target,
            kernel_size=kernel_size,
            initial_noise=initial_noise,
            learning_rate=learning_rate,
            iterations=10,
        )
        assert_shape(filtered_img, (32, 32))
        chex.assert_scalar_in_range(optimized_noise, 1e-8, 1.0)


class test_adaptive_threshold(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.img = jnp.linspace(0, 1, 32 * 32).reshape(32, 32)
        self.target = jnp.where(self.img > 0.5, 1.0, 0.0)

    @chex.all_variants
    @parameterized.parameters(
        {"initial_threshold": 0.3, "initial_slope": 5.0, "learning_rate": 0.001},
        {"initial_threshold": 0.7, "initial_slope": 15.0, "learning_rate": 0.01},
    )
    def test_output_shapes(self, initial_threshold, initial_slope, learning_rate):
        adaptive_fn = self.variant(adaptive_threshold)
        thresh_img, optimized_thresh, optimized_slope = adaptive_fn(
            self.img,
            self.target,
            initial_threshold=initial_threshold,
            initial_slope=initial_slope,
            learning_rate=learning_rate,
            iterations=10,
        )
        assert_shape(thresh_img, (32, 32))
        chex.assert_scalar_in_range(optimized_thresh, 0.0, 1.0)
        chex.assert_scalar_in_range(optimized_slope, 1.0, 50.0)
