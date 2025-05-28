import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax import random

jax.config.update("jax_enable_x64", True)

from cryoblob.image import image_resizer, gaussian_kernel, wiener

if __name__ == "__main__":
    pytest.main([__file__])


class test_image_resizer(chex.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.rng = random.PRNGKey(0)
        self.base_shape = (10, 10)
        self.base_image = random.uniform(self.rng, self.base_shape)

    @chex.all_variants
    @parameterized.parameters(
        {"shape": (10, 10), "sampling": 0.5, "expected_shape": (20, 20)},
        {"shape": (32, 48), "sampling": 2.0, "expected_shape": (16, 24)},
        {"shape": (100, 150), "sampling": (0.5, 1.0), "expected_shape": (200, 150)},
        {
            "shape": (3, 5),
            "sampling": jnp.array([[0.5, 0.75]]),
            "expected_shape": (6, 7),
        },
    )
    def test_output_shapes(self, shape, sampling, expected_shape):
        var_image_resizer = self.variant(image_resizer)
        image = random.uniform(self.rng, shape)
        result = var_image_resizer(image, sampling)
        chex.assert_shape(result, expected_shape)

    @chex.all_variants
    @parameterized.parameters(
        {
            "image": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            "sampling": 2.0,
            "expected": jnp.array([[2.5]]),
        },
        {
            "image": jnp.array([[1.0, 1.0], [1.0, 1.0]]),
            "sampling": 1.0,
            "expected": jnp.array([[1.0, 1.0], [1.0, 1.0]]),
        },
    )
    def test_known_values(self, image, sampling, expected):
        var_image_resizer = self.variant(image_resizer)
        result = var_image_resizer(image, sampling)
        chex.assert_trees_all_close(result, expected, atol=1e-5)

    @chex.all_variants
    def test_batch_consistency(self):
        var_image_resizer = self.variant(image_resizer)
        batch_size = 3
        images = random.uniform(self.rng, (batch_size,) + self.base_shape)

        batch_resize = jax.vmap(lambda x: var_image_resizer(x, 0.5))
        results = batch_resize(images)

        for i in range(batch_size):
            individual_result = var_image_resizer(images[i], 0.5)
            chex.assert_trees_all_close(results[i], individual_result)

    @chex.all_variants
    def test_dtype_consistency(self):
        var_image_resizer = self.variant(image_resizer)
        dtypes = [jnp.float32, jnp.float64]

        for dtype in dtypes:
            image = self.base_image.astype(dtype)
            result = var_image_resizer(image, 0.5)
            assert result.dtype == dtype, f"Expected dtype {dtype}, got {result.dtype}"

    @chex.all_variants
    @parameterized.parameters(
        {"sampling": 0.1},
        {"sampling": 10.0},
        {"sampling": (0.1, 10.0)},
    )
    def test_extreme_sampling(self, sampling):
        var_image_resizer = self.variant(image_resizer)
        result = var_image_resizer(self.base_image, sampling)
        chex.assert_tree_all_finite(result)

    @chex.all_variants
    def test_gradient_computation(self):
        var_image_resizer = self.variant(image_resizer)

        def loss_fn(image):
            resized = var_image_resizer(image, 0.5)
            return jnp.sum(resized)

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(self.base_image)

        chex.assert_shape(grads, self.base_shape)
        chex.assert_tree_all_finite(grads)

    @chex.all_variants
    def test_deterministic_output(self):
        var_image_resizer = self.variant(image_resizer)
        result1 = var_image_resizer(self.base_image, 0.5)
        result2 = var_image_resizer(self.base_image, 0.5)
        chex.assert_trees_all_close(result1, result2)

    @chex.all_variants
    def test_image_range_preservation(self):
        """Test that the resizer preserves the general range of values."""
        var_image_resizer = self.variant(image_resizer)
        test_image = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        result = var_image_resizer(test_image, 0.5)

        assert jnp.all(result >= test_image.min() * 0.9)
        assert jnp.all(result <= test_image.max() * 1.1)


class test_gaussian_kernel(chex.TestCase):
    @chex.all_variants
    @parameterized.parameters(
        {"size": 3, "sigma": 1.0, "expected_shape": (3, 3)},
        {"size": 5, "sigma": 0.5, "expected_shape": (5, 5)},
        {"size": 7, "sigma": 2.0, "expected_shape": (7, 7)},
        {"size": 9, "sigma": 1.5, "expected_shape": (9, 9)},
    )
    def test_output_shapes(self, size, sigma, expected_shape):
        var_gaussian_kernel = self.variant(gaussian_kernel)
        kernel = var_gaussian_kernel(size, sigma)
        chex.assert_shape(kernel, expected_shape)

    @chex.all_variants
    def test_kernel_normalization(self):
        """Test if kernel sums to 1."""
        var_gaussian_kernel = self.variant(gaussian_kernel)
        sizes = [3, 5, 7]
        sigmas = [0.5, 1.0, 2.0]

        for size in sizes:
            for sigma in sigmas:
                kernel = var_gaussian_kernel(size, sigma)
                kernel_sum = jnp.sum(kernel)
                assert jnp.isclose(
                    kernel_sum, 1.0, atol=1e-6
                ), f"Kernel sum = {kernel_sum} for size={size}, sigma={sigma}"

    @chex.all_variants
    def test_kernel_symmetry(self):
        """Test if kernel is symmetric."""
        var_gaussian_kernel = self.variant(gaussian_kernel)
        size, sigma = 5, 1.0
        kernel = var_gaussian_kernel(size, sigma)

        chex.assert_trees_all_close(kernel, jnp.flip(kernel, axis=0))
        chex.assert_trees_all_close(kernel, jnp.flip(kernel, axis=1))
        chex.assert_trees_all_close(kernel, kernel.T)

    @chex.all_variants
    def test_kernel_center_maximum(self):
        """Test if the center of the kernel has the maximum value."""
        var_gaussian_kernel = self.variant(gaussian_kernel)
        sizes = [3, 5, 7]
        sigma = 1.0

        for size in sizes:
            kernel = var_gaussian_kernel(size, sigma)
            center = size // 2
            center_value = kernel[center, center]
            assert jnp.all(
                kernel <= center_value
            ), f"Center value {center_value} is not maximum for size={size}"

    @chex.all_variants
    @parameterized.parameters(
        {"size": 3, "sigma": 0.1},
        {"size": 3, "sigma": 10.0},
        {"size": 11, "sigma": 0.5},
        {"size": 11, "sigma": 5.0},
    )
    def test_extreme_parameters(self, size, sigma):
        """Test kernel behavior with extreme parameters."""
        var_gaussian_kernel = self.variant(gaussian_kernel)
        kernel = var_gaussian_kernel(size, sigma)

        chex.assert_tree_all_finite(kernel)
        assert jnp.isclose(jnp.sum(kernel), 1.0, atol=1e-6)

    @chex.all_variants
    def test_kernel_values(self):
        """Test specific known kernel values."""
        var_gaussian_kernel = self.variant(gaussian_kernel)
        size, sigma = 3, 1.0
        kernel = var_gaussian_kernel(size, sigma)

        center_value = kernel[1, 1]
        corner_value = kernel[0, 0]
        assert (
            center_value > corner_value
        ), f"Center value {center_value} not greater than corner value {corner_value}"

    @chex.all_variants
    def test_kernel_dtype(self):
        """Test if kernel has correct dtype."""
        var_gaussian_kernel = self.variant(gaussian_kernel)
        size, sigma = 5, 1.0
        kernel = var_gaussian_kernel(size, sigma)
        assert isinstance(kernel, jnp.ndarray)
        assert kernel.dtype == jnp.float32 or kernel.dtype == jnp.float64

    @chex.all_variants
    def test_gradient_computation(self):
        """Test if gradients can be computed with respect to sigma."""
        var_gaussian_kernel = self.variant(gaussian_kernel)

        def loss_fn(sigma):
            kernel = var_gaussian_kernel(5, sigma)
            return jnp.sum(kernel)

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(1.0)
        chex.assert_tree_all_finite(grad)

    @chex.all_variants
    def test_deterministic_output(self):
        """Test if the function produces consistent results."""
        var_gaussian_kernel = self.variant(gaussian_kernel)
        kernel1 = var_gaussian_kernel(5, 1.0)
        kernel2 = var_gaussian_kernel(5, 1.0)
        chex.assert_trees_all_close(kernel1, kernel2)

    @chex.all_variants
    def test_decreasing_values(self):
        """Test if values decrease monotonically from center."""
        var_gaussian_kernel = self.variant(gaussian_kernel)
        size = 5
        sigma = 1.0
        kernel = var_gaussian_kernel(size, sigma)
        center = size // 2

        def check_monotonic(row):
            assert jnp.all(jnp.diff(row[:center]) >= -1e-6)
            assert jnp.all(jnp.diff(row[center:]) <= 1e-6)

        check_monotonic(kernel[center, :])
        check_monotonic(kernel[:, center])


class test_wiener(chex.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.rng = random.PRNGKey(0)
        self.test_image = random.uniform(self.rng, (10, 10))

    @chex.all_variants
    @parameterized.parameters(
        {"shape": (10, 10), "kernel_size": 3},
        {"shape": (32, 48), "kernel_size": 5},
        {"shape": (100, 150), "kernel_size": (3, 5)},
        {"shape": (3, 5), "kernel_size": (5, 3)},
    )
    def test_output_shapes(self, shape, kernel_size):
        """Test if output shape matches input shape for various configurations."""
        var_wiener = self.variant(wiener)
        image = random.uniform(self.rng, shape)
        result = var_wiener(image, kernel_size)
        chex.assert_shape(result, shape)

    @chex.all_variants
    @parameterized.parameters(
        {"noise_level": 0.1, "kernel_size": 3},
        {"noise_level": 0.5, "kernel_size": 5},
        {"noise_level": 1.0, "kernel_size": (3, 5)},
    )
    def test_noise_reduction(self, noise_level, kernel_size):
        """Test if filter reduces noise in image."""
        var_wiener = self.variant(wiener)

        clean_image = jnp.ones((20, 20))
        noise = random.normal(self.rng, clean_image.shape) * noise_level
        noisy_image = clean_image + noise

        filtered = var_wiener(noisy_image, kernel_size)

        noisy_variance = jnp.var(noisy_image)
        filtered_variance = jnp.var(filtered)
        assert (
            filtered_variance < noisy_variance
        ), f"Filtered variance ({filtered_variance}) not less than noisy variance ({noisy_variance})"

    @chex.all_variants
    def test_constant_image(self):
        """Test filter behavior on constant image."""
        var_wiener = self.variant(wiener)
        constant_image = jnp.full((10, 10), 1.0)
        filtered = var_wiener(constant_image)

        chex.assert_trees_all_close(filtered, constant_image, atol=1e-5)

    @chex.all_variants
    def test_kernel_size_handling(self):
        """Test different kernel size specifications."""
        var_wiener = self.variant(wiener)
        image = self.test_image

        result1 = var_wiener(image, kernel_size=3)
        result2 = var_wiener(image, kernel_size=(3, 3))

        chex.assert_trees_all_close(result1, result2)

    @chex.all_variants
    def test_noise_parameter(self):
        """Test explicit noise parameter vs automatic estimation."""
        var_wiener = self.variant(wiener)
        noisy_image = (
            self.test_image + random.normal(self.rng, self.test_image.shape) * 0.1
        )

        result1 = var_wiener(noisy_image)
        result2 = var_wiener(noisy_image, noise=0.1)

        assert not jnp.allclose(result1, result2)
        chex.assert_tree_all_finite(result1)
        chex.assert_tree_all_finite(result2)

    @chex.all_variants
    def test_gradient_computation(self):
        """Test if gradients can be computed."""
        var_wiener = self.variant(wiener)

        def loss_fn(image):
            filtered = var_wiener(image)
            return jnp.sum(filtered)

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(self.test_image)

        chex.assert_shape(grads, self.test_image.shape)
        chex.assert_tree_all_finite(grads)

    @chex.all_variants
    def test_dtype_consistency(self):
        """Test if function preserves dtype."""
        var_wiener = self.variant(wiener)
        image = self.test_image.astype(jnp.float64)
        result = var_wiener(image)
        assert result.dtype == image.dtype

    @chex.all_variants
    def test_edge_preservation(self):
        """Test if filter preserves strong edges."""
        var_wiener = self.variant(wiener)

        edge_image = jnp.zeros((20, 20))
        edge_image = edge_image.at[:, 10:].set(1.0)

        noisy_edge = edge_image + random.normal(self.rng, edge_image.shape) * 0.1

        filtered = var_wiener(noisy_edge)

        middle_row = filtered[10, :]
        edge_location = jnp.argmax(jnp.abs(jnp.diff(middle_row)))
        assert 9 <= edge_location <= 11, f"Edge not preserved at expected location"

    @chex.all_variants
    def test_deterministic_output(self):
        """Test if function produces consistent results."""
        var_wiener = self.variant(wiener)
        result1 = var_wiener(self.test_image)
        result2 = var_wiener(self.test_image)
        chex.assert_trees_all_close(result1, result2)

    @chex.all_variants
    def test_extreme_values(self):
        """Test function behavior with extreme input values."""
        var_wiener = self.variant(wiener)

        large_image = self.test_image * 1e5
        large_result = var_wiener(large_image)
        chex.assert_tree_all_finite(large_result)

        small_image = self.test_image * 1e-5
        small_result = var_wiener(small_image)
        chex.assert_tree_all_finite(small_result)
