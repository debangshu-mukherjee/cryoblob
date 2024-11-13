import jax
import jax.numpy as jnp
from jax import random
from absl.testing import parameterized
import chex
import pytest

jax.config.update("jax_enable_x64", True)

# Import your functions here
from arm_em import fast_resizer, gaussian_kernel

if __name__ == "__main__":
    pytest.main([__file__])


class test_fast_resizer(chex.TestCase):
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
        {"shape": (3, 5), "sampling": jnp.array([[0.5, 0.75]]), "expected_shape": (6, 7)},
    )
    def test_output_shapes(self, shape, sampling, expected_shape):
        var_fast_resizer = self.variant(fast_resizer)
        image = random.uniform(self.rng, shape)
        result = var_fast_resizer(image, sampling)
        chex.assert_shape(result, expected_shape)

    @chex.all_variants
    @parameterized.parameters(
        {"image": jnp.array([[1., 2.], [3., 4.]]), 
         "sampling": 2.0, 
         "expected": jnp.array([[2.5]])},
        {"image": jnp.array([[1., 1.], [1., 1.]]), 
         "sampling": 1.0, 
         "expected": jnp.array([[1., 1.], [1., 1.]])},
    )
    def test_known_values(self, image, sampling, expected):
        var_fast_resizer = self.variant(fast_resizer)
        result = var_fast_resizer(image, sampling)
        chex.assert_trees_all_close(result, expected, atol=1e-5)

    @chex.all_variants
    def test_batch_consistency(self):
        var_fast_resizer = self.variant(fast_resizer)
        batch_size = 3
        images = random.uniform(self.rng, (batch_size,) + self.base_shape)
        
        # Test with vmap
        batch_resize = jax.vmap(lambda x: var_fast_resizer(x, 0.5))
        results = batch_resize(images)
        
        # Compare with individual processing
        for i in range(batch_size):
            individual_result = var_fast_resizer(images[i], 0.5)
            chex.assert_trees_all_close(results[i], individual_result)

    @chex.all_variants
    def test_dtype_consistency(self):
        var_fast_resizer = self.variant(fast_resizer)
        dtypes = [jnp.float32, jnp.float64]
        
        for dtype in dtypes:
            image = self.base_image.astype(dtype)
            result = var_fast_resizer(image, 0.5)
            assert result.dtype == dtype, f"Expected dtype {dtype}, got {result.dtype}"

    @chex.all_variants
    @parameterized.parameters(
        {"sampling": 0.1},  # Very small sampling rate
        {"sampling": 10.0},  # Very large sampling rate
        {"sampling": (0.1, 10.0)},  # Mixed sampling rates
    )
    def test_extreme_sampling(self, sampling):
        var_fast_resizer = self.variant(fast_resizer)
        result = var_fast_resizer(self.base_image, sampling)
        chex.assert_tree_all_finite(result)

    @chex.all_variants
    def test_gradient_computation(self):
        var_fast_resizer = self.variant(fast_resizer)
        
        def loss_fn(image):
            resized = var_fast_resizer(image, 0.5)
            return jnp.sum(resized)
        
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(self.base_image)
        
        chex.assert_shape(grads, self.base_shape)
        chex.assert_tree_all_finite(grads)

    @chex.all_variants
    def test_deterministic_output(self):
        var_fast_resizer = self.variant(fast_resizer)
        result1 = var_fast_resizer(self.base_image, 0.5)
        result2 = var_fast_resizer(self.base_image, 0.5)
        chex.assert_trees_all_close(result1, result2)

    @chex.all_variants
    def test_image_range_preservation(self):
        """Test that the resizer preserves the general range of values."""
        var_fast_resizer = self.variant(fast_resizer)
        test_image = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        result = var_fast_resizer(test_image, 0.5)
        
        # Check that output values are in reasonable range
        assert jnp.all(result >= test_image.min() * 0.9)  # Allow for small numerical errors
        assert jnp.all(result <= test_image.max() * 1.1)  # Allow for small numerical errors
        

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
                assert jnp.isclose(kernel_sum, 1.0, atol=1e-6), \
                    f"Kernel sum = {kernel_sum} for size={size}, sigma={sigma}"

    @chex.all_variants
    def test_kernel_symmetry(self):
        """Test if kernel is symmetric."""
        var_gaussian_kernel = self.variant(gaussian_kernel)
        size, sigma = 5, 1.0
        kernel = var_gaussian_kernel(size, sigma)
        
        # Test horizontal symmetry
        chex.assert_trees_all_close(kernel, jnp.flip(kernel, axis=0))
        # Test vertical symmetry
        chex.assert_trees_all_close(kernel, jnp.flip(kernel, axis=1))
        # Test diagonal symmetry
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
            assert jnp.all(kernel <= center_value), \
                f"Center value {center_value} is not maximum for size={size}"

    @chex.all_variants
    @parameterized.parameters(
        {"size": 3, "sigma": 0.1},  # Small sigma
        {"size": 3, "sigma": 10.0},  # Large sigma
        {"size": 11, "sigma": 0.5},  # Large size, small sigma
        {"size": 11, "sigma": 5.0},  # Large size, large sigma
    )
    def test_extreme_parameters(self, size, sigma):
        """Test kernel behavior with extreme parameters."""
        var_gaussian_kernel = self.variant(gaussian_kernel)
        kernel = var_gaussian_kernel(size, sigma)
        
        # Check if output is finite
        chex.assert_tree_all_finite(kernel)
        # Check normalization
        assert jnp.isclose(jnp.sum(kernel), 1.0, atol=1e-6)

    @chex.all_variants
    def test_kernel_values(self):
        """Test specific known kernel values."""
        var_gaussian_kernel = self.variant(gaussian_kernel)
        size, sigma = 3, 1.0
        kernel = var_gaussian_kernel(size, sigma)
        
        # Center value should be largest
        center_value = kernel[1, 1]
        corner_value = kernel[0, 0]
        assert center_value > corner_value, \
            f"Center value {center_value} not greater than corner value {corner_value}"

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
            # Check left side
            assert jnp.all(jnp.diff(row[:center]) >= -1e-6)
            # Check right side
            assert jnp.all(jnp.diff(row[center:]) <= 1e-6)
        
        # Check center row
        check_monotonic(kernel[center, :])
        # Check center column
        check_monotonic(kernel[:, center])