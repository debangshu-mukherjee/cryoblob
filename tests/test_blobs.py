"""
Tests for cryoblob.blobs module

This module tests blob detection algorithms and preprocessing functions.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jaxtyping import Array, Bool, Float, Integer

import cryoblob as cb
from cryoblob.types import MRC_Image, make_MRC_Image, scalar_float, scalar_int


class TestConnectedComponents(chex.TestCase, parameterized.TestCase):
    """Test connected components labeling."""
    
    def setUp(self):
        super().setUp()
        # Simple 3D binary image with two disconnected components
        self.simple_binary = jnp.zeros((5, 5, 5), dtype=bool)
        self.simple_binary = self.simple_binary.at[1:3, 1:3, 1:3].set(True)  # First component
        self.simple_binary = self.simple_binary.at[3:5, 3:5, 3:5].set(True)  # Second component
    
    @chex.all_variants
    def test_find_connected_components_basic(self):
        """Test basic connected components functionality."""
        def find_components(binary_image):
            return cb.find_connected_components(binary_image, connectivity=6)
        
        labels, num_labels = self.variant(find_components)(self.simple_binary)
        
        assert labels.shape == self.simple_binary.shape
        assert num_labels == 2  # Two separate components
        
        # Check that background is labeled 0
        assert jnp.all(labels[~self.simple_binary] == 0)
        
        # Check that each component has a unique label
        unique_labels = jnp.unique(labels)
        assert len(unique_labels) == 3  # 0 (background), 1, 2
    
    @parameterized.parameters(6, 26)
    @chex.all_variants
    def test_connectivity_modes(self, connectivity):
        """Test different connectivity modes."""
        # Create touching cubes (face-connected but not fully connected)
        binary = jnp.zeros((5, 5, 5), dtype=bool)
        binary = binary.at[1:3, 1:3, 1:3].set(True)
        binary = binary.at[2:4, 1:3, 1:3].set(True)  # Overlaps with first
        
        def find_components(img):
            return cb.find_connected_components(img, connectivity=connectivity)
        
        labels, num_labels = self.variant(find_components)(binary)
        
        # With any connectivity, overlapping regions should be one component
        assert num_labels == 1
    
    @chex.all_variants
    def test_empty_image(self):
        """Test connected components on empty image."""
        empty = jnp.zeros((5, 5, 5), dtype=bool)
        
        def find_components(img):
            return cb.find_connected_components(img)
        
        labels, num_labels = self.variant(find_components)(empty)
        
        assert jnp.all(labels == 0)
        assert num_labels == 0
    
    @chex.all_variants
    def test_full_image(self):
        """Test connected components on fully connected image."""
        full = jnp.ones((5, 5, 5), dtype=bool)
        
        def find_components(img):
            return cb.find_connected_components(img)
        
        labels, num_labels = self.variant(find_components)(full)
        
        assert num_labels == 1
        assert jnp.all(labels == 1)


class TestCenterOfMass(chex.TestCase, parameterized.TestCase):
    """Test center of mass calculations."""
    
    @chex.all_variants
    def test_center_of_mass_single_component(self):
        """Test center of mass for single component."""
        # Create a simple 3D image with one labeled region
        image = jnp.zeros((5, 5, 5))
        labels = jnp.zeros((5, 5, 5), dtype=jnp.int32)
        
        # Set a cube with uniform intensity
        image = image.at[1:4, 1:4, 1:4].set(1.0)
        labels = labels.at[1:4, 1:4, 1:4].set(1)
        
        def compute_com(img, lbl):
            return cb.center_of_mass_3d(img, lbl, num_labels=1)
        
        centroids = self.variant(compute_com)(image, labels)
        
        assert centroids.shape == (1, 3)
        # Center should be at (2, 2, 2) for a 3x3x3 cube starting at (1,1,1)
        expected = jnp.array([[2.0, 2.0, 2.0]])
        chex.assert_trees_all_close(centroids, expected, atol=1e-5)
    
    @chex.all_variants
    def test_center_of_mass_multiple_components(self):
        """Test center of mass for multiple components."""
        image = jnp.ones((10, 10, 10))
        labels = jnp.zeros((10, 10, 10), dtype=jnp.int32)
        
        # Create two separate components
        labels = labels.at[1:3, 1:3, 1:3].set(1)
        labels = labels.at[7:9, 7:9, 7:9].set(2)
        
        def compute_com(img, lbl):
            return cb.center_of_mass_3d(img, lbl, num_labels=2)
        
        centroids = self.variant(compute_com)(image, labels)
        
        assert centroids.shape == (2, 3)
        # Check approximate centers
        expected = jnp.array([[1.5, 1.5, 1.5], [7.5, 7.5, 7.5]])
        chex.assert_trees_all_close(centroids, expected, atol=1e-5)
    
    @chex.all_variants
    def test_center_of_mass_weighted(self):
        """Test center of mass with non-uniform weights."""
        image = jnp.zeros((5, 5, 5))
        labels = jnp.zeros((5, 5, 5), dtype=jnp.int32)
        
        # Create gradient in one direction
        for i in range(3):
            image = image.at[i+1, 1:4, 1:4].set(float(i+1))
        labels = labels.at[1:4, 1:4, 1:4].set(1)
        
        def compute_com(img, lbl):
            return cb.center_of_mass_3d(img, lbl, num_labels=1)
        
        centroids = self.variant(compute_com)(image, labels)
        
        # Center should be weighted towards higher intensity
        assert centroids[0, 0] > 2.0  # Pulled towards higher z values


class TestFindParticleCoords(chex.TestCase, parameterized.TestCase):
    """Test particle coordinate finding."""
    
    def setUp(self):
        super().setUp()
        # Create 3D image with peaks
        self.results_3d = jnp.zeros((10, 10, 10))
        self.results_3d = self.results_3d.at[5, 5, 5].set(10.0)
        self.results_3d = self.results_3d.at[2, 2, 2].set(8.0)
        
        # Create max filtered version
        self.max_filtered = self.results_3d.copy()
    
    @chex.all_variants
    def test_find_particle_coords_basic(self):
        """Test basic particle finding."""
        def find_particles(results, max_filt, thresh):
            return cb.find_particle_coords(results, max_filt, thresh)
        
        # Set threshold to catch both peaks
        coords = self.variant(find_particles)(
            self.results_3d, self.max_filtered, 5.0
        )
        
        assert coords.shape[1] == 3  # 3D coordinates
        assert len(coords) <= 2  # At most 2 peaks
    
    @parameterized.parameters(1.0, 5.0, 9.0)
    @chex.all_variants
    def test_find_particle_coords_thresholds(self, threshold):
        """Test particle finding with different thresholds."""
        def find_particles(results, max_filt):
            return cb.find_particle_coords(results, max_filt, threshold)
        
        coords = self.variant(find_particles)(
            self.results_3d, self.max_filtered
        )
        
        # Higher thresholds should find fewer particles
        if threshold > 8.0:
            assert len(coords) <= 1
        elif threshold > 5.0:
            assert len(coords) <= 2


class TestPreprocessing(chex.TestCase, parameterized.TestCase):
    """Test image preprocessing."""
    
    def setUp(self):
        super().setUp()
        # Create test image with some structure
        x, y = jnp.meshgrid(jnp.linspace(-1, 1, 20), jnp.linspace(-1, 1, 20))
        self.test_image = jnp.exp(-(x**2 + y**2) / 0.5) + 0.1
    
    @chex.all_variants
    def test_preprocessing_basic(self):
        """Test basic preprocessing."""
        def preprocess(image):
            return cb.preprocessing(image, return_params=False)
        
        processed = self.variant(preprocess)(self.test_image)
        
        assert processed.shape == self.test_image.shape
        # Should be normalized to [0, 1] range initially
        assert jnp.min(processed) >= 0
        assert jnp.max(processed) <= jnp.exp(1)  # Due to exponential
    
    @parameterized.parameters(
        (True, False, 0, 0, 0),  # Exponential only
        (False, True, 0, 0, 0),  # Logarithm only
        (False, False, 2, 0, 0),  # Gaussian blur only
        (False, False, 0, 5, 0),  # Background subtraction only
        (False, False, 0, 0, 3),  # Wiener filter only
    )
    @chex.all_variants
    def test_preprocessing_options(self, exponential, logarizer, gblur, background, apply_filter):
        """Test different preprocessing options."""
        def preprocess(image):
            return cb.preprocessing(
                image,
                exponential=exponential,
                logarizer=logarizer,
                gblur=gblur,
                background=background,
                apply_filter=apply_filter
            )
        
        processed = self.variant(preprocess)(self.test_image)
        
        assert processed.shape == self.test_image.shape
        chex.assert_tree_all_finite(processed)
    
    @chex.all_variants
    def test_preprocessing_with_params(self):
        """Test preprocessing with parameter return."""
        def preprocess(image):
            return cb.preprocessing(image, return_params=True)
        
        processed, params = self.variant(preprocess)(self.test_image)
        
        assert processed.shape == self.test_image.shape
        assert isinstance(params, dict)
        assert 'exponential' in params
        assert 'logarizer' in params
        assert 'gblur' in params
        assert 'background' in params
        assert 'apply_filter' in params
    
    @chex.all_variants
    def test_preprocessing_constant_image(self):
        """Test preprocessing on constant image."""
        constant_image = jnp.ones((10, 10)) * 0.5
        
        def preprocess(image):
            return cb.preprocessing(image)
        
        processed = self.variant(preprocess)(constant_image)
        
        # Constant image should become zeros after normalization
        chex.assert_trees_all_close(processed, jnp.zeros_like(processed), atol=1e-6)


class TestBlobListLog(chex.TestCase, parameterized.TestCase):
    """Test blob detection using Laplacian of Gaussian."""
    
    def setUp(self):
        super().setUp()
        # Create synthetic image with blobs
        x, y = jnp.meshgrid(jnp.linspace(-10, 10, 100), jnp.linspace(-10, 10, 100))
        
        # Add multiple Gaussian blobs
        blob1 = 100 * jnp.exp(-((x-3)**2 + (y-3)**2) / 4)
        blob2 = 80 * jnp.exp(-((x+3)**2 + (y+3)**2) / 9)
        blob3 = 60 * jnp.exp(-((x-3)**2 + (y+3)**2) / 16)
        
        image_data = blob1 + blob2 + blob3 + 10  # Add background
        
        self.mrc_image = make_MRC_Image(
            image_data=image_data,
            voxel_size=jnp.array([1.0, 0.1, 0.1]),  # 0.1 nm per pixel
            origin=jnp.zeros(3),
            data_min=jnp.min(image_data),
            data_max=jnp.max(image_data),
            data_mean=jnp.mean(image_data),
            mode=2
        )
    
    @chex.all_variants
    def test_blob_list_log_basic(self):
        """Test basic blob detection."""
        def detect_blobs(mrc):
            return cb.blob_list_log(
                mrc,
                min_blob_size=5,
                max_blob_size=20,
                blob_step=2,
                downscale=2,
                std_threshold=3
            )
        
        blobs = self.variant(detect_blobs)(self.mrc_image)
        
        assert blobs.shape[1] == 3  # (Y, X, size)
        assert len(blobs) > 0  # Should detect at least one blob
        
        # Check that blob sizes are reasonable
        sizes = blobs[:, 2]
        assert jnp.all(sizes > 0)
    
    @parameterized.parameters(
        (5, 15, 1),   # Fine scale search
        (10, 30, 5),  # Coarse scale search
        (3, 10, 2),   # Small blob search
    )
    @chex.all_variants
    def test_blob_list_log_scales(self, min_size, max_size, step):
        """Test blob detection with different scale parameters."""
        def detect_blobs(mrc):
            return cb.blob_list_log(
                mrc,
                min_blob_size=min_size,
                max_blob_size=max_size,
                blob_step=step,
                downscale=2
            )
        
        blobs = self.variant(detect_blobs)(self.mrc_image)
        
        assert blobs.shape[1] == 3
        # Blob sizes should be in expected range (accounting for voxel scaling)
        if len(blobs) > 0:
            sizes = blobs[:, 2]
            voxel_scale = jnp.sqrt(self.mrc_image.voxel_size[1] * self.mrc_image.voxel_size[2])
            min_expected = min_size * voxel_scale
            max_expected = max_size * voxel_scale
            
            # Some tolerance due to LoG response
            assert jnp.all(sizes >= min_expected * 0.5)
            assert jnp.all(sizes <= max_expected * 2.0)
    
    @parameterized.parameters(1, 2, 4, 8)
    @chex.all_variants
    def test_blob_list_log_downscale(self, downscale):
        """Test blob detection with different downscaling factors."""
        def detect_blobs(mrc):
            return cb.blob_list_log(
                mrc,
                min_blob_size=5,
                max_blob_size=20,
                downscale=downscale
            )
        
        blobs = self.variant(detect_blobs)(self.mrc_image)
        
        assert blobs.shape[1] == 3
        
        # Coordinates should be scaled back to original size
        if len(blobs) > 0:
            coords = blobs[:, :2]
            # Check coordinates are within image bounds
            image_shape = self.mrc_image.image_data.shape
            max_coords = jnp.array([
                image_shape[0] * self.mrc_image.voxel_size[1],
                image_shape[1] * self.mrc_image.voxel_size[2]
            ])
            assert jnp.all(coords >= 0)
            assert jnp.all(coords <= max_coords)
    
    @chex.all_variants
    def test_blob_list_log_empty_image(self):
        """Test blob detection on image with no blobs."""
        # Create empty image
        empty_mrc = make_MRC_Image(
            image_data=jnp.ones((100, 100)) * 10,  # Constant image
            voxel_size=jnp.array([1.0, 0.1, 0.1]),
            origin=jnp.zeros(3),
            data_min=10.0,
            data_max=10.0,
            data_mean=10.0,
            mode=2
        )
        
        def detect_blobs(mrc):
            return cb.blob_list_log(mrc, std_threshold=6)
        
        blobs = self.variant(detect_blobs)(empty_mrc)
        
        # Should detect no blobs in constant image
        assert len(blobs) == 0


class TestIntegration(chex.TestCase, parameterized.TestCase):
    """Integration tests for complete blob detection pipeline."""
    
    @chex.all_variants
    def test_full_pipeline(self):
        """Test complete preprocessing and blob detection pipeline."""
        # Create test image
        x, y = jnp.meshgrid(jnp.linspace(-5, 5, 50), jnp.linspace(-5, 5, 50))
        image = jnp.exp(-(x**2 + y**2) / 2) + 0.1 * jax.random.normal(jax.random.PRNGKey(42), (50, 50))
        
        def full_pipeline(img):
            # Preprocess
            processed = cb.preprocessing(
                img,
                exponential=True,
                gblur=2,
                background=5
            )
            
            # Create MRC image
            mrc = make_MRC_Image(
                image_data=processed,
                voxel_size=jnp.array([1.0, 1.0, 1.0]),
                origin=jnp.zeros(3),
                data_min=jnp.min(processed),
                data_max=jnp.max(processed),
                data_mean=jnp.mean(processed),
                mode=2
            )
            
            # Detect blobs
            blobs = cb.blob_list_log(mrc, min_blob_size=3, max_blob_size=10)
            
            return blobs
        
        blobs = self.variant(full_pipeline)(image)
        
        # Should detect the central blob
        assert len(blobs) >= 1
        
        # Check that detected blob is near center
        if len(blobs) > 0:
            # Find blob closest to center
            center = jnp.array([25.0, 25.0])
            distances = jnp.linalg.norm(blobs[:, :2] - center, axis=1)
            closest_idx = jnp.argmin(distances)
            closest_distance = distances[closest_idx]
            
            # Should be within a few pixels of center
            assert closest_distance < 5.0


if __name__ == "__main__":
    from absl.testing import absltest
    absltest.main()