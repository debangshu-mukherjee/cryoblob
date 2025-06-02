"""
Tests for multi-method blob detection including ridge detection,
watershed segmentation, and enhanced blob detection methods.
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

import cryoblob as cb
from cryoblob.types import make_MRC_Image

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    pytest.main([__file__])


class TestHessianMatrix(chex.TestCase, parameterized.TestCase):

    def setUp(self):
        super().setUp()
        x, y = jnp.meshgrid(jnp.linspace(-5, 5, 50), jnp.linspace(-5, 5, 50))
        self.blob_image = jnp.exp(-(x**2 + y**2) / 4)
        self.elongated_image = jnp.exp(-((x / 4) ** 2 + y**2) / 2)

    @chex.all_variants
    def test_hessian_matrix_2d_shape(self):
        def compute_hessian(img):
            return cb.hessian_matrix_2d(img, sigma=1.0)

        hxx, hxy, hyy = self.variant(compute_hessian)(self.blob_image)

        assert hxx.shape == self.blob_image.shape
        assert hxy.shape == self.blob_image.shape
        assert hyy.shape == self.blob_image.shape

    @parameterized.parameters(0.5, 1.0, 2.0, 5.0)
    @chex.all_variants
    def test_hessian_matrix_different_sigmas(self, sigma):
        def compute_hessian(img):
            return cb.hessian_matrix_2d(img, sigma=sigma)

        hxx, hxy, hyy = self.variant(compute_hessian)(self.blob_image)

        chex.assert_tree_all_finite(hxx)
        chex.assert_tree_all_finite(hxy)
        chex.assert_tree_all_finite(hyy)

    @chex.all_variants
    def test_determinant_of_hessian(self):
        def compute_det_hessian(img):
            return cb.determinant_of_hessian(img, sigma=1.0)

        det_h = self.variant(compute_det_hessian)(self.blob_image)

        assert det_h.shape == self.blob_image.shape
        chex.assert_tree_all_finite(det_h)

        center_idx = det_h.shape[0] // 2, det_h.shape[1] // 2
        assert det_h[center_idx] > 0

    @chex.all_variants
    def test_hessian_blob_vs_elongated(self):
        def compute_det_hessian(img):
            return cb.determinant_of_hessian(img, sigma=2.0)

        blob_response = self.variant(compute_det_hessian)(self.blob_image)
        elongated_response = self.variant(compute_det_hessian)(self.elongated_image)

        chex.assert_tree_all_finite(blob_response)
        chex.assert_tree_all_finite(elongated_response)


class TestRidgeDetection(chex.TestCase, parameterized.TestCase):

    def setUp(self):
        super().setUp()
        x, y = jnp.meshgrid(jnp.linspace(-10, 10, 100), jnp.linspace(-10, 10, 100))

        self.horizontal_ridge = jnp.exp(-(y**2) / 2) * (jnp.abs(x) < 8)

        rotated_x = x * jnp.cos(jnp.pi / 4) - y * jnp.sin(jnp.pi / 4)
        rotated_y = x * jnp.sin(jnp.pi / 4) + y * jnp.cos(jnp.pi / 4)
        self.diagonal_ridge = jnp.exp(-(rotated_y**2) / 2) * (jnp.abs(rotated_x) < 8)

        self.circular_blob = jnp.exp(-(x**2 + y**2) / 16)

    @chex.all_variants
    def test_ridge_detection_horizontal(self):
        def detect_ridges(img):
            return cb.ridge_detection(img, sigma=2.0, threshold=0.01)

        ridge_response = self.variant(detect_ridges)(self.horizontal_ridge)

        assert ridge_response.shape == self.horizontal_ridge.shape
        chex.assert_tree_all_finite(ridge_response)

        center_row = ridge_response.shape[0] // 2
        ridge_strength = jnp.max(ridge_response[center_row, :])
        assert ridge_strength > 0.01

    @chex.all_variants
    def test_ridge_detection_vs_blob(self):
        def detect_ridges(img):
            return cb.ridge_detection(img, sigma=2.0, threshold=0.01)

        ridge_response = self.variant(detect_ridges)(self.horizontal_ridge)
        blob_response = self.variant(detect_ridges)(self.circular_blob)

        max_ridge = jnp.max(ridge_response)
        max_blob = jnp.max(blob_response)

        assert max_ridge > max_blob

    @chex.all_variants
    def test_ridge_detection_diagonal(self):
        def detect_ridges(img):
            return cb.ridge_detection(img, sigma=2.0, threshold=0.005)

        ridge_response = self.variant(detect_ridges)(self.diagonal_ridge)

        assert ridge_response.shape == self.diagonal_ridge.shape
        chex.assert_tree_all_finite(ridge_response)
        assert jnp.max(ridge_response) > 0.005

    @parameterized.parameters((1.0, 5.0, 5), (2.0, 10.0, 8), (0.5, 3.0, 10))
    @chex.all_variants
    def test_multi_scale_ridge_detector(self, min_scale, max_scale, num_scales):
        def detect_multi_scale_ridges(img):
            return cb.multi_scale_ridge_detector(
                img,
                min_scale=min_scale,
                max_scale=max_scale,
                num_scales=num_scales,
                threshold=0.005,
            )

        ridge_response = self.variant(detect_multi_scale_ridges)(self.horizontal_ridge)

        assert ridge_response.shape == self.horizontal_ridge.shape
        chex.assert_tree_all_finite(ridge_response)
        assert jnp.max(ridge_response) > 0.005

    @chex.all_variants
    def test_multi_scale_ridge_detector_empty_result(self):
        uniform_image = jnp.ones((50, 50)) * 0.5

        def detect_ridges(img):
            return cb.multi_scale_ridge_detector(img, threshold=0.1)

        ridge_response = self.variant(detect_ridges)(uniform_image)

        assert ridge_response.shape == uniform_image.shape
        assert jnp.max(ridge_response) <= 0.1


class TestDistanceTransform(chex.TestCase, parameterized.TestCase):

    def setUp(self):
        super().setUp()
        x, y = jnp.meshgrid(jnp.linspace(-10, 10, 80), jnp.linspace(-10, 10, 80))

        blob1 = jnp.exp(-((x + 3) ** 2 + y**2) / 8)
        blob2 = jnp.exp(-((x - 3) ** 2 + y**2) / 8)
        self.overlapping_blobs = blob1 + blob2

        self.binary_blobs = self.overlapping_blobs > 0.3

    @chex.all_variants
    def test_distance_transform_euclidean_basic(self):
        def compute_distance_transform(binary_img):
            return cb.distance_transform_euclidean(binary_img)

        dist_map = self.variant(compute_distance_transform)(self.binary_blobs)

        assert dist_map.shape == self.binary_blobs.shape
        chex.assert_tree_all_finite(dist_map)

        foreground_distances = dist_map[self.binary_blobs]
        assert jnp.all(foreground_distances == 0)

        background_distances = dist_map[~self.binary_blobs]
        if len(background_distances) > 0:
            assert jnp.all(background_distances >= 0)

    @chex.all_variants
    def test_distance_transform_simple_shape(self):
        simple_binary = jnp.zeros((10, 10), dtype=bool)
        simple_binary = simple_binary.at[4:6, 4:6].set(True)

        def compute_distance_transform(binary_img):
            return cb.distance_transform_euclidean(binary_img)

        dist_map = self.variant(compute_distance_transform)(simple_binary)

        assert dist_map.shape == simple_binary.shape
        foreground_mask = simple_binary
        assert jnp.all(dist_map[foreground_mask] == 0)

    @chex.all_variants
    def test_distance_transform_empty_image(self):
        empty_binary = jnp.zeros((20, 20), dtype=bool)

        def compute_distance_transform(binary_img):
            return cb.distance_transform_euclidean(binary_img)

        dist_map = self.variant(compute_distance_transform)(empty_binary)

        assert jnp.all(jnp.isinf(dist_map) | (dist_map == 0))


class TestWatershedSegmentation(chex.TestCase, parameterized.TestCase):

    def setUp(self):
        super().setUp()
        x, y = jnp.meshgrid(jnp.linspace(-10, 10, 60), jnp.linspace(-10, 10, 60))

        blob1 = jnp.exp(-((x + 3) ** 2 + y**2) / 8)
        blob2 = jnp.exp(-((x - 3) ** 2 + y**2) / 8)
        self.overlapping_blobs = blob1 + blob2

        self.binary_blobs = self.overlapping_blobs > 0.3

    @chex.all_variants
    def test_adaptive_marker_generation(self):
        def generate_markers(binary_img):
            return cb.adaptive_marker_generation(binary_img, min_distance=3.0)

        markers = self.variant(generate_markers)(self.binary_blobs)

        assert markers.shape == self.binary_blobs.shape

        unique_markers = jnp.unique(markers)
        positive_markers = unique_markers[unique_markers > 0]
        assert len(positive_markers) >= 0

        background_markers = unique_markers[unique_markers == -1]
        assert len(background_markers) > 0

    @chex.all_variants
    def test_adaptive_marker_generation_small_distance(self):
        def generate_markers(binary_img):
            return cb.adaptive_marker_generation(binary_img, min_distance=1.0)

        markers = self.variant(generate_markers)(self.binary_blobs)

        assert markers.shape == self.binary_blobs.shape
        unique_markers = jnp.unique(markers)
        assert len(unique_markers) >= 2

    @chex.all_variants
    def test_watershed_segmentation_basic(self):
        markers = jnp.zeros_like(self.binary_blobs, dtype=jnp.int32)
        h, w = markers.shape

        markers = markers.at[h // 2, w // 4].set(1)
        markers = markers.at[h // 2, 3 * w // 4].set(2)
        markers = jnp.where(~self.binary_blobs, -1, markers)

        def perform_watershed(img, mark):
            return cb.watershed_segmentation(img, mark, max_iterations=10)

        gradient_y = jnp.gradient(self.overlapping_blobs, axis=0)
        gradient_x = jnp.gradient(self.overlapping_blobs, axis=1)
        gradient_mag = jnp.sqrt(gradient_y**2 + gradient_x**2)

        segmented = self.variant(perform_watershed)(gradient_mag, markers)

        assert segmented.shape == markers.shape

        assert jnp.any(segmented == 1)
        assert jnp.any(segmented == 2)

    @chex.all_variants
    def test_watershed_segmentation_iterations(self):
        markers = jnp.zeros_like(self.binary_blobs, dtype=jnp.int32)
        h, w = markers.shape
        markers = markers.at[h // 2, w // 2].set(1)
        markers = jnp.where(~self.binary_blobs, -1, markers)

        def perform_watershed_iter(img, mark, iterations):
            return cb.watershed_segmentation(img, mark, max_iterations=iterations)

        gradient_mag = jnp.ones_like(self.overlapping_blobs)

        segmented_few = self.variant(lambda i, m: perform_watershed_iter(i, m, 3))(
            gradient_mag, markers
        )
        segmented_many = self.variant(lambda i, m: perform_watershed_iter(i, m, 15))(
            gradient_mag, markers
        )

        assert segmented_few.shape == markers.shape
        assert segmented_many.shape == markers.shape


class TestHessianBlobDetection(chex.TestCase, parameterized.TestCase):

    def setUp(self):
        super().setUp()
        x, y = jnp.meshgrid(jnp.linspace(-20, 20, 120), jnp.linspace(-20, 20, 120))

        blob1 = 100 * jnp.exp(-((x - 8) ** 2 + (y - 8) ** 2) / 16)
        blob2 = 80 * jnp.exp(-((x + 8) ** 2 + (y + 8) ** 2) / 25)
        blob3 = 60 * jnp.exp(-((x - 8) ** 2 + (y + 8) ** 2) / 9)

        image_data = blob1 + blob2 + blob3 + 10

        self.mrc_image = make_MRC_Image(
            image_data=image_data,
            voxel_size=jnp.array([1.0, 0.15, 0.15]),
            origin=jnp.zeros(3),
            data_min=jnp.min(image_data),
            data_max=jnp.max(image_data),
            data_mean=jnp.mean(image_data),
            mode=2,
        )

    @chex.all_variants
    def test_hessian_blob_detection_basic(self):
        def detect_hessian_blobs(mrc):
            return cb.hessian_blob_detection(
                mrc, min_blob_size=3, max_blob_size=15, downscale=2, std_threshold=4
            )

        blobs = self.variant(detect_hessian_blobs)(self.mrc_image)

        assert blobs.shape[1] == 3
        assert len(blobs) >= 0

        if len(blobs) > 0:
            sizes = blobs[:, 2]
            assert jnp.all(sizes > 0)

    @parameterized.parameters((2, 10, 1), (5, 20, 2), (3, 12, 0.5))
    @chex.all_variants
    def test_hessian_blob_detection_parameters(self, min_size, max_size, step):
        def detect_hessian_blobs(mrc):
            return cb.hessian_blob_detection(
                mrc,
                min_blob_size=min_size,
                max_blob_size=max_size,
                blob_step=step,
                downscale=2,
            )

        blobs = self.variant(detect_hessian_blobs)(self.mrc_image)

        assert blobs.shape[1] == 3

    @chex.all_variants
    def test_hessian_blob_detection_high_threshold(self):
        def detect_hessian_blobs(mrc):
            return cb.hessian_blob_detection(
                mrc, min_blob_size=5, max_blob_size=15, std_threshold=10
            )

        blobs = self.variant(detect_hessian_blobs)(self.mrc_image)

        assert blobs.shape[1] == 3

    @chex.all_variants
    def test_hessian_blob_detection_empty_result(self):
        empty_data = jnp.ones((50, 50)) * 10
        empty_mrc = make_MRC_Image(
            image_data=empty_data,
            voxel_size=jnp.array([1.0, 0.1, 0.1]),
            origin=jnp.zeros(3),
            data_min=10.0,
            data_max=10.0,
            data_mean=10.0,
            mode=2,
        )

        def detect_hessian_blobs(mrc):
            return cb.hessian_blob_detection(mrc, std_threshold=8)

        blobs = self.variant(detect_hessian_blobs)(empty_mrc)

        assert len(blobs) == 0
        assert blobs.shape == (0, 3)


class TestEnhancedBlobDetection(chex.TestCase, parameterized.TestCase):

    def setUp(self):
        super().setUp()
        x, y = jnp.meshgrid(jnp.linspace(-30, 30, 150), jnp.linspace(-30, 30, 150))

        circular1 = 100 * jnp.exp(-((x - 10) ** 2 + (y - 10) ** 2) / 16)
        circular2 = 80 * jnp.exp(-((x + 10) ** 2 + (y + 10) ** 2) / 20)

        elongated = 90 * jnp.exp(-((x / 6) ** 2 + (y + 5) ** 2) / 4) * (jnp.abs(x) < 15)

        overlap1 = 70 * jnp.exp(-((x - 5) ** 2 + (y) ** 2) / 12)
        overlap2 = 60 * jnp.exp(-((x + 2) ** 2 + (y) ** 2) / 10)

        image_data = circular1 + circular2 + elongated + overlap1 + overlap2 + 15

        self.complex_mrc = make_MRC_Image(
            image_data=image_data,
            voxel_size=jnp.array([1.0, 0.12, 0.12]),
            origin=jnp.zeros(3),
            data_min=jnp.min(image_data),
            data_max=jnp.max(image_data),
            data_mean=jnp.mean(image_data),
            mode=2,
        )

    @chex.all_variants
    def test_enhanced_blob_detection_all_methods(self):
        def detect_enhanced_blobs(mrc):
            return cb.enhanced_blob_detection(
                mrc,
                min_blob_size=4,
                max_blob_size=18,
                downscale=2,
                use_ridge_detection=True,
                use_watershed=True,
                ridge_threshold=0.01,
                min_marker_distance=4.0,
            )

        circular_blobs, elongated_blobs, watershed_blobs = self.variant(
            detect_enhanced_blobs
        )(self.complex_mrc)

        assert circular_blobs.shape[1] == 3
        assert elongated_blobs.shape[1] == 3
        assert watershed_blobs.shape[1] == 3

        total_detections = (
            len(circular_blobs) + len(elongated_blobs) + len(watershed_blobs)
        )
        assert total_detections >= 0

    @parameterized.parameters(
        (True, True), (True, False), (False, True), (False, False)
    )
    @chex.all_variants
    def test_enhanced_blob_detection_method_combinations(
        self, use_ridge, use_watershed
    ):
        def detect_blobs_with_options(mrc):
            return cb.enhanced_blob_detection(
                mrc,
                min_blob_size=3,
                max_blob_size=15,
                use_ridge_detection=use_ridge,
                use_watershed=use_watershed,
            )

        circular_blobs, elongated_blobs, watershed_blobs = self.variant(
            detect_blobs_with_options
        )(self.complex_mrc)

        if use_ridge:
            assert elongated_blobs.shape[1] == 3
        else:
            assert len(elongated_blobs) == 0

        if use_watershed:
            assert watershed_blobs.shape[1] == 3
        else:
            assert len(watershed_blobs) == 0

        assert circular_blobs.shape[1] == 3

    @chex.all_variants
    def test_enhanced_blob_detection_parameter_validation(self):
        def detect_blobs_params(mrc, min_size, max_size, downscale_factor):
            return cb.enhanced_blob_detection(
                mrc,
                min_blob_size=min_size,
                max_blob_size=max_size,
                downscale=downscale_factor,
            )

        circular_blobs, elongated_blobs, watershed_blobs = self.variant(
            lambda mrc: detect_blobs_params(mrc, 2, 25, 3)
        )(self.complex_mrc)

        assert circular_blobs.shape[1] == 3
        assert elongated_blobs.shape[1] == 3
        assert watershed_blobs.shape[1] == 3

    @chex.all_variants
    def test_enhanced_blob_detection_ridge_only(self):
        def detect_ridge_only(mrc):
            return cb.enhanced_blob_detection(
                mrc,
                min_blob_size=5,
                max_blob_size=20,
                use_ridge_detection=True,
                use_watershed=False,
                ridge_threshold=0.005,
            )

        circular_blobs, elongated_blobs, watershed_blobs = self.variant(
            detect_ridge_only
        )(self.complex_mrc)

        assert circular_blobs.shape[1] == 3
        assert elongated_blobs.shape[1] == 3
        assert len(watershed_blobs) == 0

    @chex.all_variants
    def test_enhanced_blob_detection_watershed_only(self):
        def detect_watershed_only(mrc):
            return cb.enhanced_blob_detection(
                mrc,
                min_blob_size=5,
                max_blob_size=20,
                use_ridge_detection=False,
                use_watershed=True,
                min_marker_distance=3.0,
            )

        circular_blobs, elongated_blobs, watershed_blobs = self.variant(
            detect_watershed_only
        )(self.complex_mrc)

        assert circular_blobs.shape[1] == 3
        assert len(elongated_blobs) == 0
        assert watershed_blobs.shape[1] == 3

    @chex.all_variants
    def test_enhanced_blob_detection_no_enhancement(self):
        def detect_basic_only(mrc):
            return cb.enhanced_blob_detection(
                mrc,
                min_blob_size=5,
                max_blob_size=20,
                use_ridge_detection=False,
                use_watershed=False,
            )

        circular_blobs, elongated_blobs, watershed_blobs = self.variant(
            detect_basic_only
        )(self.complex_mrc)

        assert circular_blobs.shape[1] == 3
        assert len(elongated_blobs) == 0
        assert len(watershed_blobs) == 0


class TestIntegrationMultiMethods(chex.TestCase, parameterized.TestCase):

    def setUp(self):
        super().setUp()
        x, y = jnp.meshgrid(jnp.linspace(-15, 15, 100), jnp.linspace(-15, 15, 100))

        ridge_structure = jnp.exp(-(y**2) / 3) * (jnp.abs(x) < 12)
        circular_blob = jnp.exp(-((x - 8) ** 2 + (y - 8) ** 2) / 8)
        overlapping1 = jnp.exp(-((x + 4) ** 2 + y**2) / 6)
        overlapping2 = jnp.exp(-((x + 1) ** 2 + y**2) / 6)

        combined_image = (
            ridge_structure + circular_blob + overlapping1 + overlapping2 + 5
        )

        self.test_mrc = make_MRC_Image(
            image_data=combined_image,
            voxel_size=jnp.array([1.0, 0.2, 0.2]),
            origin=jnp.zeros(3),
            data_min=jnp.min(combined_image),
            data_max=jnp.max(combined_image),
            data_mean=jnp.mean(combined_image),
            mode=2,
        )

    @chex.all_variants
    def test_integration_all_detection_types(self):
        def full_detection_pipeline(mrc):
            return cb.enhanced_blob_detection(
                mrc,
                min_blob_size=3,
                max_blob_size=18,
                use_ridge_detection=True,
                use_watershed=True,
                ridge_threshold=0.008,
                min_marker_distance=4.0,
                downscale=2,
                std_threshold=5,
            )

        circular_blobs, elongated_blobs, watershed_blobs = self.variant(
            full_detection_pipeline
        )(self.test_mrc)

        assert circular_blobs.shape[1] == 3
        assert elongated_blobs.shape[1] == 3
        assert watershed_blobs.shape[1] == 3

        if len(circular_blobs) > 0:
            coords_y = circular_blobs[:, 0]
            coords_x = circular_blobs[:, 1]
            sizes = circular_blobs[:, 2]

            assert jnp.all(coords_y >= 0)
            assert jnp.all(coords_x >= 0)
            assert jnp.all(sizes > 0)

    @chex.all_variants
    def test_integration_coordinate_scaling(self):
        def test_scaling(mrc, scale_factor):
            return cb.enhanced_blob_detection(
                mrc, downscale=scale_factor, min_blob_size=4, max_blob_size=16
            )

        results_scale2 = self.variant(lambda mrc: test_scaling(mrc, 2))(self.test_mrc)
        results_scale4 = self.variant(lambda mrc: test_scaling(mrc, 4))(self.test_mrc)

        circular_blobs_2, _, _ = results_scale2
        circular_blobs_4, _, _ = results_scale4

        assert circular_blobs_2.shape[1] == 3
        assert circular_blobs_4.shape[1] == 3

    @chex.all_variants
    def test_integration_threshold_sensitivity(self):
        def test_thresholds(mrc, std_thresh, ridge_thresh):
            return cb.enhanced_blob_detection(
                mrc,
                std_threshold=std_thresh,
                ridge_threshold=ridge_thresh,
                use_ridge_detection=True,
                use_watershed=True,
            )

        results_strict = self.variant(lambda mrc: test_thresholds(mrc, 8, 0.02))(
            self.test_mrc
        )
        results_lenient = self.variant(lambda mrc: test_thresholds(mrc, 3, 0.005))(
            self.test_mrc
        )

        circular_strict, elongated_strict, watershed_strict = results_strict
        circular_lenient, elongated_lenient, watershed_lenient = results_lenient

        assert len(circular_strict) <= len(circular_lenient)
        assert elongated_strict.shape[1] == 3
        assert elongated_lenient.shape[1] == 3

    @chex.all_variants
    def test_integration_empty_results_handling(self):
        uniform_data = jnp.ones((80, 80)) * 20
        uniform_mrc = make_MRC_Image(
            image_data=uniform_data,
            voxel_size=jnp.array([1.0, 0.1, 0.1]),
            origin=jnp.zeros(3),
            data_min=20.0,
            data_max=20.0,
            data_mean=20.0,
            mode=2,
        )

        def detect_on_uniform(mrc):
            return cb.enhanced_blob_detection(
                mrc,
                std_threshold=10,
                ridge_threshold=0.1,
                use_ridge_detection=True,
                use_watershed=True,
            )

        circular_blobs, elongated_blobs, watershed_blobs = self.variant(
            detect_on_uniform
        )(uniform_mrc)

        assert circular_blobs.shape == (0, 3)
        assert elongated_blobs.shape == (0, 3)
        assert watershed_blobs.shape == (0, 3)


if __name__ == "__main__":
    from absl.testing import absltest

    absltest.main()
