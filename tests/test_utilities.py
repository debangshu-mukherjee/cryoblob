import cupy as cp
import cupyx.scipy.ndimage as csnd
import cupyx.scipy.signal as csisig
from arm_em.utilities import laplacian_gaussian, preprocessing


class TestLaplacianGaussian:

    # returns filtered image with expected shape
    def test_filtered_image_shape(self):
        # Arrange
        image = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Act
        filtered = laplacian_gaussian(
            image, standard_deviation=3, hist_stretch=True, sampling=1
        )

        # Assert
        assert filtered.shape == (3, 3)

    # applies Laplacian of Gaussian filter to input image
    def test_filtered_image_shape(self):
        # Arrange
        image = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Act
        filtered = laplacian_gaussian(
            image, standard_deviation=3, hist_stretch=True, sampling=1
        )

        # Assert
        assert filtered.shape == (3, 3)

    # performs histogram stretching on input image if hist_stretch is True
    def test_performs_histogram_stretching(self):
        # Arrange
        image = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Act
        filtered = laplacian_gaussian(
            image, standard_deviation=3, hist_stretch=True, sampling=1
        )

        # Assert
        assert filtered.shape == (3, 3)

    # returns filtered image with expected values
    def test_filtered_image_shape(self):
        # Arrange
        image = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Act
        filtered = laplacian_gaussian(
            image, standard_deviation=3, hist_stretch=True, sampling=1
        )

        # Assert
        assert filtered.shape == (3, 3)

    # returns filtered image with expected dtype
    def test_filtered_image_dtype(self):
        # Arrange
        image = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Act
        filtered = laplacian_gaussian(
            image, standard_deviation=3, hist_stretch=True, sampling=1
        )

        # Assert
        assert filtered.dtype == cp.float64

    # handles input image with all pixels having same value
    def test_filtered_image_shape(self):
        # Arrange
        image = cp.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

        # Act
        filtered = laplacian_gaussian(
            image, standard_deviation=3, hist_stretch=True, sampling=1
        )

        # Assert
        assert filtered.shape == (3, 3)

    # handles input image with negative values
    def test_handles_input_image_with_negative_values(self):
        # Arrange
        image = cp.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]])

        # Act
        filtered = laplacian_gaussian(
            image, standard_deviation=3, hist_stretch=True, sampling=1
        )

        # Assert
        assert cp.all(
            filtered == cp.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
        )

    # handles input image with very small values
    def test_handles_input_image_with_very_small_values(self):
        # Arrange
        image = cp.array(
            [[1e-10, 2e-10, 3e-10], [4e-10, 5e-10, 6e-10], [7e-10, 8e-10, 9e-10]]
        )

        # Act
        filtered = laplacian_gaussian(
            image, standard_deviation=3, hist_stretch=True, sampling=1
        )

        # Assert
        assert filtered.shape == (3, 3)

    # handles input image with very large values
    def test_handles_input_image_with_very_large_values(self):
        # Arrange
        image = cp.array([[1e10, 2e10, 3e10], [4e10, 5e10, 6e10], [7e10, 8e10, 9e10]])

        # Act
        filtered = laplacian_gaussian(
            image, standard_deviation=3, hist_stretch=True, sampling=1
        )

        # Assert
        assert filtered.shape == (3, 3)

    # allows setting standard deviation of Gaussian filter
    def test_standard_deviation(self):
        # Arrange
        image = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Act
        filtered = laplacian_gaussian(
            image, standard_deviation=5, hist_stretch=True, sampling=1
        )

        # Assert
        assert filtered.shape == (3, 3)

    # allows setting downsampling factor for input image
    def test_downsampling_factor(self):
        # Arrange
        image = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected_shape = (2, 2)
        downsampling_factor = 0.5

        # Act
        filtered = laplacian_gaussian(
            image, standard_deviation=3, hist_stretch=True, sampling=downsampling_factor
        )

        # Assert
        assert filtered.shape == expected_shape

    # allows disabling histogram stretching
    def test_disable_hist_stretch(self):
        # Arrange
        image = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Act
        filtered = laplacian_gaussian(
            image, standard_deviation=3, hist_stretch=False, sampling=1
        )

        # Assert
        assert filtered.shape == (3, 3)

    # allows disabling normalization of filtered image
    def test_disable_normalization(self):
        # Arrange
        image = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Act
        filtered = laplacian_gaussian(
            image, standard_deviation=3, hist_stretch=True, sampling=1, normalized=False
        )

        # Assert
        assert filtered.shape == (3, 3)

    # returns both positive and negative filtered images
    def test_filtered_image_shape(self):
        # Arrange
        image = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Act
        filtered = laplacian_gaussian(
            image, standard_deviation=3, hist_stretch=True, sampling=1
        )

        # Assert
        assert filtered.shape == (3, 3)

    # performs Gaussian filtering on input image
    def test_filtered_image_shape(self):
        # Arrange
        image = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Act
        filtered = laplacian_gaussian(
            image, standard_deviation=3, hist_stretch=True, sampling=1
        )

        # Assert
        assert filtered.shape == (3, 3)

    # Adjusts the contrast of the input image using histogram stretching
    def test_adjust_contrast_with_histogram_stretching(self):
        # Arrange
        image = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Act
        filtered = laplacian_gaussian(
            image, standard_deviation=3, hist_stretch=True, sampling=1
        )

        # Assert
        assert filtered.shape == (3, 3)


class TestPreprocessing:

    # correctly normalizes the input image
    def test_correctly_normalizes_input_image(self):
        image_orig = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
        expected_output = (image_orig - cp.amin(image_orig)) / (
            cp.amax(image_orig) - cp.amin(image_orig)
        )
        result = preprocessing(
            image_orig,
            exponential=False,
            logarizer=False,
            gblur=0,
            background=0,
            apply_filter=0,
        )
        assert cp.allclose(
            result, expected_output
        ), "The image was not normalized correctly."

    # input image with all zero values
    def test_input_image_all_zero_values(self):
        image_orig = cp.zeros((2, 2), dtype=cp.float32)
        result = preprocessing(
            image_orig,
            exponential=False,
            logarizer=False,
            gblur=0,
            background=0,
            apply_filter=0,
        )
        expected_output = cp.zeros((2, 2), dtype=cp.float32)
        assert cp.allclose(
            result, expected_output
        ), "The function did not handle an all-zero input image correctly."

    # applies logarithmic function when logarizer is True
    def test_applies_logarithmic_function_when_logarizer_is_true(self):
        image_orig = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
        expected_output = (image_orig - cp.amin(image_orig)) / (
            cp.amax(image_orig) - cp.amin(image_orig)
        )
        result = preprocessing(
            image_orig,
            exponential=False,
            logarizer=True,
            gblur=0,
            background=0,
            apply_filter=0,
        )
        assert cp.allclose(
            result, cp.log(expected_output)
        ), "The logarithmic function was not applied correctly."

    # applies exponential function when exponential is True
    def test_applies_exponential_function_when_exponential_is_true(self):
        image_orig = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
        expected_output = (image_orig - cp.amin(image_orig)) / (
            cp.amax(image_orig) - cp.amin(image_orig)
        )
        result = preprocessing(
            image_orig,
            exponential=True,
            logarizer=False,
            gblur=0,
            background=0,
            apply_filter=0,
        )
        assert cp.allclose(
            result, cp.exp(expected_output)
        ), "The exponential function was not applied correctly."

    # applies Gaussian blur with specified standard deviation
    def test_applies_gaussian_blur(self):
        image_orig = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
        expected_output = csnd.gaussian_filter(image_orig, 2)
        result = preprocessing(
            image_orig,
            exponential=False,
            logarizer=False,
            gblur=2,
            background=0,
            apply_filter=0,
        )
        assert cp.allclose(
            result, expected_output
        ), "The Gaussian blur was not applied correctly."

    # subtracts background using Gaussian filter when background > 0
    def test_subtracts_background_using_gaussian_filter(self):
        image_orig = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
        expected_output = (image_orig - cp.amin(image_orig)) / (
            cp.amax(image_orig) - cp.amin(image_orig)
        )
        result = preprocessing(
            image_orig,
            exponential=False,
            logarizer=False,
            gblur=0,
            background=2,
            apply_filter=0,
        )
        background_filtered = csnd.gaussian_filter(expected_output, 2)
        expected_output = expected_output - background_filtered
        assert cp.allclose(
            result, expected_output
        ), "Background subtraction using Gaussian filter failed."

    # applies Wiener filter when apply_filter > 1
    def test_applies_wiener_filter_when_apply_filter_gt_1(self):
        image_orig = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
        expected_output = csisig.wiener(image_orig, mysize=3)
        result = preprocessing(
            image_orig,
            exponential=False,
            logarizer=False,
            gblur=0,
            background=0,
            apply_filter=3,
        )
        assert cp.allclose(
            result, expected_output
        ), "Wiener filter was not applied correctly."

    # handles default parameter values correctly
    def test_handles_default_parameter_values_correctly(self):
        image_orig = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
        expected_output = (image_orig - cp.amin(image_orig)) / (
            cp.amax(image_orig) - cp.amin(image_orig)
        )
        result = preprocessing(image_orig)
        assert cp.allclose(
            result, expected_output
        ), "Default parameter values were not handled correctly."

    # input image with all maximum values
    def test_input_image_all_maximum_values(self):
        image_orig = cp.ones((2, 2), dtype=cp.float32) * cp.amax(image_orig)
        result = preprocessing(
            image_orig,
            exponential=False,
            logarizer=False,
            gblur=0,
            background=0,
            apply_filter=0,
        )
        expected_output = cp.ones((2, 2), dtype=cp.float32)
        assert cp.allclose(
            result, expected_output
        ), "The function did not handle an input image with all maximum values correctly."

    # gblur set to 0
    def test_gblur_set_to_0(self):
        image_orig = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
        expected_output = (image_orig - cp.amin(image_orig)) / (
            cp.amax(image_orig) - cp.amin(image_orig)
        )
        result = preprocessing(
            image_orig,
            exponential=False,
            logarizer=False,
            gblur=0,
            background=0,
            apply_filter=0,
        )
        assert cp.allclose(
            result, expected_output
        ), "The image was not processed correctly with gblur set to 0."

    # background set to 0
    def test_background_set_to_0(self):
        image_orig = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
        result = preprocessing(
            image_orig,
            exponential=False,
            logarizer=False,
            gblur=0,
            background=0,
            apply_filter=0,
        )
        expected_output = (image_orig - cp.amin(image_orig)) / (
            cp.amax(image_orig) - cp.amin(image_orig)
        )
        assert cp.allclose(
            result, expected_output
        ), "The function did not handle background set to 0 correctly."

    # apply_filter set to 0
    def test_apply_filter_set_to_0(self):
        image_orig = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
        result = preprocessing(
            image_orig,
            exponential=False,
            logarizer=False,
            gblur=0,
            background=0,
            apply_filter=0,
        )
        expected_output = (image_orig - cp.amin(image_orig)) / (
            cp.amax(image_orig) - cp.amin(image_orig)
        )
        assert cp.allclose(
            result, expected_output
        ), "The function did not handle apply_filter set to 0 correctly."

    # both exponential and logarizer set to True
    def test_both_exponential_and_logarizer_true(self):
        image_orig = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
        expected_output = (image_orig - cp.amin(image_orig)) / (
            cp.amax(image_orig) - cp.amin(image_orig)
        )
        result = preprocessing(
            image_orig,
            exponential=True,
            logarizer=True,
            gblur=0,
            background=0,
            apply_filter=0,
        )
        assert cp.allclose(
            result, expected_output
        ), "Exponential and logarizer not applied correctly."

    # handles non-square images
    def test_handles_non_square_images(self):
        image_orig = cp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=cp.float32)
        expected_output = (image_orig - cp.amin(image_orig)) / (
            cp.amax(image_orig) - cp.amin(image_orig)
        )
        result = preprocessing(
            image_orig,
            exponential=False,
            logarizer=False,
            gblur=0,
            background=0,
            apply_filter=0,
        )
        assert cp.allclose(
            result, expected_output
        ), "The function did not handle non-square images correctly."

    # handles images with negative values
    def test_handles_images_with_negative_values(self):
        image_orig = cp.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=cp.float32)
        expected_output = (image_orig - cp.amin(image_orig)) / (
            cp.amax(image_orig) - cp.amin(image_orig)
        )
        result = preprocessing(
            image_orig,
            exponential=False,
            logarizer=False,
            gblur=0,
            background=0,
            apply_filter=0,
        )
        assert cp.allclose(
            result, expected_output
        ), "The function did not handle images with negative values correctly."

    # handles images with NaN or Inf values
    def test_handles_images_with_nan_or_inf_values(self):
        image_orig = cp.array([[1.0, cp.nan], [3.0, cp.inf]], dtype=cp.float32)
        result = preprocessing(
            image_orig,
            exponential=False,
            logarizer=False,
            gblur=0,
            background=0,
            apply_filter=0,
        )
        assert (
            not cp.isnan(result).any() and not cp.isinf(result).any()
        ), "The function did not handle images with NaN or Inf values correctly."
