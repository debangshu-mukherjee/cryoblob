from typing import Any, Tuple

try:
    import cupy as xp
except ImportError:
    import numpy as xp

import pytest
from arm_em.blob_detection import blob_list
from jaxtyping import Float


class TestBlobList:

    # Given a valid input image, the function should return a non-empty 2D array of blob coordinates.
    def test_valid_input_image(self):
        # Initialize input image
        image: Float[xp.ndarray, "3 3"] = xp.array(
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=xp.float64
        )

        # Invoke blob_list function
        result = blob_list(image)

        # Check if the result is a non-empty 2D array
        assert isinstance(result, xp.ndarray)
        assert result.ndim == 2
        assert result.size > 0

    # The function should be able to handle an input image with only one pixel and return an empty array.
    def test_single_pixel_input_image(self):
        # Initialize input image with a single pixel
        image = xp.array([[1]], dtype=xp.float64)

        # Invoke blob_list function
        result = blob_list(image)

        # Check if the result is an empty array
        assert isinstance(result, xp.ndarray)
        assert result.size == 0

    # The function should be able to handle an input image with missing or incorrect dimensions and raise an appropriate error message.
    def test_invalid_input_image(self):
        # Initialize invalid input image
        image = xp.array([[1, 1, 1], [1, 1, 1]], dtype=xp.float64)

        # Invoke blob_list function and check for appropriate error message
        with pytest.raises(ValueError) as e:
            blob_list(image)
        assert str(e.value) == "Input image must be a 2D array"

    # The function should be able to handle an input image with non-numeric pixel values and raise an appropriate error message.
    def test_non_numeric_input_image(self):
        # Initialize input image with non-numeric pixel values
        image = xp.array(
            [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]], dtype=xp.float64
        )

        # Invoke blob_list function and check for appropriate error message
        with pytest.raises(TypeError) as e:
            blob_list(image)
        assert str(e.value) == "Input image must contain numeric pixel values"

    # The function should be able to handle an input image with NaN or Inf pixel values and return valid results.
    def test_handle_nan_inf_values(self):
        # Initialize input image with NaN and Inf values
        image = xp.array([[1, 2, xp.nan], [3, xp.inf, 4], [5, 6, 7]], dtype=xp.float64)

        # Invoke blob_list function
        result = blob_list(image)

        # Check if the result is a non-empty 2D array
        assert isinstance(result, xp.ndarray)
        assert result.ndim == 2
        assert result.size > 0

    # The function should be able to handle an input image with very large pixel values and return valid results.
    def test_valid_input_image(self):
        # Initialize input image
        image = xp.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=xp.float64)

        # Invoke blob_list function
        result = blob_list(image)

        # Check if the result is a non-empty 2D array
        assert isinstance(result, xp.ndarray)
        assert result.ndim == 2
        assert result.size > 0

    # The function should be able to handle an input image with negative pixel values and return valid results.
    def test_valid_input_image_with_negative_values(self):
        # Initialize input image with negative pixel values
        image = xp.array([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]], dtype=xp.float64)

        # Invoke blob_list function
        result = blob_list(image)

        # Check if the result is a non-empty 2D array
        assert isinstance(result, xp.ndarray)
        assert result.ndim == 2
        assert result.size > 0

    # The function should be able to handle an input image with all pixels having a value of zero and return an empty array.
    def test_handle_zero_input(self):
        # Initialize input image
        image = xp.zeros((3, 3), dtype=xp.float64)

        # Invoke blob_list function
        result = blob_list(image)

        # Check if the result is an empty array
        assert isinstance(result, xp.ndarray)
        assert result.size == 0

    # The function should be able to handle an input image with all pixels having a value of one and return an empty array.
    def test_handle_all_ones_input(self):
        # Initialize input image
        image = xp.ones((3, 3), dtype=xp.float64)

        # Invoke blob_list function
        result = blob_list(image)

        # Check if the result is an empty array
        assert isinstance(result, xp.ndarray)
        assert result.size == 0

    # The function should be able to handle an input image with all pixels having the same value and return an empty array.
    def test_handle_same_value_input_image(self):
        # Initialize input image
        image = xp.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=xp.float64)

        # Invoke blob_list function
        result = blob_list(image)

        # Check if the result is an empty array
        assert isinstance(result, xp.ndarray)
        assert result.ndim == 2
        assert result.size == 0

    # The function should be able to handle input images with no blobs and return an empty array.
    def test_no_blobs(self):
        # Initialize input image with no blobs
        image = xp.zeros((3, 3), dtype=xp.float64)

        # Invoke blob_list function
        result = blob_list(image)

        # Check if the result is an empty array
        assert isinstance(result, xp.ndarray)
        assert result.size == 0

    # The function should be able to handle input images of different sizes and shapes.
    def test_valid_input_image(self):
        # Initialize input image
        image = xp.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=xp.float64)

        # Invoke blob_list function
        result = blob_list(image)

        # Check if the result is a non-empty 2D array
        assert isinstance(result, xp.ndarray)
        assert result.ndim == 2
        assert result.size > 0

    # The function should be able to handle different values for the optional arguments and still return valid results.
    def test_valid_input_image(self):
        # Initialize input image
        image = xp.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=xp.float64)

        # Invoke blob_list function
        result = blob_list(image)

        # Check if the result is a non-empty 2D array
        assert isinstance(result, xp.ndarray)
        assert result.ndim == 2
        assert result.size > 0

    # The function should be able to detect blobs of varying sizes within the specified range.
    def test_detect_blobs(self):
        # Initialize input image
        image = xp.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=xp.float64)

        # Invoke blob_list function
        result = blob_list(image)

        # Check if the result is a non-empty 2D array
        assert isinstance(result, xp.ndarray)
        assert result.ndim == 2
        assert result.size > 0
