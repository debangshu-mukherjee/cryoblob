"""
Module: valid
-------------
JAX PyTree factory functions for configuration management
in the cryoblob preprocessing pipeline. This module provides
type-safe validation using JAX's functional approach with
jax.lax.cond for preprocessing parameters, file paths,
and blob detection configurations.

Functions
---------
- `make_mrc_image`:
    Factory function to create an MRC_Image instance.
- `make_preprocessing_config`:
    Factory function for preprocessing configuration PyTree
- `make_blob_detection_config`:
    Factory function for blob detection configuration PyTree
- `make_file_processing_config`:
    Factory function for file processing configuration PyTree
- `make_mrc_metadata`:
    Factory function for MRC metadata PyTree
- `make_ridge_detection_config`:
    Factory function for ridge detection configuration PyTree
- `make_watershed_config`:
    Factory function for watershed configuration PyTree
- `make_enhanced_blob_detection_config`:
    Factory function for enhanced blob detection configuration PyTree
- `make_hessian_blob_config`:
    Factory function for Hessian blob detection configuration PyTree
- `make_adaptive_filter_config`:
    Factory function for adaptive filter configuration PyTree
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Union
from cryoblob.types import (
    AdaptiveFilterConfig,
    BlobDetectionConfig,
    EnhancedBlobDetectionConfig,
    FileProcessingConfig,
    HessianBlobConfig,
    MRC_Image,
    MRCMetadata,
    PreprocessingConfig,
    RidgeDetectionConfig,
    WatershedConfig,
    scalar_bool,
    scalar_float,
    scalar_int,
)
from jax import lax
from jaxtyping import Array, Bool, Float, Int, Num, jaxtyped


@jaxtyped(typechecker=beartype)
def make_mrc_image(
    image_data: Union[Num[Array, "H W"], Num[Array, "D H W"]],
    voxel_size: Float[Array, "3"],
    origin: Float[Array, "3"],
    data_min: scalar_float,
    data_max: scalar_float,
    data_mean: scalar_float,
    mode: scalar_int,
) -> MRC_Image:
    """
    Description
    -----------
    Factory function to create an MRC_Image instance with validation.

    Parameters
    ----------
    - `image_data` (Num[Array, "H W"] | Num[Array, "D H W"]):
        The image data array from the MRC file. Can be 2D or 3D.
    - `voxel_size` (Float[Array, "3"]):
        Voxel size in the order (Z, Y, X).
    - `origin` (Float[Array, "3"]):
        Origin coordinates from the MRC file header (Z, Y, X).
    - `data_min` (scalar_float):
        Minimum value of image data (as stored in header).
    - `data_max` (scalar_float):
        Maximum value of image data (as stored in header).
    - `data_mean` (scalar_float):
        Mean value of image data (as stored in header).
    - `mode` (scalar_int):
        Data type mode from MRC header (e.g., 0: int8, 2: float32).

    Flow
    ----
    - Extract image shape and number of dimensions
    - Validate image_data has 2 or 3 dimensions with non-zero size, fallback to 1x1 array if invalid
    - Convert voxel_size to JAX array and validate it has exactly 3 elements
    - Ensure all voxel_size elements are positive, replace non-positive values with 1.0
    - Convert origin to JAX array and validate it has exactly 3 elements, default to zeros if invalid
    - Convert all scalar parameters to JAX arrays with appropriate dtypes
    - Validate data_max is not less than data_min, set equal if invalid
    - Clamp data_mean to be within [data_min, data_max] range
    - Validate mode is within valid MRC mode range [0-6], default to mode 2 (float32) if invalid
    - Return validated MRC_Image PyTree

    Returns
    -------
    - `MRC_Image`:
        A validated instance of the MRC_Image PyTree structure.
    """
    image_shape = jnp.shape(image_data)
    ndim = len(image_shape)
    
    image_data_validated = lax.cond(
        (ndim < 2) | (ndim > 3) | jnp.any(jnp.array(image_shape) <= 0),
        lambda x: jnp.ones((1, 1), dtype=x.dtype),
        lambda x: x,
        image_data
    )
    
    voxel_size_arr: Float[Array, "3"] = jnp.asarray(voxel_size, dtype=jnp.float32)
    voxel_size_validated: Float[Array, "3"] = lax.cond(
        (jnp.size(voxel_size_arr) != 3) | jnp.any(voxel_size_arr <= 0),
        lambda x: jnp.ones(3, dtype=jnp.float32),
        lambda x: x,
        voxel_size_arr
    )
    
    voxel_size_validated = jnp.where(
        voxel_size_validated <= 0,
        jnp.ones_like(voxel_size_validated),
        voxel_size_validated
    )
    
    origin_arr: Float[Array, "3"] = jnp.asarray(origin, dtype=jnp.float32)
    origin_validated: Float[Array, "3"] = lax.cond(
        jnp.size(origin_arr) != 3,
        lambda x: jnp.zeros(3, dtype=jnp.float32),
        lambda x: x,
        origin_arr
    )
    
    data_min_arr: Float[Array, ""] = jnp.asarray(data_min, dtype=jnp.float32)
    data_max_arr: Float[Array, ""] = jnp.asarray(data_max, dtype=jnp.float32)
    data_mean_arr: Float[Array, ""] = jnp.asarray(data_mean, dtype=jnp.float32)
    mode_arr: Int[Array, ""] = jnp.asarray(mode, dtype=jnp.int32)
    
    data_max_validated: Float[Array, ""] = lax.cond(
        data_max_arr < data_min_arr,
        lambda x: data_min_arr,
        lambda x: x,
        data_max_arr
    )
    
    data_mean_validated: Float[Array, ""] = jnp.clip(
        data_mean_arr, 
        data_min_arr, 
        data_max_validated
    )
    
    mode_validated: Int[Array, ""] = lax.cond(
        (mode_arr < 0) | (mode_arr > 6),
        lambda x: jnp.asarray(2, dtype=jnp.int32),
        lambda x: x,
        mode_arr
    )
    
    return MRC_Image(
        image_data=image_data_validated,
        voxel_size=voxel_size_validated,
        origin=origin_validated,
        data_min=data_min_arr,
        data_max=data_max_validated,
        data_mean=data_mean_validated,
        mode=mode_validated,
    )


def make_preprocessing_config(
    exponential: Optional[scalar_bool] = True,
    logarizer: Optional[scalar_bool] = False,
    gblur: Optional[scalar_int] = 2,
    background: Optional[scalar_int] = 0,
    apply_filter: Optional[scalar_int] = 0,
) -> PreprocessingConfig:
    """
    Description
    -----------
    Factory function to create a PreprocessingConfig PyTree with validation.

    Parameters
    ----------
    - `exponential` (bool, optional):
        Apply exponential function to enhance contrast (default: True).
    - `logarizer` (bool, optional):
        Apply logarithmic transformation (default: False).
    - `gblur` (int, optional):
        Gaussian blur sigma, 0 means no blur (default: 2, range: 0-50).
    - `background` (int, optional):
        Background subtraction sigma, 0 means no subtraction (default: 0, range: 0-100).
    - `apply_filter` (int, optional):
        Wiener filter kernel size, 0 means no filter (default: 0, range: 0-20).

    Flow
    ----
    - Convert all input parameters to JAX arrays with appropriate dtypes
    - Validate gblur parameter is within range [0, 50], reset to default if not
    - Validate background parameter is within range [0, 100], reset to default if not
    - Validate apply_filter parameter is within range [0, 20], reset to default if not
    - Check for conflicting exponential and logarizer options, disable logarizer if both are True
    - Return validated PreprocessingConfig PyTree

    Returns
    -------
    - `PreprocessingConfig`:
        Validated preprocessing configuration PyTree.
    """
    exponential_arr: Bool[Array, ""] = jnp.asarray(exponential, dtype=jnp.bool_)
    logarizer_arr: Bool[Array, ""] = jnp.asarray(logarizer, dtype=jnp.bool_)
    gblur_arr: Int[Array, ""] = jnp.asarray(gblur, dtype=jnp.int32)
    background_arr: Int[Array, ""] = jnp.asarray(background, dtype=jnp.int32)
    apply_filter_arr: Int[Array, ""] = jnp.asarray(apply_filter, dtype=jnp.int32)

    gblur_validated: Int[Array, ""] = lax.cond(
        (gblur_arr < 0) | (gblur_arr > 50),
        lambda x: lax.stop_gradient(jnp.asarray(2, dtype=jnp.int32)),
        lambda x: x,
        gblur_arr,
    )

    background_validated: Int[Array, ""] = lax.cond(
        (background_arr < 0) | (background_arr > 100),
        lambda x: lax.stop_gradient(jnp.asarray(0, dtype=jnp.int32)),
        lambda x: x,
        background_arr,
    )

    apply_filter_validated: Int[Array, ""] = lax.cond(
        (apply_filter_arr < 0) | (apply_filter_arr > 20),
        lambda x: lax.stop_gradient(jnp.asarray(0, dtype=jnp.int32)),
        lambda x: x,
        apply_filter_arr,
    )

    logarizer_validated: Bool[Array, ""] = lax.cond(
        exponential_arr & logarizer_arr,
        lambda x: lax.stop_gradient(jnp.asarray(False, dtype=jnp.bool_)),
        lambda x: x,
        logarizer_arr,
    )

    return PreprocessingConfig(
        exponential=exponential_arr,
        logarizer=logarizer_validated,
        gblur=gblur_validated,
        background=background_validated,
        apply_filter=apply_filter_validated,
    )


def make_blob_detection_config(
    min_sigma: Optional[scalar_float] = 1.0,
    max_sigma: Optional[scalar_float] = 50.0,
    num_sigma: Optional[scalar_int] = 10,
    threshold: Optional[scalar_float] = 0.01,
    exclude_border: Optional[scalar_int] = 0,
) -> BlobDetectionConfig:
    """
    Description
    -----------
    Factory function to create a BlobDetectionConfig PyTree with validation.

    Parameters
    ----------
    - `min_sigma` (float, optional):
        Minimum sigma for Laplacian of Gaussian (default: 1.0).
    - `max_sigma` (float, optional):
        Maximum sigma for Laplacian of Gaussian (default: 50.0).
    - `num_sigma` (int, optional):
        Number of sigma values to test (default: 10).
    - `threshold` (float, optional):
        Detection threshold (default: 0.01).
    - `exclude_border` (int, optional):
        Pixels to exclude from border (default: 0).

    Flow
    ----
    - Convert all input parameters to JAX arrays with appropriate dtypes
    - Validate min_sigma is positive, reset to 1.0 if not
    - Validate max_sigma is positive, reset to 50.0 if not
    - Ensure max_sigma is not less than min_sigma
    - Validate num_sigma is positive, reset to 10 if not
    - Validate exclude_border is non-negative, reset to 0 if negative
    - Return validated BlobDetectionConfig PyTree

    Returns
    -------
    - `BlobDetectionConfig`:
        Validated blob detection configuration PyTree.
    """
    min_sigma_arr: Float[Array, ""] = jnp.asarray(min_sigma, dtype=jnp.float32)
    max_sigma_arr: Float[Array, ""] = jnp.asarray(max_sigma, dtype=jnp.float32)
    num_sigma_arr: Int[Array, ""] = jnp.asarray(num_sigma, dtype=jnp.int32)
    threshold_arr: Float[Array, ""] = jnp.asarray(threshold, dtype=jnp.float32)
    exclude_border_arr: Int[Array, ""] = jnp.asarray(exclude_border, dtype=jnp.int32)

    min_sigma_validated: Float[Array, ""] = lax.cond(
        min_sigma_arr <= 0,
        lambda x: jnp.asarray(1.0, dtype=jnp.float32),
        lambda x: x,
        min_sigma_arr,
    )

    max_sigma_validated: Float[Array, ""] = lax.cond(
        max_sigma_arr <= 0,
        lambda x: jnp.asarray(50.0, dtype=jnp.float32),
        lambda x: x,
        max_sigma_arr,
    )

    max_sigma_validated: Float[Array, ""] = lax.cond(
        max_sigma_validated < min_sigma_validated,
        lambda x: min_sigma_validated,
        lambda x: x,
        max_sigma_validated,
    )

    num_sigma_validated: Int[Array, ""] = lax.cond(
        num_sigma_arr <= 0,
        lambda x: jnp.asarray(10, dtype=jnp.int32),
        lambda x: x,
        num_sigma_arr,
    )

    exclude_border_validated: Int[Array, ""] = lax.cond(
        exclude_border_arr < 0,
        lambda x: jnp.asarray(0, dtype=jnp.int32),
        lambda x: x,
        exclude_border_arr,
    )

    return BlobDetectionConfig(
        min_sigma=min_sigma_validated,
        max_sigma=max_sigma_validated,
        num_sigma=num_sigma_validated,
        threshold=threshold_arr,
        exclude_border=exclude_border_validated,
    )


def make_file_processing_config(
    batch_size: Optional[scalar_int] = 4, memory_limit_gb: Optional[scalar_float] = 8.0
) -> FileProcessingConfig:
    """
    Description
    -----------
    Factory function to create a FileProcessingConfig PyTree with validation.

    Parameters
    ----------
    - `batch_size` (int, optional):
        Number of files to process in parallel (default: 4).
    - `memory_limit_gb` (float, optional):
        Memory limit in GB (default: 8.0).

    Flow
    ----
    - Convert all input parameters to JAX arrays with appropriate dtypes
    - Validate batch_size is positive, reset to 4 if not
    - Validate memory_limit_gb is positive, reset to 8.0 if not
    - Return validated FileProcessingConfig PyTree

    Returns
    -------
    - `FileProcessingConfig`:
        Validated file processing configuration PyTree.
    """
    batch_size_arr: Int[Array, ""] = jnp.asarray(batch_size, dtype=jnp.int32)
    memory_limit_gb_arr: Float[Array, ""] = jnp.asarray(
        memory_limit_gb, dtype=jnp.float32
    )

    batch_size_validated: Int[Array, ""] = lax.cond(
        batch_size_arr <= 0,
        lambda x: jnp.asarray(4, dtype=jnp.int32),
        lambda x: x,
        batch_size_arr,
    )

    memory_limit_gb_validated: Float[Array, ""] = lax.cond(
        memory_limit_gb_arr <= 0,
        lambda x: jnp.asarray(8.0, dtype=jnp.float32),
        lambda x: x,
        memory_limit_gb_arr,
    )

    return FileProcessingConfig(
        batch_size=batch_size_validated, memory_limit_gb=memory_limit_gb_validated
    )


def make_mrc_metadata(
    nx: scalar_int,
    ny: scalar_int,
    nz: scalar_int,
    mode: scalar_int,
    dmin: scalar_float,
    dmax: scalar_float,
    dmean: scalar_float,
) -> MRCMetadata:
    """
    Description
    -----------
    Factory function to create an MRCMetadata PyTree with validation.

    Parameters
    ----------
    - `nx` (int):
        Number of columns.
    - `ny` (int):
        Number of rows.
    - `nz` (int):
        Number of sections.
    - `mode` (int):
        Data type.
    - `dmin` (float):
        Minimum density value.
    - `dmax` (float):
        Maximum density value.
    - `dmean` (float):
        Mean density value.

    Flow
    ----
    - Convert all input parameters to JAX arrays with appropriate dtypes
    - Validate nx, ny, nz are positive, reset to 1 if not
    - Ensure dmax is not less than dmin
    - Clamp dmean to be within [dmin, dmax] range
    - Return validated MRCMetadata PyTree

    Returns
    -------
    - `MRCMetadata`:
        Validated MRC metadata PyTree.
    """
    nx_arr: Int[Array, ""] = jnp.asarray(nx, dtype=jnp.int32)
    ny_arr: Int[Array, ""] = jnp.asarray(ny, dtype=jnp.int32)
    nz_arr: Int[Array, ""] = jnp.asarray(nz, dtype=jnp.int32)
    mode_arr: Int[Array, ""] = jnp.asarray(mode, dtype=jnp.int32)
    dmin_arr: Float[Array, ""] = jnp.asarray(dmin, dtype=jnp.float32)
    dmax_arr: Float[Array, ""] = jnp.asarray(dmax, dtype=jnp.float32)
    dmean_arr: Float[Array, ""] = jnp.asarray(dmean, dtype=jnp.float32)

    nx_validated: Int[Array, ""] = lax.cond(
        nx_arr <= 0, lambda x: jnp.asarray(1, dtype=jnp.int32), lambda x: x, nx_arr
    )
    ny_validated: Int[Array, ""] = lax.cond(
        ny_arr <= 0, lambda x: jnp.asarray(1, dtype=jnp.int32), lambda x: x, ny_arr
    )
    nz_validated: Int[Array, ""] = lax.cond(
        nz_arr <= 0, lambda x: jnp.asarray(1, dtype=jnp.int32), lambda x: x, nz_arr
    )

    dmax_validated: Float[Array, ""] = lax.cond(
        dmax_arr < dmin_arr, lambda x: dmin_arr, lambda x: x, dmax_arr
    )

    dmean_validated: Float[Array, ""] = jnp.clip(dmean_arr, dmin_arr, dmax_validated)

    return MRCMetadata(
        nx=nx_validated,
        ny=ny_validated,
        nz=nz_validated,
        mode=mode_arr,
        dmin=dmin_arr,
        dmax=dmax_validated,
        dmean=dmean_validated,
    )


def make_adaptive_filter_config(
    kernel_size: Optional[scalar_int] = 5,
    noise_estimate: Optional[scalar_float] = 0.01,
    iterations: Optional[scalar_int] = 10,
) -> AdaptiveFilterConfig:
    """
    Description
    -----------
    Factory function to create an AdaptiveFilterConfig PyTree with validation.

    Parameters
    ----------
    - `kernel_size` (int, optional):
        Size of the filter kernel (default: 5).
    - `noise_estimate` (float, optional):
        Initial noise estimate (default: 0.01).
    - `iterations` (int, optional):
        Number of adaptation iterations (default: 10).

    Flow
    ----
    - Convert all input parameters to JAX arrays with appropriate dtypes
    - Validate kernel_size is positive, reset to 5 if not
    - Ensure kernel_size is odd by adding 1 if even
    - Validate noise_estimate is positive, reset to 0.01 if not
    - Validate iterations is positive, reset to 10 if not
    - Return validated AdaptiveFilterConfig PyTree

    Returns
    -------
    - `AdaptiveFilterConfig`:
        Validated adaptive filter configuration PyTree.
    """
    kernel_size_arr: Int[Array, ""] = jnp.asarray(kernel_size, dtype=jnp.int32)
    noise_estimate_arr: Float[Array, ""] = jnp.asarray(
        noise_estimate, dtype=jnp.float32
    )
    iterations_arr: Int[Array, ""] = jnp.asarray(iterations, dtype=jnp.int32)

    kernel_size_validated: Int[Array, ""] = lax.cond(
        kernel_size_arr <= 0,
        lambda x: jnp.asarray(5, dtype=jnp.int32),
        lambda x: x,
        kernel_size_arr,
    )
    kernel_size_validated: Int[Array, ""] = lax.cond(
        kernel_size_validated % 2 == 0,
        lambda x: x + 1,
        lambda x: x,
        kernel_size_validated,
    )

    noise_estimate_validated: Float[Array, ""] = lax.cond(
        noise_estimate_arr <= 0,
        lambda x: jnp.asarray(0.01, dtype=jnp.float32),
        lambda x: x,
        noise_estimate_arr,
    )

    iterations_validated: Int[Array, ""] = lax.cond(
        iterations_arr <= 0,
        lambda x: jnp.asarray(10, dtype=jnp.int32),
        lambda x: x,
        iterations_arr,
    )

    return AdaptiveFilterConfig(
        kernel_size=kernel_size_validated,
        noise_estimate=noise_estimate_validated,
        iterations=iterations_validated,
    )


def make_ridge_detection_config(
    min_scale: Optional[scalar_float] = 1.0,
    max_scale: Optional[scalar_float] = 10.0,
    scale_step: Optional[scalar_float] = 0.5,
    threshold: Optional[scalar_float] = 0.1,
) -> RidgeDetectionConfig:
    """
    Description
    -----------
    Factory function to create a RidgeDetectionConfig PyTree with validation.

    Parameters
    ----------
    - `min_scale` (float, optional):
        Minimum scale for ridge detection (default: 1.0).
    - `max_scale` (float, optional):
        Maximum scale for ridge detection (default: 10.0).
    - `scale_step` (float, optional):
        Step size for scale space (default: 0.5).
    - `threshold` (float, optional):
        Detection threshold (default: 0.1).

    Flow
    ----
    - Convert all input parameters to JAX arrays with appropriate dtypes
    - Validate min_scale is positive, reset to 1.0 if not
    - Validate max_scale is positive, reset to 10.0 if not
    - Ensure max_scale is not less than min_scale
    - Validate scale_step is positive, reset to 0.5 if not
    - Return validated RidgeDetectionConfig PyTree

    Returns
    -------
    - `RidgeDetectionConfig`:
        Validated ridge detection configuration PyTree.
    """
    min_scale_arr: Float[Array, ""] = jnp.asarray(min_scale, dtype=jnp.float32)
    max_scale_arr: Float[Array, ""] = jnp.asarray(max_scale, dtype=jnp.float32)
    scale_step_arr: Float[Array, ""] = jnp.asarray(scale_step, dtype=jnp.float32)
    threshold_arr: Float[Array, ""] = jnp.asarray(threshold, dtype=jnp.float32)

    min_scale_validated: Float[Array, ""] = lax.cond(
        min_scale_arr <= 0,
        lambda x: jnp.asarray(1.0, dtype=jnp.float32),
        lambda x: x,
        min_scale_arr,
    )

    max_scale_validated: Float[Array, ""] = lax.cond(
        max_scale_arr <= 0,
        lambda x: jnp.asarray(10.0, dtype=jnp.float32),
        lambda x: x,
        max_scale_arr,
    )

    max_scale_validated: Float[Array, ""] = lax.cond(
        max_scale_validated < min_scale_validated,
        lambda x: min_scale_validated,
        lambda x: x,
        max_scale_validated,
    )

    scale_step_validated: Float[Array, ""] = lax.cond(
        scale_step_arr <= 0,
        lambda x: jnp.asarray(0.5, dtype=jnp.float32),
        lambda x: x,
        scale_step_arr,
    )

    return RidgeDetectionConfig(
        min_scale=min_scale_validated,
        max_scale=max_scale_validated,
        scale_step=scale_step_validated,
        threshold=threshold_arr,
    )


def make_watershed_config(
    min_distance: Optional[scalar_int] = 10,
    threshold_abs: Optional[scalar_float] = None,
    compactness: Optional[scalar_float] = 0.0,
) -> WatershedConfig:
    """
    Description
    -----------
    Factory function to create a WatershedConfig PyTree with validation.

    Parameters
    ----------
    - `min_distance` (int, optional):
        Minimum distance between markers (default: 10).
    - `threshold_abs` (float, optional):
        Absolute threshold for markers (default: None).
    - `compactness` (float, optional):
        Compactness parameter for watershed (default: 0.0).

    Flow
    ----
    - Convert all input parameters to JAX arrays with appropriate dtypes
    - Convert None threshold_abs to -1.0 for JAX compatibility
    - Validate min_distance is positive, reset to 10 if not
    - Validate compactness is non-negative, reset to 0.0 if negative
    - Return validated WatershedConfig PyTree

    Returns
    -------
    - `WatershedConfig`:
        Validated watershed configuration PyTree.
    """
    min_distance_arr: Int[Array, ""] = jnp.asarray(min_distance, dtype=jnp.int32)
    threshold_abs_val: Float[Array, ""] = jnp.asarray(
        -1.0 if threshold_abs is None else threshold_abs, dtype=jnp.float32
    )
    compactness_arr: Float[Array, ""] = jnp.asarray(compactness, dtype=jnp.float32)

    min_distance_validated: Int[Array, ""] = lax.cond(
        min_distance_arr <= 0,
        lambda x: jnp.asarray(10, dtype=jnp.int32),
        lambda x: x,
        min_distance_arr,
    )

    compactness_validated: Float[Array, ""] = lax.cond(
        compactness_arr < 0,
        lambda x: jnp.asarray(0.0, dtype=jnp.float32),
        lambda x: x,
        compactness_arr,
    )

    return WatershedConfig(
        min_distance=min_distance_validated,
        threshold_abs=threshold_abs_val,
        compactness=compactness_validated,
    )


def make_hessian_blob_config(
    min_sigma: Optional[scalar_float] = 1.0,
    max_sigma: Optional[scalar_float] = 30.0,
    num_sigma: Optional[scalar_int] = 10,
    threshold: Optional[scalar_float] = 0.01,
) -> HessianBlobConfig:
    """
    Description
    -----------
    Factory function to create a HessianBlobConfig PyTree with validation.

    Parameters
    ----------
    - `min_sigma` (float, optional):
        Minimum sigma for scale space (default: 1.0).
    - `max_sigma` (float, optional):
        Maximum sigma for scale space (default: 30.0).
    - `num_sigma` (int, optional):
        Number of scales to test (default: 10).
    - `threshold` (float, optional):
        Detection threshold (default: 0.01).

    Flow
    ----
    - Convert all input parameters to JAX arrays with appropriate dtypes
    - Validate min_sigma is positive, reset to 1.0 if not
    - Validate max_sigma is positive, reset to 30.0 if not
    - Ensure max_sigma is not less than min_sigma
    - Validate num_sigma is positive, reset to 10 if not
    - Return validated HessianBlobConfig PyTree

    Returns
    -------
    - `HessianBlobConfig`:
        Validated Hessian blob detection configuration PyTree.
    """
    min_sigma_arr: Float[Array, ""] = jnp.asarray(min_sigma, dtype=jnp.float32)
    max_sigma_arr: Float[Array, ""] = jnp.asarray(max_sigma, dtype=jnp.float32)
    num_sigma_arr: Int[Array, ""] = jnp.asarray(num_sigma, dtype=jnp.int32)
    threshold_arr: Float[Array, ""] = jnp.asarray(threshold, dtype=jnp.float32)

    min_sigma_validated: Float[Array, ""] = lax.cond(
        min_sigma_arr <= 0,
        lambda x: jnp.asarray(1.0, dtype=jnp.float32),
        lambda x: x,
        min_sigma_arr,
    )

    max_sigma_validated: Float[Array, ""] = lax.cond(
        max_sigma_arr <= 0,
        lambda x: jnp.asarray(30.0, dtype=jnp.float32),
        lambda x: x,
        max_sigma_arr,
    )

    max_sigma_validated: Float[Array, ""] = lax.cond(
        max_sigma_validated < min_sigma_validated,
        lambda x: min_sigma_validated,
        lambda x: x,
        max_sigma_validated,
    )

    num_sigma_validated: Int[Array, ""] = lax.cond(
        num_sigma_arr <= 0,
        lambda x: jnp.asarray(10, dtype=jnp.int32),
        lambda x: x,
        num_sigma_arr,
    )

    return HessianBlobConfig(
        min_sigma=min_sigma_validated,
        max_sigma=max_sigma_validated,
        num_sigma=num_sigma_validated,
        threshold=threshold_arr,
    )


def make_enhanced_blob_detection_config(
    min_blob_size: Optional[scalar_float] = 5.0,
    max_blob_size: Optional[scalar_float] = 50.0,
    detection_threshold: Optional[scalar_float] = 0.05,
    use_ridge_detection: Optional[scalar_bool] = True,
    use_watershed: Optional[scalar_bool] = True,
) -> EnhancedBlobDetectionConfig:
    """
    Description
    -----------
    Factory function to create an EnhancedBlobDetectionConfig PyTree with validation.

    Parameters
    ----------
    - `min_blob_size` (float, optional):
        Minimum expected blob size (default: 5.0).
    - `max_blob_size` (float, optional):
        Maximum expected blob size (default: 50.0).
    - `detection_threshold` (float, optional):
        Overall detection threshold (default: 0.05).
    - `use_ridge_detection` (bool, optional):
        Enable ridge detection for elongated objects (default: True).
    - `use_watershed` (bool, optional):
        Enable watershed for overlapping blobs (default: True).

    Flow
    ----
    - Convert all input parameters to JAX arrays with appropriate dtypes
    - Validate min_blob_size is positive, reset to 5.0 if not
    - Validate max_blob_size is positive, reset to 50.0 if not
    - Ensure max_blob_size is not less than min_blob_size
    - Validate detection_threshold is positive, reset to 0.05 if not
    - Return validated EnhancedBlobDetectionConfig PyTree

    Returns
    -------
    - `EnhancedBlobDetectionConfig`:
        Validated enhanced blob detection configuration PyTree.
    """
    min_blob_size_arr: Float[Array, ""] = jnp.asarray(min_blob_size, dtype=jnp.float32)
    max_blob_size_arr: Float[Array, ""] = jnp.asarray(max_blob_size, dtype=jnp.float32)
    detection_threshold_arr: Float[Array, ""] = jnp.asarray(
        detection_threshold, dtype=jnp.float32
    )
    use_ridge_detection_arr: Bool[Array, ""] = jnp.asarray(
        use_ridge_detection, dtype=jnp.bool_
    )
    use_watershed_arr: Bool[Array, ""] = jnp.asarray(use_watershed, dtype=jnp.bool_)

    min_blob_size_validated: Float[Array, ""] = lax.cond(
        min_blob_size_arr <= 0,
        lambda x: jnp.asarray(5.0, dtype=jnp.float32),
        lambda x: x,
        min_blob_size_arr,
    )

    max_blob_size_validated: Float[Array, ""] = lax.cond(
        max_blob_size_arr <= 0,
        lambda x: jnp.asarray(50.0, dtype=jnp.float32),
        lambda x: x,
        max_blob_size_arr,
    )

    max_blob_size_validated: Float[Array, ""] = lax.cond(
        max_blob_size_validated < min_blob_size_validated,
        lambda x: min_blob_size_validated,
        lambda x: x,
        max_blob_size_validated,
    )

    detection_threshold_validated: Float[Array, ""] = lax.cond(
        detection_threshold_arr <= 0,
        lambda x: jnp.asarray(0.05, dtype=jnp.float32),
        lambda x: x,
        detection_threshold_arr,
    )

    return EnhancedBlobDetectionConfig(
        min_blob_size=min_blob_size_validated,
        max_blob_size=max_blob_size_validated,
        detection_threshold=detection_threshold_validated,
        use_ridge_detection=use_ridge_detection_arr,
        use_watershed=use_watershed_arr,
    )
