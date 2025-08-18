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

from beartype.typing import Optional, Union
from beartype import beartype
from jaxtyping import Array, Float, Num, jaxtyped
import jax.numpy as jnp
from jax import lax

from cryoblob.types import (
    scalar_float,
    scalar_int,
    PreprocessingConfig,
    BlobDetectionConfig,
    FileProcessingConfig,
    MRCMetadata,
    MRC_Image,
    AdaptiveFilterConfig,
    RidgeDetectionConfig,
    WatershedConfig,
    HessianBlobConfig,
    EnhancedBlobDetectionConfig,
)

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
    Factory function to create an MRC_Image instance.

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

    Returns
    -------
    - `MRC_Image`:
        An instance of the MRC_Image PyTree structure.
    """
    return MRC_Image(
        image_data=image_data,
        voxel_size=voxel_size,
        origin=origin,
        data_min=data_min,
        data_max=data_max,
        data_mean=data_mean,
        mode=mode,
    )


def make_preprocessing_config(
    exponential: bool = True,
    logarizer: bool = False,
    gblur: int = 2,
    background: int = 0,
    apply_filter: int = 0
) -> PreprocessingConfig:
    """
    Factory function to create a PreprocessingConfig PyTree with validation.
    
    Parameters
    ----------
    exponential : bool
        Apply exponential function to enhance contrast (default: True)
    logarizer : bool
        Apply logarithmic transformation (default: False)
    gblur : int
        Gaussian blur sigma, 0 means no blur (default: 2, range: 0-50)
    background : int
        Background subtraction sigma, 0 means no subtraction (default: 0, range: 0-100)
    apply_filter : int
        Wiener filter kernel size, 0 means no filter (default: 0, range: 0-20)
    
    Returns
    -------
    PreprocessingConfig
        Validated preprocessing configuration PyTree
    
    Raises
    ------
    ValueError
        If validation fails
    """
    # Convert to JAX arrays
    exponential_arr = jnp.asarray(exponential, dtype=jnp.bool_)
    logarizer_arr = jnp.asarray(logarizer, dtype=jnp.bool_)
    gblur_arr = jnp.asarray(gblur, dtype=jnp.int32)
    background_arr = jnp.asarray(background, dtype=jnp.int32)
    apply_filter_arr = jnp.asarray(apply_filter, dtype=jnp.int32)
    
    # Validation using lax.cond
    # Check ranges
    gblur_validated = lax.cond(
        (gblur_arr < 0) | (gblur_arr > 50),
        lambda x: lax.stop_gradient(jnp.asarray(2, dtype=jnp.int32)),  # Default value
        lambda x: x,
        gblur_arr
    )
    
    background_validated = lax.cond(
        (background_arr < 0) | (background_arr > 100),
        lambda x: lax.stop_gradient(jnp.asarray(0, dtype=jnp.int32)),  # Default value
        lambda x: x,
        background_arr
    )
    
    apply_filter_validated = lax.cond(
        (apply_filter_arr < 0) | (apply_filter_arr > 20),
        lambda x: lax.stop_gradient(jnp.asarray(0, dtype=jnp.int32)),  # Default value
        lambda x: x,
        apply_filter_arr
    )
    
    # Check conflicting options
    logarizer_validated = lax.cond(
        exponential_arr & logarizer_arr,
        lambda x: lax.stop_gradient(jnp.asarray(False, dtype=jnp.bool_)),  # Disable logarizer if both are True
        lambda x: x,
        logarizer_arr
    )
    
    return PreprocessingConfig(
        exponential=exponential_arr,
        logarizer=logarizer_validated,
        gblur=gblur_validated,
        background=background_validated,
        apply_filter=apply_filter_validated
    )


def make_blob_detection_config(
    min_sigma: float = 1.0,
    max_sigma: float = 50.0,
    num_sigma: int = 10,
    threshold: float = 0.01,
    exclude_border: int = 0
) -> BlobDetectionConfig:
    """
    Factory function to create a BlobDetectionConfig PyTree with validation.
    
    Parameters
    ----------
    min_sigma : float
        Minimum sigma for Laplacian of Gaussian (default: 1.0)
    max_sigma : float
        Maximum sigma for Laplacian of Gaussian (default: 50.0)
    num_sigma : int
        Number of sigma values to test (default: 10)
    threshold : float
        Detection threshold (default: 0.01)
    exclude_border : int
        Pixels to exclude from border (default: 0)
    
    Returns
    -------
    BlobDetectionConfig
        Validated blob detection configuration PyTree
    """
    # Convert to JAX arrays
    min_sigma_arr = jnp.asarray(min_sigma, dtype=jnp.float32)
    max_sigma_arr = jnp.asarray(max_sigma, dtype=jnp.float32)
    num_sigma_arr = jnp.asarray(num_sigma, dtype=jnp.int32)
    threshold_arr = jnp.asarray(threshold, dtype=jnp.float32)
    exclude_border_arr = jnp.asarray(exclude_border, dtype=jnp.int32)
    
    # Validation
    min_sigma_validated = lax.cond(
        min_sigma_arr <= 0,
        lambda x: jnp.asarray(1.0, dtype=jnp.float32),
        lambda x: x,
        min_sigma_arr
    )
    
    max_sigma_validated = lax.cond(
        max_sigma_arr <= 0,
        lambda x: jnp.asarray(50.0, dtype=jnp.float32),
        lambda x: x,
        max_sigma_arr
    )
    
    max_sigma_validated = lax.cond(
        max_sigma_validated < min_sigma_validated,
        lambda x: min_sigma_validated,
        lambda x: x,
        max_sigma_validated
    )
    
    num_sigma_validated = lax.cond(
        num_sigma_arr <= 0,
        lambda x: jnp.asarray(10, dtype=jnp.int32),
        lambda x: x,
        num_sigma_arr
    )
    
    exclude_border_validated = lax.cond(
        exclude_border_arr < 0,
        lambda x: jnp.asarray(0, dtype=jnp.int32),
        lambda x: x,
        exclude_border_arr
    )
    
    return BlobDetectionConfig(
        min_sigma=min_sigma_validated,
        max_sigma=max_sigma_validated,
        num_sigma=num_sigma_validated,
        threshold=threshold_arr,
        exclude_border=exclude_border_validated
    )


def make_file_processing_config(
    batch_size: int = 4,
    memory_limit_gb: float = 8.0
) -> FileProcessingConfig:
    """
    Factory function to create a FileProcessingConfig PyTree with validation.
    
    Parameters
    ----------
    batch_size : int
        Number of files to process in parallel (default: 4)
    memory_limit_gb : float
        Memory limit in GB (default: 8.0)
    
    Returns
    -------
    FileProcessingConfig
        Validated file processing configuration PyTree
    """
    # Convert to JAX arrays
    batch_size_arr = jnp.asarray(batch_size, dtype=jnp.int32)
    memory_limit_gb_arr = jnp.asarray(memory_limit_gb, dtype=jnp.float32)
    
    # Validation
    batch_size_validated = lax.cond(
        batch_size_arr <= 0,
        lambda x: jnp.asarray(4, dtype=jnp.int32),
        lambda x: x,
        batch_size_arr
    )
    
    memory_limit_gb_validated = lax.cond(
        memory_limit_gb_arr <= 0,
        lambda x: jnp.asarray(8.0, dtype=jnp.float32),
        lambda x: x,
        memory_limit_gb_arr
    )
    
    return FileProcessingConfig(
        batch_size=batch_size_validated,
        memory_limit_gb=memory_limit_gb_validated
    )


def make_mrc_metadata(
    nx: int,
    ny: int,
    nz: int,
    mode: int,
    dmin: float,
    dmax: float,
    dmean: float
) -> MRCMetadata:
    """
    Factory function to create an MRCMetadata PyTree with validation.
    
    Parameters
    ----------
    nx : int
        Number of columns
    ny : int
        Number of rows
    nz : int
        Number of sections
    mode : int
        Data type
    dmin : float
        Minimum density value
    dmax : float
        Maximum density value
    dmean : float
        Mean density value
    
    Returns
    -------
    MRCMetadata
        Validated MRC metadata PyTree
    """
    # Convert to JAX arrays
    nx_arr = jnp.asarray(nx, dtype=jnp.int32)
    ny_arr = jnp.asarray(ny, dtype=jnp.int32)
    nz_arr = jnp.asarray(nz, dtype=jnp.int32)
    mode_arr = jnp.asarray(mode, dtype=jnp.int32)
    dmin_arr = jnp.asarray(dmin, dtype=jnp.float32)
    dmax_arr = jnp.asarray(dmax, dtype=jnp.float32)
    dmean_arr = jnp.asarray(dmean, dtype=jnp.float32)
    
    # Validation
    nx_validated = lax.cond(nx_arr <= 0, lambda x: jnp.asarray(1, dtype=jnp.int32), lambda x: x, nx_arr)
    ny_validated = lax.cond(ny_arr <= 0, lambda x: jnp.asarray(1, dtype=jnp.int32), lambda x: x, ny_arr)
    nz_validated = lax.cond(nz_arr <= 0, lambda x: jnp.asarray(1, dtype=jnp.int32), lambda x: x, nz_arr)
    
    # Ensure dmax >= dmin
    dmax_validated = lax.cond(dmax_arr < dmin_arr, lambda x: dmin_arr, lambda x: x, dmax_arr)
    
    # Clamp dmean to [dmin, dmax]
    dmean_validated = jnp.clip(dmean_arr, dmin_arr, dmax_validated)
    
    return MRCMetadata(
        nx=nx_validated, ny=ny_validated, nz=nz_validated, mode=mode_arr,
        dmin=dmin_arr, dmax=dmax_validated, dmean=dmean_validated
    )


def make_adaptive_filter_config(
    kernel_size: int = 5,
    noise_estimate: float = 0.01,
    iterations: int = 10
) -> AdaptiveFilterConfig:
    """
    Factory function to create an AdaptiveFilterConfig PyTree with validation.
    
    Parameters
    ----------
    kernel_size : int
        Size of the filter kernel (default: 5)
    noise_estimate : float
        Initial noise estimate (default: 0.01)
    iterations : int
        Number of adaptation iterations (default: 10)
    
    Returns
    -------
    AdaptiveFilterConfig
        Validated adaptive filter configuration PyTree
    """
    # Convert to JAX arrays
    kernel_size_arr = jnp.asarray(kernel_size, dtype=jnp.int32)
    noise_estimate_arr = jnp.asarray(noise_estimate, dtype=jnp.float32)
    iterations_arr = jnp.asarray(iterations, dtype=jnp.int32)
    
    # Ensure kernel_size is odd and positive
    kernel_size_validated = lax.cond(
        kernel_size_arr <= 0,
        lambda x: jnp.asarray(5, dtype=jnp.int32),
        lambda x: x,
        kernel_size_arr
    )
    kernel_size_validated = lax.cond(
        kernel_size_validated % 2 == 0,
        lambda x: x + 1,
        lambda x: x,
        kernel_size_validated
    )
    
    noise_estimate_validated = lax.cond(
        noise_estimate_arr <= 0,
        lambda x: jnp.asarray(0.01, dtype=jnp.float32),
        lambda x: x,
        noise_estimate_arr
    )
    
    iterations_validated = lax.cond(
        iterations_arr <= 0,
        lambda x: jnp.asarray(10, dtype=jnp.int32),
        lambda x: x,
        iterations_arr
    )
    
    return AdaptiveFilterConfig(
        kernel_size=kernel_size_validated,
        noise_estimate=noise_estimate_validated,
        iterations=iterations_validated
    )


def make_ridge_detection_config(
    min_scale: float = 1.0,
    max_scale: float = 10.0,
    scale_step: float = 0.5,
    threshold: float = 0.1
) -> RidgeDetectionConfig:
    """
    Factory function to create a RidgeDetectionConfig PyTree with validation.
    
    Parameters
    ----------
    min_scale : float
        Minimum scale for ridge detection (default: 1.0)
    max_scale : float
        Maximum scale for ridge detection (default: 10.0)
    scale_step : float
        Step size for scale space (default: 0.5)
    threshold : float
        Detection threshold (default: 0.1)
    
    Returns
    -------
    RidgeDetectionConfig
        Validated ridge detection configuration PyTree
    """
    # Convert to JAX arrays
    min_scale_arr = jnp.asarray(min_scale, dtype=jnp.float32)
    max_scale_arr = jnp.asarray(max_scale, dtype=jnp.float32)
    scale_step_arr = jnp.asarray(scale_step, dtype=jnp.float32)
    threshold_arr = jnp.asarray(threshold, dtype=jnp.float32)
    
    # Validation
    min_scale_validated = lax.cond(
        min_scale_arr <= 0,
        lambda x: jnp.asarray(1.0, dtype=jnp.float32),
        lambda x: x,
        min_scale_arr
    )
    
    max_scale_validated = lax.cond(
        max_scale_arr <= 0,
        lambda x: jnp.asarray(10.0, dtype=jnp.float32),
        lambda x: x,
        max_scale_arr
    )
    
    max_scale_validated = lax.cond(
        max_scale_validated < min_scale_validated,
        lambda x: min_scale_validated,
        lambda x: x,
        max_scale_validated
    )
    
    scale_step_validated = lax.cond(
        scale_step_arr <= 0,
        lambda x: jnp.asarray(0.5, dtype=jnp.float32),
        lambda x: x,
        scale_step_arr
    )
    
    return RidgeDetectionConfig(
        min_scale=min_scale_validated,
        max_scale=max_scale_validated,
        scale_step=scale_step_validated,
        threshold=threshold_arr
    )


def make_watershed_config(
    min_distance: int = 10,
    threshold_abs: Optional[float] = None,
    compactness: float = 0.0
) -> WatershedConfig:
    """
    Factory function to create a WatershedConfig PyTree with validation.
    
    Parameters
    ----------
    min_distance : int
        Minimum distance between markers (default: 10)
    threshold_abs : Optional[float]
        Absolute threshold for markers (default: None)
    compactness : float
        Compactness parameter for watershed (default: 0.0)
    
    Returns
    -------
    WatershedConfig
        Validated watershed configuration PyTree
    """
    # Convert to JAX arrays
    min_distance_arr = jnp.asarray(min_distance, dtype=jnp.int32)
    threshold_abs_val = jnp.asarray(
        -1.0 if threshold_abs is None else threshold_abs,
        dtype=jnp.float32
    )
    compactness_arr = jnp.asarray(compactness, dtype=jnp.float32)
    
    # Validation
    min_distance_validated = lax.cond(
        min_distance_arr <= 0,
        lambda x: jnp.asarray(10, dtype=jnp.int32),
        lambda x: x,
        min_distance_arr
    )
    
    compactness_validated = lax.cond(
        compactness_arr < 0,
        lambda x: jnp.asarray(0.0, dtype=jnp.float32),
        lambda x: x,
        compactness_arr
    )
    
    return WatershedConfig(
        min_distance=min_distance_validated,
        threshold_abs=threshold_abs_val,
        compactness=compactness_validated
    )


def make_hessian_blob_config(
    min_sigma: float = 1.0,
    max_sigma: float = 30.0,
    num_sigma: int = 10,
    threshold: float = 0.01
) -> HessianBlobConfig:
    """
    Factory function to create a HessianBlobConfig PyTree with validation.
    
    Parameters
    ----------
    min_sigma : float
        Minimum sigma for scale space (default: 1.0)
    max_sigma : float
        Maximum sigma for scale space (default: 30.0)
    num_sigma : int
        Number of scales to test (default: 10)
    threshold : float
        Detection threshold (default: 0.01)
    
    Returns
    -------
    HessianBlobConfig
        Validated Hessian blob detection configuration PyTree
    """
    # Convert to JAX arrays
    min_sigma_arr = jnp.asarray(min_sigma, dtype=jnp.float32)
    max_sigma_arr = jnp.asarray(max_sigma, dtype=jnp.float32)
    num_sigma_arr = jnp.asarray(num_sigma, dtype=jnp.int32)
    threshold_arr = jnp.asarray(threshold, dtype=jnp.float32)
    
    # Validation
    min_sigma_validated = lax.cond(
        min_sigma_arr <= 0,
        lambda x: jnp.asarray(1.0, dtype=jnp.float32),
        lambda x: x,
        min_sigma_arr
    )
    
    max_sigma_validated = lax.cond(
        max_sigma_arr <= 0,
        lambda x: jnp.asarray(30.0, dtype=jnp.float32),
        lambda x: x,
        max_sigma_arr
    )
    
    max_sigma_validated = lax.cond(
        max_sigma_validated < min_sigma_validated,
        lambda x: min_sigma_validated,
        lambda x: x,
        max_sigma_validated
    )
    
    num_sigma_validated = lax.cond(
        num_sigma_arr <= 0,
        lambda x: jnp.asarray(10, dtype=jnp.int32),
        lambda x: x,
        num_sigma_arr
    )
    
    return HessianBlobConfig(
        min_sigma=min_sigma_validated,
        max_sigma=max_sigma_validated,
        num_sigma=num_sigma_validated,
        threshold=threshold_arr
    )


def make_enhanced_blob_detection_config(
    min_blob_size: float = 5.0,
    max_blob_size: float = 50.0,
    detection_threshold: float = 0.05,
    use_ridge_detection: bool = True,
    use_watershed: bool = True
) -> EnhancedBlobDetectionConfig:
    """
    Factory function to create an EnhancedBlobDetectionConfig PyTree with validation.
    
    Parameters
    ----------
    min_blob_size : float
        Minimum expected blob size (default: 5.0)
    max_blob_size : float
        Maximum expected blob size (default: 50.0)
    detection_threshold : float
        Overall detection threshold (default: 0.05)
    use_ridge_detection : bool
        Enable ridge detection for elongated objects (default: True)
    use_watershed : bool
        Enable watershed for overlapping blobs (default: True)
    
    Returns
    -------
    EnhancedBlobDetectionConfig
        Validated enhanced blob detection configuration PyTree
    """
    # Convert to JAX arrays
    min_blob_size_arr = jnp.asarray(min_blob_size, dtype=jnp.float32)
    max_blob_size_arr = jnp.asarray(max_blob_size, dtype=jnp.float32)
    detection_threshold_arr = jnp.asarray(detection_threshold, dtype=jnp.float32)
    use_ridge_detection_arr = jnp.asarray(use_ridge_detection, dtype=jnp.bool_)
    use_watershed_arr = jnp.asarray(use_watershed, dtype=jnp.bool_)
    
    # Validation
    min_blob_size_validated = lax.cond(
        min_blob_size_arr <= 0,
        lambda x: jnp.asarray(5.0, dtype=jnp.float32),
        lambda x: x,
        min_blob_size_arr
    )
    
    max_blob_size_validated = lax.cond(
        max_blob_size_arr <= 0,
        lambda x: jnp.asarray(50.0, dtype=jnp.float32),
        lambda x: x,
        max_blob_size_arr
    )
    
    max_blob_size_validated = lax.cond(
        max_blob_size_validated < min_blob_size_validated,
        lambda x: min_blob_size_validated,
        lambda x: x,
        max_blob_size_validated
    )
    
    detection_threshold_validated = lax.cond(
        detection_threshold_arr <= 0,
        lambda x: jnp.asarray(0.05, dtype=jnp.float32),
        lambda x: x,
        detection_threshold_arr
    )
    
    return EnhancedBlobDetectionConfig(
        min_blob_size=min_blob_size_validated,
        max_blob_size=max_blob_size_validated,
        detection_threshold=detection_threshold_validated,
        use_ridge_detection=use_ridge_detection_arr,
        use_watershed=use_watershed_arr
    )