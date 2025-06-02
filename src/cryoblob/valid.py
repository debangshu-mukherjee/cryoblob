"""
Module: valid
-------------
Pydantic models for data validation and configuration management
in the cryoblob preprocessing pipeline. This module provides
type-safe validation for preprocessing parameters, file paths,
and blob detection configurations.

Classes
-------
- `PreprocessingConfig`:
    Configuration for image preprocessing parameters
- `BlobDetectionConfig`:
    Configuration for blob detection parameters
- `FileProcessingConfig`:
    Configuration for file processing and batch operations
- `MRCMetadata`:
    Validation for MRC file metadata
- `ValidationPipeline`:
    Main pipeline class for validating all configurations
- `RidgeDetectionConfig`:
    Configuration for ridge detection parameters
- `WatershedConfig`:
    Configuration for watershed segmentation parameters
- `EnhancedBlobDetectionConfig`:
    Configuration for enhanced blob detection combining multiple methods
- `HessianBlobConfig`:
    Configuration for Hessian-based blob detection
"""

from pathlib import Path

from beartype.typing import Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.types import PositiveFloat, PositiveInt


class PreprocessingConfig(BaseModel):
    """
    Configuration model for image preprocessing parameters.

    This validates all parameters used in the preprocessing function
    to ensure they are within valid ranges and types before being
    passed to JAX-compiled functions.
    """

    exponential: bool = Field(
        default=True, description="Apply exponential function to enhance contrast"
    )

    logarizer: bool = Field(
        default=False, description="Apply logarithmic transformation"
    )

    gblur: int = Field(
        default=2, ge=0, le=50, description="Gaussian blur sigma (0 means no blur)"
    )

    background: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Background subtraction sigma (0 means no subtraction)",
    )

    apply_filter: int = Field(
        default=0,
        ge=0,
        le=20,
        description="Wiener filter kernel size (0 means no filter)",
    )

    @field_validator("gblur", "background")
    @classmethod
    def validate_sigma_values(cls, v: int) -> int:
        """Ensure sigma values are reasonable for image processing."""
        if v > 0 and v < 1:
            raise ValueError("Sigma values should be >= 1 when applied")
        return v

    @model_validator(mode="after")
    def validate_conflicting_options(self):
        """Ensure conflicting preprocessing options aren't both enabled."""
        if self.exponential and self.logarizer:
            raise ValueError(
                "Cannot apply both exponential and logarithmic transformations"
            )
        return self

    class Config:
        frozen = True  # Immutable for JAX compatibility
        extra = "forbid"  # Prevent extra fields


class BlobDetectionConfig(BaseModel):
    """
    Configuration model for blob detection parameters.

    Validates parameters used in blob_list_log function.
    """

    min_blob_size: PositiveFloat = Field(
        default=5.0, le=1000.0, description="Minimum blob size to detect (pixels)"
    )

    max_blob_size: PositiveFloat = Field(
        default=20.0, le=2000.0, description="Maximum blob size to detect (pixels)"
    )

    blob_step: PositiveFloat = Field(
        default=1.0, le=10.0, description="Step size between consecutive blob scales"
    )

    downscale: PositiveFloat = Field(
        default=4.0, le=20.0, description="Image downscaling factor before detection"
    )

    std_threshold: PositiveFloat = Field(
        default=6.0,
        le=20.0,
        description="Threshold in standard deviations for blob detection",
    )

    @field_validator("max_blob_size")
    @classmethod
    def validate_max_blob_size(cls, v: float, info) -> float:
        """Ensure max_blob_size > min_blob_size."""
        if hasattr(info, "data") and "min_blob_size" in info.data:
            min_size = info.data["min_blob_size"]
            if v <= min_size:
                raise ValueError(
                    f"max_blob_size ({v}) must be > min_blob_size ({min_size})"
                )
        return v

    class Config:
        frozen = True
        extra = "forbid"


class FileProcessingConfig(BaseModel):
    """
    Configuration model for file processing and batch operations.

    Validates parameters used in folder_blobs function.
    """

    folder_location: Path = Field(description="Path to folder containing images")

    file_type: Literal["mrc", "tiff", "png", "jpg"] = Field(
        default="mrc", description="File type to process"
    )

    blob_downscale: PositiveFloat = Field(
        default=7.0, le=50.0, description="Downscaling factor for blob detection"
    )

    target_memory_gb: PositiveFloat = Field(
        default=4.0, le=128.0, description="Target GPU memory usage in GB"
    )

    stream_large_files: bool = Field(
        default=True, description="Whether to use streaming for large files"
    )

    batch_size: Optional[PositiveInt] = Field(
        default=None, le=1000, description="Override automatic batch size calculation"
    )

    @field_validator("folder_location")
    @classmethod
    def validate_folder_exists(cls, v: Path) -> Path:
        """Ensure the folder exists and is accessible."""
        if not v.exists():
            raise ValueError(f"Folder does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"Path is not a directory: {v}")
        return v

    class Config:
        frozen = True
        extra = "forbid"


class MRCMetadata(BaseModel):
    """
    Validation model for MRC file metadata.

    Ensures MRC file headers contain valid values.
    """

    voxel_size: Tuple[PositiveFloat, PositiveFloat, PositiveFloat] = Field(
        description="Voxel size in Angstroms (Z, Y, X)"
    )

    origin: Tuple[float, float, float] = Field(
        description="Origin coordinates (Z, Y, X)"
    )

    data_min: float = Field(description="Minimum pixel value")
    data_max: float = Field(description="Maximum pixel value")
    data_mean: float = Field(description="Mean pixel value")

    mode: int = Field(
        ge=0, le=6, description="MRC data type mode (0=int8, 1=int16, 2=float32, etc.)"
    )

    image_shape: Tuple[PositiveInt, PositiveInt] = Field(
        description="Image dimensions (height, width)"
    )

    @field_validator("data_max")
    @classmethod
    def validate_data_range(cls, v: float, info) -> float:
        """Ensure data_max > data_min."""
        if hasattr(info, "data") and "data_min" in info.data:
            data_min = info.data["data_min"]
            if v <= data_min:
                raise ValueError(f"data_max ({v}) must be > data_min ({data_min})")
        return v

    @field_validator("data_mean")
    @classmethod
    def validate_mean_in_range(cls, v: float, info) -> float:
        """Ensure data_mean is between data_min and data_max."""
        if (
            hasattr(info, "data")
            and "data_min" in info.data
            and "data_max" in info.data
        ):
            data_min = info.data["data_min"]
            data_max = info.data["data_max"]
            if not (data_min <= v <= data_max):
                raise ValueError(
                    f"data_mean ({v}) must be between data_min ({data_min}) and data_max ({data_max})"
                )
        return v

    class Config:
        frozen = True
        extra = "forbid"


class AdaptiveFilterConfig(BaseModel):
    """
    Configuration model for adaptive filtering parameters.

    Validates parameters used in adaptive_wiener and adaptive_threshold functions.
    """

    kernel_size: Union[PositiveInt, Tuple[PositiveInt, PositiveInt]] = Field(
        default=3, description="Kernel size for filtering"
    )

    initial_noise: PositiveFloat = Field(
        default=0.1,
        le=1.0,
        description="Initial noise estimate for adaptive Wiener filter",
    )

    initial_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Initial threshold for adaptive thresholding",
    )

    initial_slope: PositiveFloat = Field(
        default=10.0, le=100.0, description="Initial slope for sigmoid thresholding"
    )

    learning_rate: PositiveFloat = Field(
        default=0.01, le=1.0, description="Learning rate for optimization"
    )

    iterations: PositiveInt = Field(
        default=100, le=1000, description="Number of optimization iterations"
    )

    @field_validator("kernel_size")
    @classmethod
    def validate_kernel_size(
        cls, v: Union[int, Tuple[int, int]]
    ) -> Union[int, Tuple[int, int]]:
        """Ensure kernel size is odd for proper centering."""
        if isinstance(v, int):
            if v % 2 == 0:
                raise ValueError(f"Kernel size must be odd, got {v}")
        elif isinstance(v, tuple):
            if len(v) != 2:
                raise ValueError(
                    f"Kernel size tuple must have 2 elements, got {len(v)}"
                )
            if v[0] % 2 == 0 or v[1] % 2 == 0:
                raise ValueError(f"Both kernel dimensions must be odd, got {v}")
        return v

    class Config:
        frozen = True
        extra = "forbid"


class ValidationPipeline(BaseModel):
    """
    Main validation pipeline that combines all configuration models.

    This provides a single entry point for validating complete
    processing configurations.
    """

    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig,
        description="Image preprocessing configuration",
    )

    blob_detection: BlobDetectionConfig = Field(
        default_factory=BlobDetectionConfig, description="Blob detection configuration"
    )

    file_processing: Optional[FileProcessingConfig] = Field(
        default=None, description="File processing configuration (for batch operations)"
    )

    adaptive_filtering: Optional[AdaptiveFilterConfig] = Field(
        default=None, description="Adaptive filtering configuration"
    )

    def validate_for_single_image(
        self,
    ) -> Tuple[PreprocessingConfig, BlobDetectionConfig]:
        """
        Validate configuration for single image processing.

        Returns
        -------
        - preprocessing_config: Validated preprocessing parameters
        - blob_config: Validated blob detection parameters
        """
        return self.preprocessing, self.blob_detection

    def validate_for_batch_processing(
        self,
    ) -> Tuple[PreprocessingConfig, BlobDetectionConfig, FileProcessingConfig]:
        """
        Validate configuration for batch file processing.

        Returns
        -------
        - preprocessing_config: Validated preprocessing parameters
        - blob_config: Validated blob detection parameters
        - file_config: Validated file processing parameters

        Raises
        ------
        ValueError: If file_processing configuration is not provided
        """
        if self.file_processing is None:
            raise ValueError(
                "file_processing configuration is required for batch processing"
            )

        return self.preprocessing, self.blob_detection, self.file_processing

    def validate_for_adaptive_processing(
        self,
    ) -> Tuple[PreprocessingConfig, AdaptiveFilterConfig]:
        """
        Validate configuration for adaptive filtering.

        Returns
        -------
        - preprocessing_config: Validated preprocessing parameters
        - adaptive_config: Validated adaptive filtering parameters

        Raises
        ------
        ValueError: If adaptive_filtering configuration is not provided
        """
        if self.adaptive_filtering is None:
            raise ValueError(
                "adaptive_filtering configuration is required for adaptive processing"
            )

        return self.preprocessing, self.adaptive_filtering

    def to_preprocessing_kwargs(self) -> dict:
        """
        Convert preprocessing config to kwargs dict for existing functions.

        Returns
        -------
        - kwargs: Dictionary compatible with existing preprocessing function
        """
        return self.preprocessing.model_dump()

    def to_blob_kwargs(self) -> dict:
        """
        Convert blob detection config to kwargs dict for existing functions.

        Returns
        -------
        - kwargs: Dictionary compatible with existing blob_list_log function
        """
        return self.blob_detection.model_dump()

    class Config:
        frozen = True
        extra = "forbid"


# Factory functions for common configurations
def create_default_pipeline() -> ValidationPipeline:
    """Create a validation pipeline with default settings."""
    return ValidationPipeline()


def create_high_quality_pipeline() -> ValidationPipeline:
    """Create a validation pipeline optimized for high-quality blob detection."""
    return ValidationPipeline(
        preprocessing=PreprocessingConfig(
            exponential=True, logarizer=False, gblur=1, background=10, apply_filter=3
        ),
        blob_detection=BlobDetectionConfig(
            min_blob_size=3.0,
            max_blob_size=30.0,
            blob_step=0.5,
            downscale=2.0,
            std_threshold=4.0,
        ),
    )


def create_fast_pipeline() -> ValidationPipeline:
    """Create a validation pipeline optimized for speed."""
    return ValidationPipeline(
        preprocessing=PreprocessingConfig(
            exponential=False, logarizer=False, gblur=0, background=0, apply_filter=0
        ),
        blob_detection=BlobDetectionConfig(
            min_blob_size=5.0,
            max_blob_size=15.0,
            blob_step=2.0,
            downscale=8.0,
            std_threshold=8.0,
        ),
    )


def validate_mrc_metadata(
    voxel_size: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    data_min: float,
    data_max: float,
    data_mean: float,
    mode: int,
    image_shape: Tuple[int, int],
) -> MRCMetadata:
    """
    Validate MRC metadata and return validated model.

    Parameters
    ----------
    - voxel_size: Voxel dimensions (Z, Y, X)
    - origin: Origin coordinates (Z, Y, X)
    - data_min: Minimum pixel value
    - data_max: Maximum pixel value
    - data_mean: Mean pixel value
    - mode: MRC data type mode
    - image_shape: Image dimensions (height, width)

    Returns
    -------
    - metadata: Validated MRC metadata model

    Raises
    ------
    ValidationError: If any metadata values are invalid
    """
    return MRCMetadata(
        voxel_size=voxel_size,
        origin=origin,
        data_min=data_min,
        data_max=data_max,
        data_mean=data_mean,
        mode=mode,
        image_shape=image_shape,
    )


class RidgeDetectionConfig(BaseModel):
    """
    Configuration model for ridge detection parameters.

    Validates parameters used for detecting elongated objects.
    """

    min_scale: PositiveFloat = Field(
        default=1.0, le=50.0, description="Minimum scale for ridge detection"
    )

    max_scale: PositiveFloat = Field(
        default=10.0, le=100.0, description="Maximum scale for ridge detection"
    )

    num_scales: PositiveInt = Field(
        default=10, le=50, description="Number of scales to test"
    )

    ridge_threshold: PositiveFloat = Field(
        default=0.01, le=1.0, description="Ridge strength threshold"
    )

    enable_multi_scale: bool = Field(
        default=True, description="Use multi-scale ridge detection"
    )

    @field_validator("max_scale")
    @classmethod
    def validate_scale_range(cls, v: float, info) -> float:
        """Ensure max_scale > min_scale."""
        if hasattr(info, "data") and "min_scale" in info.data:
            min_scale = info.data["min_scale"]
            if v <= min_scale:
                raise ValueError(f"max_scale ({v}) must be > min_scale ({min_scale})")
        return v

    class Config:
        frozen = True
        extra = "forbid"


class WatershedConfig(BaseModel):
    """
    Configuration model for watershed segmentation parameters.

    Validates parameters used for separating overlapping blobs.
    """

    min_marker_distance: PositiveFloat = Field(
        default=5.0, le=50.0, description="Minimum distance between watershed markers"
    )

    flooding_iterations: PositiveInt = Field(
        default=10, le=100, description="Number of flooding iterations"
    )

    enable_adaptive_markers: bool = Field(
        default=True, description="Use adaptive marker generation"
    )

    distance_transform_method: str = Field(
        default="euclidean",
        pattern="^(euclidean|manhattan|chebyshev)$",
        description="Distance transform method for marker generation",
    )

    marker_erosion_size: PositiveInt = Field(
        default=3, le=15, description="Erosion size for marker refinement"
    )

    class Config:
        frozen = True
        extra = "forbid"


class HessianBlobConfig(BaseModel):
    """
    Configuration model for Hessian-based blob detection.

    Validates parameters for Determinant of Hessian blob detection.
    """

    scale_normalization: bool = Field(
        default=True, description="Apply scale normalization to Hessian determinant"
    )

    eigenvalue_threshold: PositiveFloat = Field(
        default=0.001, le=1.0, description="Eigenvalue threshold for blob detection"
    )

    boundary_enhancement: bool = Field(
        default=True, description="Enhance blob boundaries using gradient information"
    )

    non_maximum_suppression: bool = Field(
        default=True, description="Apply non-maximum suppression to detected blobs"
    )

    suppression_radius: PositiveFloat = Field(
        default=2.0, le=20.0, description="Radius for non-maximum suppression"
    )

    class Config:
        frozen = True
        extra = "forbid"


class EnhancedBlobDetectionConfig(BaseModel):
    """
    Configuration model for enhanced blob detection combining multiple methods.

    This integrates circular blob detection, ridge detection, and watershed segmentation.
    """

    # Base blob detection parameters
    min_blob_size: PositiveFloat = Field(
        default=5.0, le=1000.0, description="Minimum blob size to detect"
    )

    max_blob_size: PositiveFloat = Field(
        default=20.0, le=2000.0, description="Maximum blob size to detect"
    )

    blob_step: PositiveFloat = Field(
        default=1.0, le=10.0, description="Step size between blob scales"
    )

    downscale: PositiveFloat = Field(
        default=4.0, le=20.0, description="Image downscaling factor"
    )

    std_threshold: PositiveFloat = Field(
        default=6.0, le=20.0, description="Standard deviation threshold for detection"
    )

    # Enhanced detection options
    enable_ridge_detection: bool = Field(
        default=True, description="Enable ridge detection for elongated objects"
    )

    enable_watershed: bool = Field(
        default=True, description="Enable watershed segmentation for overlapping blobs"
    )

    enable_hessian_blobs: bool = Field(
        default=False, description="Use Hessian-based blob detection instead of LoG"
    )

    # Method-specific configurations
    ridge_config: Optional[RidgeDetectionConfig] = Field(
        default_factory=RidgeDetectionConfig,
        description="Ridge detection configuration",
    )

    watershed_config: Optional[WatershedConfig] = Field(
        default_factory=WatershedConfig,
        description="Watershed segmentation configuration",
    )

    hessian_config: Optional[HessianBlobConfig] = Field(
        default_factory=HessianBlobConfig,
        description="Hessian blob detection configuration",
    )

    # Post-processing options
    merge_overlapping_detections: bool = Field(
        default=True, description="Merge overlapping detections from different methods"
    )

    overlap_threshold: PositiveFloat = Field(
        default=0.5,
        le=1.0,
        description="IoU threshold for merging overlapping detections",
    )

    confidence_weighting: bool = Field(
        default=True, description="Weight detections by confidence scores"
    )

    @field_validator("max_blob_size")
    @classmethod
    def validate_blob_size_range(cls, v: float, info) -> float:
        """Ensure max_blob_size > min_blob_size."""
        if hasattr(info, "data") and "min_blob_size" in info.data:
            min_size = info.data["min_blob_size"]
            if v <= min_size:
                raise ValueError(
                    f"max_blob_size ({v}) must be > min_blob_size ({min_size})"
                )
        return v

    @model_validator(mode="after")
    def validate_method_dependencies(self):
        """Ensure required configurations are present when methods are enabled."""
        if self.enable_ridge_detection and self.ridge_config is None:
            raise ValueError(
                "ridge_config is required when enable_ridge_detection is True"
            )

        if self.enable_watershed and self.watershed_config is None:
            raise ValueError(
                "watershed_config is required when enable_watershed is True"
            )

        if self.enable_hessian_blobs and self.hessian_config is None:
            raise ValueError(
                "hessian_config is required when enable_hessian_blobs is True"
            )

        return self

    def to_enhanced_kwargs(self) -> dict:
        """
        Convert configuration to kwargs dict for enhanced_blob_detection function.

        Returns
        -------
        - kwargs: Dictionary compatible with enhanced_blob_detection function
        """
        base_kwargs = {
            "min_blob_size": self.min_blob_size,
            "max_blob_size": self.max_blob_size,
            "blob_step": self.blob_step,
            "downscale": self.downscale,
            "std_threshold": self.std_threshold,
            "use_ridge_detection": self.enable_ridge_detection,
            "use_watershed": self.enable_watershed,
        }

        if self.ridge_config:
            base_kwargs.update(
                {
                    "ridge_threshold": self.ridge_config.ridge_threshold,
                }
            )

        if self.watershed_config:
            base_kwargs.update(
                {
                    "min_marker_distance": self.watershed_config.min_marker_distance,
                }
            )

        return base_kwargs

    class Config:
        frozen = True
        extra = "forbid"


class BlobAnalysisConfig(BaseModel):
    """
    Configuration for blob analysis and post-processing.

    Validates parameters for analyzing detected blobs and combining results.
    """

    size_filtering: bool = Field(
        default=True, description="Filter blobs by size constraints"
    )

    aspect_ratio_filtering: bool = Field(
        default=True, description="Filter blobs by aspect ratio"
    )

    min_aspect_ratio: PositiveFloat = Field(
        default=0.1, le=1.0, description="Minimum aspect ratio for valid blobs"
    )

    max_aspect_ratio: PositiveFloat = Field(
        default=10.0, ge=1.0, description="Maximum aspect ratio for valid blobs"
    )

    circularity_filtering: bool = Field(
        default=False, description="Filter blobs by circularity"
    )

    min_circularity: PositiveFloat = Field(
        default=0.1, le=1.0, description="Minimum circularity for valid blobs"
    )

    convexity_filtering: bool = Field(
        default=False, description="Filter blobs by convexity"
    )

    min_convexity: PositiveFloat = Field(
        default=0.5, le=1.0, description="Minimum convexity for valid blobs"
    )

    inertia_filtering: bool = Field(
        default=False, description="Filter blobs by inertia ratio"
    )

    min_inertia_ratio: PositiveFloat = Field(
        default=0.01, le=1.0, description="Minimum inertia ratio for valid blobs"
    )

    class Config:
        frozen = True
        extra = "forbid"


# Factory functions for common enhanced configurations
def create_elongated_objects_pipeline() -> EnhancedBlobDetectionConfig:
    """Create a pipeline optimized for elongated objects."""
    return EnhancedBlobDetectionConfig(
        enable_ridge_detection=True,
        enable_watershed=False,
        enable_hessian_blobs=True,
        ridge_config=RidgeDetectionConfig(
            min_scale=2.0, max_scale=15.0, num_scales=15, ridge_threshold=0.005
        ),
        hessian_config=HessianBlobConfig(
            boundary_enhancement=True,
            non_maximum_suppression=True,
            suppression_radius=3.0,
        ),
    )


def create_overlapping_blobs_pipeline() -> EnhancedBlobDetectionConfig:
    """Create a pipeline optimized for overlapping blobs."""
    return EnhancedBlobDetectionConfig(
        enable_ridge_detection=False,
        enable_watershed=True,
        enable_hessian_blobs=True,
        watershed_config=WatershedConfig(
            min_marker_distance=3.0,
            flooding_iterations=15,
            enable_adaptive_markers=True,
        ),
        hessian_config=HessianBlobConfig(
            boundary_enhancement=True, eigenvalue_threshold=0.0005
        ),
        merge_overlapping_detections=True,
        overlap_threshold=0.3,
    )


def create_comprehensive_pipeline() -> EnhancedBlobDetectionConfig:
    """Create a comprehensive pipeline using all methods."""
    return EnhancedBlobDetectionConfig(
        enable_ridge_detection=True,
        enable_watershed=True,
        enable_hessian_blobs=True,
        ridge_config=RidgeDetectionConfig(
            min_scale=1.5, max_scale=12.0, num_scales=12, ridge_threshold=0.008
        ),
        watershed_config=WatershedConfig(
            min_marker_distance=4.0,
            flooding_iterations=12,
            enable_adaptive_markers=True,
        ),
        hessian_config=HessianBlobConfig(
            boundary_enhancement=True,
            non_maximum_suppression=True,
            suppression_radius=2.5,
        ),
        merge_overlapping_detections=True,
        overlap_threshold=0.4,
        confidence_weighting=True,
    )
