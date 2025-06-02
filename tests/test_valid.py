from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydantic import ValidationError

from cryoblob.valid import (
    AdaptiveFilterConfig,
    BlobDetectionConfig,
    FileProcessingConfig,
    MRCMetadata,
    PreprocessingConfig,
    ValidationPipeline,
    create_default_pipeline,
    create_fast_pipeline,
    create_high_quality_pipeline,
    validate_mrc_metadata,
    RidgeDetectionConfig,
    WatershedConfig,
    HessianBlobConfig,
    EnhancedBlobDetectionConfig,
    BlobAnalysisConfig,
    create_elongated_objects_pipeline,
    create_overlapping_blobs_pipeline,
    create_comprehensive_pipeline,
)


class TestPreprocessingConfig:
    def test_default_config(self):
        config = PreprocessingConfig()
        assert config.exponential is True
        assert config.logarizer is False
        assert config.gblur == 2
        assert config.background == 0
        assert config.apply_filter == 0

    def test_valid_custom_config(self):
        config = PreprocessingConfig(
            exponential=False, logarizer=True, gblur=5, background=10, apply_filter=3
        )
        assert config.logarizer is True
        assert config.gblur == 5

    def test_conflicting_transformations(self):
        with pytest.raises(
            ValidationError, match="Cannot apply both exponential and logarithmic"
        ):
            PreprocessingConfig(exponential=True, logarizer=True)

    def test_invalid_sigma_values(self):
        with pytest.raises(ValidationError):
            PreprocessingConfig(gblur=100)

        with pytest.raises(ValidationError):
            PreprocessingConfig(background=-1)

    def test_immutability(self):
        config = PreprocessingConfig()
        with pytest.raises(ValidationError):
            config.exponential = False


class TestBlobDetectionConfig:
    def test_default_config(self):
        config = BlobDetectionConfig()
        assert config.min_blob_size == 5.0
        assert config.max_blob_size == 20.0
        assert config.blob_step == 1.0
        assert config.downscale == 4.0
        assert config.std_threshold == 6.0

    def test_valid_custom_config(self):
        config = BlobDetectionConfig(
            min_blob_size=2.0,
            max_blob_size=50.0,
            blob_step=0.5,
            downscale=8.0,
            std_threshold=4.0,
        )
        assert config.min_blob_size == 2.0
        assert config.max_blob_size == 50.0

    def test_invalid_blob_size_range(self):
        with pytest.raises(
            ValidationError, match="max_blob_size.*must be.*min_blob_size"
        ):
            BlobDetectionConfig(min_blob_size=20.0, max_blob_size=10.0)

    def test_boundary_values(self):
        config = BlobDetectionConfig(
            min_blob_size=1.0, max_blob_size=1000.0, std_threshold=20.0
        )
        assert config.min_blob_size == 1.0

        with pytest.raises(ValidationError):
            BlobDetectionConfig(max_blob_size=3000.0)


class TestFileProcessingConfig:
    def test_valid_config_with_existing_folder(self):
        with TemporaryDirectory() as temp_dir:
            config = FileProcessingConfig(
                folder_location=Path(temp_dir), file_type="mrc", blob_downscale=5.0
            )
            assert config.folder_location == Path(temp_dir)
            assert config.file_type == "mrc"

    def test_nonexistent_folder(self):
        with pytest.raises(ValidationError, match="Folder does not exist"):
            FileProcessingConfig(folder_location=Path("/nonexistent/folder"))

    def test_file_instead_of_directory(self):
        with TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test")

            with pytest.raises(ValidationError, match="Path is not a directory"):
                FileProcessingConfig(folder_location=test_file)

    def test_valid_file_types(self):
        with TemporaryDirectory() as temp_dir:
            for file_type in ["mrc", "tiff", "png", "jpg"]:
                config = FileProcessingConfig(
                    folder_location=Path(temp_dir), file_type=file_type
                )
                assert config.file_type == file_type


class TestMRCMetadata:
    def test_valid_metadata(self):
        metadata = MRCMetadata(
            voxel_size=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            data_min=0.0,
            data_max=255.0,
            data_mean=127.5,
            mode=2,
            image_shape=(512, 512),
        )
        assert metadata.voxel_size == (1.0, 1.0, 1.0)
        assert metadata.data_mean == 127.5

    def test_invalid_data_range(self):
        with pytest.raises(ValidationError, match="data_max.*must be.*data_min"):
            MRCMetadata(
                voxel_size=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                data_min=100.0,
                data_max=50.0,
                data_mean=75.0,
                mode=2,
                image_shape=(512, 512),
            )

    def test_invalid_mean_outside_range(self):
        with pytest.raises(ValidationError, match="data_mean.*must be between"):
            MRCMetadata(
                voxel_size=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                data_min=0.0,
                data_max=100.0,
                data_mean=150.0,
                mode=2,
                image_shape=(512, 512),
            )

    def test_invalid_mode(self):
        with pytest.raises(ValidationError):
            MRCMetadata(
                voxel_size=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                data_min=0.0,
                data_max=100.0,
                data_mean=50.0,
                mode=10,
                image_shape=(512, 512),
            )


class TestAdaptiveFilterConfig:
    def test_default_config(self):
        config = AdaptiveFilterConfig()
        assert config.kernel_size == 3
        assert config.initial_noise == 0.1
        assert config.learning_rate == 0.01

    def test_kernel_size_validation(self):
        config = AdaptiveFilterConfig(kernel_size=5)
        assert config.kernel_size == 5

        config = AdaptiveFilterConfig(kernel_size=(3, 5))
        assert config.kernel_size == (3, 5)

        with pytest.raises(ValidationError, match="must be odd"):
            AdaptiveFilterConfig(kernel_size=4)

        with pytest.raises(ValidationError, match="must be odd"):
            AdaptiveFilterConfig(kernel_size=(3, 4))


class TestRidgeDetectionConfig:
    def test_default_config(self):
        config = RidgeDetectionConfig()
        assert config.min_scale == 1.0
        assert config.max_scale == 10.0
        assert config.num_scales == 10
        assert config.ridge_threshold == 0.01
        assert config.enable_multi_scale is True

    def test_valid_custom_config(self):
        config = RidgeDetectionConfig(
            min_scale=2.0,
            max_scale=15.0,
            num_scales=12,
            ridge_threshold=0.005,
            enable_multi_scale=False,
        )
        assert config.min_scale == 2.0
        assert config.max_scale == 15.0
        assert config.num_scales == 12
        assert config.ridge_threshold == 0.005
        assert config.enable_multi_scale is False

    def test_invalid_scale_range(self):
        with pytest.raises(ValidationError, match="max_scale.*must be.*min_scale"):
            RidgeDetectionConfig(min_scale=15.0, max_scale=10.0)

    def test_boundary_values(self):
        config = RidgeDetectionConfig(
            min_scale=0.5, max_scale=50.0, num_scales=50, ridge_threshold=1.0
        )
        assert config.min_scale == 0.5
        assert config.max_scale == 50.0

        with pytest.raises(ValidationError):
            RidgeDetectionConfig(max_scale=150.0)

        with pytest.raises(ValidationError):
            RidgeDetectionConfig(num_scales=100)

    def test_immutability(self):
        config = RidgeDetectionConfig()
        with pytest.raises(ValidationError):
            config.min_scale = 5.0


class TestWatershedConfig:
    def test_default_config(self):
        config = WatershedConfig()
        assert config.min_marker_distance == 5.0
        assert config.flooding_iterations == 10
        assert config.enable_adaptive_markers is True
        assert config.distance_transform_method == "euclidean"
        assert config.marker_erosion_size == 3

    def test_valid_custom_config(self):
        config = WatershedConfig(
            min_marker_distance=3.0,
            flooding_iterations=15,
            enable_adaptive_markers=False,
            distance_transform_method="manhattan",
            marker_erosion_size=5,
        )
        assert config.min_marker_distance == 3.0
        assert config.flooding_iterations == 15
        assert config.enable_adaptive_markers is False
        assert config.distance_transform_method == "manhattan"
        assert config.marker_erosion_size == 5

    def test_invalid_distance_transform_method(self):
        with pytest.raises(ValidationError):
            WatershedConfig(distance_transform_method="invalid_method")

    def test_boundary_values(self):
        config = WatershedConfig(
            min_marker_distance=50.0, flooding_iterations=100, marker_erosion_size=15
        )
        assert config.min_marker_distance == 50.0
        assert config.flooding_iterations == 100
        assert config.marker_erosion_size == 15

        with pytest.raises(ValidationError):
            WatershedConfig(min_marker_distance=100.0)

        with pytest.raises(ValidationError):
            WatershedConfig(flooding_iterations=200)


class TestHessianBlobConfig:
    def test_default_config(self):
        config = HessianBlobConfig()
        assert config.scale_normalization is True
        assert config.eigenvalue_threshold == 0.001
        assert config.boundary_enhancement is True
        assert config.non_maximum_suppression is True
        assert config.suppression_radius == 2.0

    def test_valid_custom_config(self):
        config = HessianBlobConfig(
            scale_normalization=False,
            eigenvalue_threshold=0.0005,
            boundary_enhancement=False,
            non_maximum_suppression=False,
            suppression_radius=5.0,
        )
        assert config.scale_normalization is False
        assert config.eigenvalue_threshold == 0.0005
        assert config.boundary_enhancement is False
        assert config.non_maximum_suppression is False
        assert config.suppression_radius == 5.0

    def test_boundary_values(self):
        config = HessianBlobConfig(eigenvalue_threshold=1.0, suppression_radius=20.0)
        assert config.eigenvalue_threshold == 1.0
        assert config.suppression_radius == 20.0

        with pytest.raises(ValidationError):
            HessianBlobConfig(eigenvalue_threshold=2.0)

        with pytest.raises(ValidationError):
            HessianBlobConfig(suppression_radius=25.0)


class TestEnhancedBlobDetectionConfig:
    def test_default_config(self):
        config = EnhancedBlobDetectionConfig()
        assert config.min_blob_size == 5.0
        assert config.max_blob_size == 20.0
        assert config.enable_ridge_detection is True
        assert config.enable_watershed is True
        assert config.enable_hessian_blobs is False
        assert config.ridge_config is not None
        assert config.watershed_config is not None

    def test_valid_custom_config(self):
        ridge_config = RidgeDetectionConfig(min_scale=2.0, max_scale=12.0)
        watershed_config = WatershedConfig(min_marker_distance=3.0)
        hessian_config = HessianBlobConfig(scale_normalization=False)

        config = EnhancedBlobDetectionConfig(
            min_blob_size=3.0,
            max_blob_size=25.0,
            enable_ridge_detection=True,
            enable_watershed=True,
            enable_hessian_blobs=True,
            ridge_config=ridge_config,
            watershed_config=watershed_config,
            hessian_config=hessian_config,
            merge_overlapping_detections=False,
            overlap_threshold=0.3,
        )
        assert config.min_blob_size == 3.0
        assert config.max_blob_size == 25.0
        assert config.enable_hessian_blobs is True
        assert config.merge_overlapping_detections is False
        assert config.overlap_threshold == 0.3

    def test_invalid_blob_size_range(self):
        with pytest.raises(
            ValidationError, match="max_blob_size.*must be.*min_blob_size"
        ):
            EnhancedBlobDetectionConfig(min_blob_size=25.0, max_blob_size=15.0)

    def test_method_dependency_validation(self):
        # Should work with default configs
        config = EnhancedBlobDetectionConfig(
            enable_ridge_detection=True,
            enable_watershed=True,
            enable_hessian_blobs=True,
        )
        assert config.ridge_config is not None
        assert config.watershed_config is not None
        assert config.hessian_config is not None

        # Should work when methods are disabled
        config = EnhancedBlobDetectionConfig(
            enable_ridge_detection=False,
            enable_watershed=False,
            enable_hessian_blobs=False,
        )
        assert config.ridge_config is not None  # Still has default
        assert config.watershed_config is not None  # Still has default

    def test_to_enhanced_kwargs_conversion(self):
        ridge_config = RidgeDetectionConfig(ridge_threshold=0.005)
        watershed_config = WatershedConfig(min_marker_distance=4.0)

        config = EnhancedBlobDetectionConfig(
            min_blob_size=4.0,
            max_blob_size=22.0,
            ridge_config=ridge_config,
            watershed_config=watershed_config,
        )

        kwargs = config.to_enhanced_kwargs()

        assert kwargs["min_blob_size"] == 4.0
        assert kwargs["max_blob_size"] == 22.0
        assert kwargs["ridge_threshold"] == 0.005
        assert kwargs["min_marker_distance"] == 4.0
        assert kwargs["use_ridge_detection"] is True
        assert kwargs["use_watershed"] is True

    def test_immutability(self):
        config = EnhancedBlobDetectionConfig()
        with pytest.raises(ValidationError):
            config.min_blob_size = 10.0


class TestBlobAnalysisConfig:
    def test_default_config(self):
        config = BlobAnalysisConfig()
        assert config.size_filtering is True
        assert config.aspect_ratio_filtering is True
        assert config.min_aspect_ratio == 0.1
        assert config.max_aspect_ratio == 10.0
        assert config.circularity_filtering is False

    def test_valid_custom_config(self):
        config = BlobAnalysisConfig(
            size_filtering=False,
            aspect_ratio_filtering=True,
            min_aspect_ratio=0.2,
            max_aspect_ratio=5.0,
            circularity_filtering=True,
            min_circularity=0.3,
            convexity_filtering=True,
            min_convexity=0.7,
            inertia_filtering=True,
            min_inertia_ratio=0.05,
        )
        assert config.size_filtering is False
        assert config.min_aspect_ratio == 0.2
        assert config.max_aspect_ratio == 5.0
        assert config.circularity_filtering is True
        assert config.min_circularity == 0.3
        assert config.convexity_filtering is True
        assert config.min_convexity == 0.7
        assert config.inertia_filtering is True
        assert config.min_inertia_ratio == 0.05

    def test_boundary_values(self):
        config = BlobAnalysisConfig(
            min_aspect_ratio=1.0,
            max_aspect_ratio=1.0,
            min_circularity=1.0,
            min_convexity=1.0,
            min_inertia_ratio=1.0,
        )
        assert config.min_aspect_ratio == 1.0
        assert config.max_aspect_ratio == 1.0


class TestValidationPipeline:
    def test_default_pipeline(self):
        pipeline = ValidationPipeline()
        assert isinstance(pipeline.preprocessing, PreprocessingConfig)
        assert isinstance(pipeline.blob_detection, BlobDetectionConfig)
        assert pipeline.file_processing is None

    def test_single_image_validation(self):
        pipeline = ValidationPipeline()
        prep_config, blob_config = pipeline.validate_for_single_image()

        assert isinstance(prep_config, PreprocessingConfig)
        assert isinstance(blob_config, BlobDetectionConfig)

    def test_batch_processing_validation_without_file_config(self):
        pipeline = ValidationPipeline()

        with pytest.raises(
            ValueError, match="file_processing configuration is required"
        ):
            pipeline.validate_for_batch_processing()

    def test_batch_processing_validation_with_file_config(self):
        with TemporaryDirectory() as temp_dir:
            file_config = FileProcessingConfig(folder_location=Path(temp_dir))
            pipeline = ValidationPipeline(file_processing=file_config)

            prep_config, blob_config, file_config = (
                pipeline.validate_for_batch_processing()
            )

            assert isinstance(prep_config, PreprocessingConfig)
            assert isinstance(blob_config, BlobDetectionConfig)
            assert isinstance(file_config, FileProcessingConfig)

    def test_adaptive_processing_validation(self):
        adaptive_config = AdaptiveFilterConfig()
        pipeline = ValidationPipeline(adaptive_filtering=adaptive_config)

        prep_config, adaptive_config = pipeline.validate_for_adaptive_processing()

        assert isinstance(prep_config, PreprocessingConfig)
        assert isinstance(adaptive_config, AdaptiveFilterConfig)

    def test_to_kwargs_conversion(self):
        pipeline = ValidationPipeline()

        prep_kwargs = pipeline.to_preprocessing_kwargs()
        blob_kwargs = pipeline.to_blob_kwargs()

        assert isinstance(prep_kwargs, dict)
        assert isinstance(blob_kwargs, dict)
        assert "exponential" in prep_kwargs
        assert "min_blob_size" in blob_kwargs


class TestFactoryFunctions:
    def test_create_default_pipeline(self):
        pipeline = create_default_pipeline()
        assert isinstance(pipeline, ValidationPipeline)
        assert pipeline.preprocessing.exponential is True

    def test_create_high_quality_pipeline(self):
        pipeline = create_high_quality_pipeline()
        assert pipeline.blob_detection.min_blob_size == 3.0
        assert pipeline.blob_detection.std_threshold == 4.0

    def test_create_fast_pipeline(self):
        pipeline = create_fast_pipeline()
        assert pipeline.preprocessing.exponential is False
        assert pipeline.blob_detection.downscale == 8.0

    def test_create_elongated_objects_pipeline(self):
        pipeline = create_elongated_objects_pipeline()
        assert isinstance(pipeline, EnhancedBlobDetectionConfig)
        assert pipeline.enable_ridge_detection is True
        assert pipeline.enable_watershed is False
        assert pipeline.enable_hessian_blobs is True
        assert pipeline.ridge_config.min_scale == 2.0
        assert pipeline.ridge_config.max_scale == 15.0
        assert pipeline.hessian_config.boundary_enhancement is True

    def test_create_overlapping_blobs_pipeline(self):
        pipeline = create_overlapping_blobs_pipeline()
        assert isinstance(pipeline, EnhancedBlobDetectionConfig)
        assert pipeline.enable_ridge_detection is False
        assert pipeline.enable_watershed is True
        assert pipeline.enable_hessian_blobs is True
        assert pipeline.watershed_config.min_marker_distance == 3.0
        assert pipeline.watershed_config.flooding_iterations == 15
        assert pipeline.merge_overlapping_detections is True
        assert pipeline.overlap_threshold == 0.3

    def test_create_comprehensive_pipeline(self):
        pipeline = create_comprehensive_pipeline()
        assert isinstance(pipeline, EnhancedBlobDetectionConfig)
        assert pipeline.enable_ridge_detection is True
        assert pipeline.enable_watershed is True
        assert pipeline.enable_hessian_blobs is True
        assert pipeline.ridge_config.min_scale == 1.5
        assert pipeline.ridge_config.max_scale == 12.0
        assert pipeline.watershed_config.min_marker_distance == 4.0
        assert pipeline.hessian_config.boundary_enhancement is True
        assert pipeline.merge_overlapping_detections is True
        assert pipeline.confidence_weighting is True

    def test_factory_function_configurations_valid(self):
        # Test that all factory functions produce valid configurations
        pipelines = [
            create_elongated_objects_pipeline(),
            create_overlapping_blobs_pipeline(),
            create_comprehensive_pipeline(),
        ]

        for pipeline in pipelines:
            # Should not raise validation errors
            kwargs = pipeline.to_enhanced_kwargs()
            assert isinstance(kwargs, dict)
            assert "min_blob_size" in kwargs
            assert "max_blob_size" in kwargs
            assert "use_ridge_detection" in kwargs
            assert "use_watershed" in kwargs


class TestMRCMetadataValidation:
    def test_validate_mrc_metadata_function(self):
        metadata = validate_mrc_metadata(
            voxel_size=(1.2, 1.2, 1.2),
            origin=(0.0, 0.0, 0.0),
            data_min=-10.0,
            data_max=100.0,
            data_mean=45.0,
            mode=2,
            image_shape=(1024, 1024),
        )

        assert isinstance(metadata, MRCMetadata)
        assert metadata.voxel_size == (1.2, 1.2, 1.2)
        assert metadata.image_shape == (1024, 1024)

    def test_validate_mrc_metadata_with_invalid_data(self):
        with pytest.raises(ValidationError):
            validate_mrc_metadata(
                voxel_size=(1.2, 1.2, 1.2),
                origin=(0.0, 0.0, 0.0),
                data_min=100.0,
                data_max=50.0,
                data_mean=75.0,
                mode=2,
                image_shape=(1024, 1024),
            )


class TestEnhancedConfigurationIntegration:
    """Test integration between enhanced configurations and validation."""

    def test_enhanced_config_with_validation_pipeline(self):
        # Test that enhanced configs can be used with existing ValidationPipeline
        enhanced_config = EnhancedBlobDetectionConfig(
            min_blob_size=4.0,
            max_blob_size=30.0,
            enable_ridge_detection=True,
            enable_watershed=True,
        )

        # Should be able to extract kwargs for function calls
        kwargs = enhanced_config.to_enhanced_kwargs()

        assert kwargs["min_blob_size"] == 4.0
        assert kwargs["max_blob_size"] == 30.0
        assert kwargs["use_ridge_detection"] is True
        assert kwargs["use_watershed"] is True

    def test_enhanced_config_parameter_inheritance(self):
        # Test that enhanced config properly inherits base blob detection parameters
        enhanced_config = EnhancedBlobDetectionConfig(
            min_blob_size=6.0,
            max_blob_size=35.0,
            blob_step=1.5,
            downscale=6.0,
            std_threshold=7.0,
        )

        kwargs = enhanced_config.to_enhanced_kwargs()

        assert kwargs["min_blob_size"] == 6.0
        assert kwargs["max_blob_size"] == 35.0
        assert kwargs["blob_step"] == 1.5
        assert kwargs["downscale"] == 6.0
        assert kwargs["std_threshold"] == 7.0

    def test_enhanced_config_method_specific_parameters(self):
        # Test that method-specific parameters are properly included
        ridge_config = RidgeDetectionConfig(ridge_threshold=0.008)
        watershed_config = WatershedConfig(min_marker_distance=6.0)

        enhanced_config = EnhancedBlobDetectionConfig(
            ridge_config=ridge_config, watershed_config=watershed_config
        )

        kwargs = enhanced_config.to_enhanced_kwargs()

        assert kwargs["ridge_threshold"] == 0.008
        assert kwargs["min_marker_distance"] == 6.0

    def test_enhanced_config_disabled_methods(self):
        # Test behavior when methods are disabled
        enhanced_config = EnhancedBlobDetectionConfig(
            enable_ridge_detection=False,
            enable_watershed=False,
            enable_hessian_blobs=False,
        )

        kwargs = enhanced_config.to_enhanced_kwargs()

        assert kwargs["use_ridge_detection"] is False
        assert kwargs["use_watershed"] is False
        # Should still have default values for thresholds
        assert "ridge_threshold" in kwargs
        assert "min_marker_distance" in kwargs


if __name__ == "__main__":
    pytest.main([__file__])
