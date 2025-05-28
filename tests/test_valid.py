from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydantic import ValidationError

from cryoblob.valid import (AdaptiveFilterConfig, BlobDetectionConfig,
                            FileProcessingConfig, MRCMetadata,
                            PreprocessingConfig, ValidationPipeline,
                            create_default_pipeline, create_fast_pipeline,
                            create_high_quality_pipeline,
                            validate_mrc_metadata)


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
            PreprocessingConfig(gblur=100)  # Too large

        with pytest.raises(ValidationError):
            PreprocessingConfig(background=-1)  # Negative

    def test_immutability(self):
        config = PreprocessingConfig()
        with pytest.raises(ValidationError):
            config.exponential = False  # Should fail due to frozen=True


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
        # Should work
        config = BlobDetectionConfig(
            min_blob_size=1.0, max_blob_size=1000.0, std_threshold=20.0
        )
        assert config.min_blob_size == 1.0

        # Should fail - exceeds maximum
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
            # Create a file
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test")

            with pytest.raises(ValidationError, match="Path is not a directory"):
                FileProcessingConfig(folder_location=test_file)

    def test_valid_file_types(self):
        with TemporaryDirectory() as temp_dir:
            # Valid file types
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
                data_max=50.0,  # max < min
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
                data_mean=150.0,  # mean > max
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
                mode=10,  # Invalid mode
                image_shape=(512, 512),
            )


class TestAdaptiveFilterConfig:
    def test_default_config(self):
        config = AdaptiveFilterConfig()
        assert config.kernel_size == 3
        assert config.initial_noise == 0.1
        assert config.learning_rate == 0.01

    def test_kernel_size_validation(self):
        # Valid odd kernel sizes
        config = AdaptiveFilterConfig(kernel_size=5)
        assert config.kernel_size == 5

        config = AdaptiveFilterConfig(kernel_size=(3, 5))
        assert config.kernel_size == (3, 5)

        # Invalid even kernel sizes
        with pytest.raises(ValidationError, match="must be odd"):
            AdaptiveFilterConfig(kernel_size=4)

        with pytest.raises(ValidationError, match="must be odd"):
            AdaptiveFilterConfig(kernel_size=(3, 4))


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
                data_max=50.0,  # Invalid: max < min
                data_mean=75.0,
                mode=2,
                image_shape=(1024, 1024),
            )


if __name__ == "__main__":
    pytest.main([__file__])
