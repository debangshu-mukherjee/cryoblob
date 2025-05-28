import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import chex
import jax.numpy as jnp
import mrcfile
import numpy as np
import pandas as pd
from absl.testing import parameterized

import cryoblob as cb
from cryoblob.types import MRC_Image, make_MRC_Image, scalar_float
from cryoblob.valid import PreprocessingConfig


class TestFileParams(chex.TestCase):

    @patch("cryoblob.files.files")
    @patch("builtins.open")
    @patch("json.load")
    def test_file_params(self, mock_json_load, mock_open, mock_files):
        mock_json_load.return_value = {
            "data": {"test": "path"},
            "results": {"test": "results"},
        }
        mock_files.return_value.joinpath.return_value = "mock_path"

        main_dir, folder_struct = cb.file_params()

        assert isinstance(main_dir, str)
        assert isinstance(folder_struct, dict)
        assert "data" in folder_struct
        assert "results" in folder_struct


class TestLoadMRC(chex.TestCase, parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.test_dir)
        super().tearDown()

    def create_test_mrc(self, filename, shape=(10, 10), dtype=np.float32):
        filepath = os.path.join(self.test_dir, filename)

        with mrcfile.new(filepath, overwrite=True) as mrc:
            data = np.random.rand(*shape).astype(dtype)
            mrc.set_data(data)
            mrc.voxel_size = (1.0, 1.2, 1.2)
            mrc.header.origin = (0.0, 0.0, 0.0)
            mrc.update_header_from_data()

        return filepath, data

    @chex.all_variants
    def test_load_mrc_2d(self):
        filepath, original_data = self.create_test_mrc("test_2d.mrc", shape=(50, 50))

        def load_fn():
            return cb.load_mrc(filepath)

        mrc_image = self.variant(load_fn)()

        assert isinstance(mrc_image, MRC_Image)
        assert mrc_image.image_data.shape == (50, 50)
        chex.assert_trees_all_close(mrc_image.image_data, original_data, atol=1e-6)

        assert mrc_image.voxel_size.shape == (3,)
        chex.assert_trees_all_close(mrc_image.voxel_size, jnp.array([1.0, 1.2, 1.2]))
        assert mrc_image.mode == 2

    @chex.all_variants
    def test_load_mrc_3d(self):
        filepath, original_data = self.create_test_mrc(
            "test_3d.mrc", shape=(20, 30, 40)
        )

        def load_fn():
            return cb.load_mrc(filepath)

        mrc_image = self.variant(load_fn)()

        assert mrc_image.image_data.shape == (20, 30, 40)
        assert isinstance(mrc_image.image_data, jnp.ndarray)

    @parameterized.parameters(
        (np.float32, 2),
        (np.int16, 1),
        (np.uint8, 0),
    )
    def test_load_mrc_dtypes(self, dtype, expected_mode):
        filepath, _ = self.create_test_mrc(f"test_{dtype.__name__}.mrc", dtype=dtype)

        mrc_image = cb.load_mrc(filepath)

        assert mrc_image.mode == expected_mode


class TestProcessSingleFile(chex.TestCase, parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.test_dir)
        super().tearDown()

    def create_test_file_with_blobs(self, filename):
        filepath = os.path.join(self.test_dir, filename)

        x, y = jnp.meshgrid(jnp.linspace(-5, 5, 100), jnp.linspace(-5, 5, 100))
        blob1 = jnp.exp(-((x - 2) ** 2 + (y - 2) ** 2) / 1.0)
        blob2 = jnp.exp(-((x + 2) ** 2 + (y + 2) ** 2) / 1.0)
        data = np.array(blob1 + blob2)

        with mrcfile.new(filepath, overwrite=True) as mrc:
            mrc.set_data(data.astype(np.float32))
            mrc.voxel_size = (1.0, 0.1, 0.1)
            mrc.update_header_from_data()

        return filepath

    @patch("cryoblob.files.device_put")
    @patch("cryoblob.files.device_get")
    def test_process_single_file_basic(self, mock_device_get, mock_device_put):
        mock_device_put.side_effect = lambda x: x
        mock_device_get.side_effect = lambda x: x

        filepath = self.create_test_file_with_blobs("test_blobs.mrc")

        preprocessing_config = PreprocessingConfig(
            exponential=False,
            logarizer=False,
            gblur=0,
            background=0,
            apply_filter=0,
        )

        blobs, returned_path = cb.process_single_file(
            filepath, preprocessing_config, blob_downscale=4.0, stream_mode=False
        )

        assert returned_path == filepath
        assert isinstance(blobs, jnp.ndarray)
        assert blobs.ndim == 2
        assert blobs.shape[1] == 3

        assert len(blobs) >= 1

    @parameterized.parameters(True, False)
    def test_process_single_file_stream_mode(self, stream_mode):
        filepath = self.create_test_file_with_blobs("test_stream.mrc")

        preprocessing_config = PreprocessingConfig(exponential=True)

        with patch("mrcfile.mmap" if stream_mode else "mrcfile.open"):
            blobs, _ = cb.process_single_file(
                filepath,
                preprocessing_config,
                blob_downscale=4.0,
                stream_mode=stream_mode,
            )

            assert isinstance(blobs, jnp.ndarray)

    def test_process_single_file_error_handling(self):
        preprocessing_config = PreprocessingConfig()
        
        blobs, filepath = cb.process_single_file(
            "nonexistent.mrc", preprocessing_config, blob_downscale=1.0
        )

        assert len(blobs) == 0
        assert filepath == "nonexistent.mrc"

    def test_process_single_file_different_configs(self):
        filepath = self.create_test_file_with_blobs("test_config.mrc")
        
        configs = [
            PreprocessingConfig(exponential=True, gblur=2),
            PreprocessingConfig(logarizer=True, background=5),
            PreprocessingConfig(apply_filter=3),
        ]
        
        for config in configs:
            blobs, returned_path = cb.process_single_file(
                filepath, config, blob_downscale=4.0
            )
            assert isinstance(blobs, jnp.ndarray)
            assert returned_path == filepath


class TestProcessBatchOfFiles(chex.TestCase):

    @patch("cryoblob.files.process_single_file")
    @patch("jax.vmap")
    def test_process_batch_of_files(self, mock_vmap, mock_process_single):
        mock_process_single.return_value = (jnp.array([[1.0, 2.0, 3.0]]), "test.mrc")

        def mock_vmap_impl(fn):
            def wrapped(files):
                return [fn(f) for f in files]
            return wrapped

        mock_vmap.side_effect = mock_vmap_impl

        file_batch = ["file1.mrc", "file2.mrc", "file3.mrc"]
        preprocessing_config = PreprocessingConfig()

        results = cb.process_batch_of_files(
            file_batch, preprocessing_config, blob_downscale=1.0
        )

        assert len(results) == 3

    def test_process_batch_of_files_with_config(self):
        preprocessing_config = PreprocessingConfig(
            exponential=True,
            gblur=2,
            background=5
        )
        
        with patch("cryoblob.files.process_single_file") as mock_process:
            mock_process.return_value = (jnp.array([[1.0, 2.0, 3.0]]), "test.mrc")
            
            with patch("jax.vmap") as mock_vmap:
                mock_vmap.return_value = lambda x: [(jnp.array([[1.0, 2.0, 3.0]]), "test.mrc")]
                
                results = cb.process_batch_of_files(
                    ["test.mrc"], preprocessing_config, 1.0
                )
                
                assert len(results) == 1


class TestFolderBlobs(chex.TestCase, parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.test_dir)
        super().tearDown()

    def create_test_folder(self, num_files=3):
        for i in range(num_files):
            x, y = jnp.meshgrid(jnp.linspace(-5, 5, 50), jnp.linspace(-5, 5, 50))
            blob = jnp.exp(-((x - i + 1) ** 2 + y**2) / 1.0)
            data = np.array(blob)

            filepath = os.path.join(self.test_dir, f"test_{i}.mrc")
            with mrcfile.new(filepath, overwrite=True) as mrc:
                mrc.set_data(data.astype(np.float32))
                mrc.voxel_size = (1.0, 0.1, 0.1)
                mrc.update_header_from_data()

    @patch("cryoblob.files.estimate_batch_size")
    @patch("cryoblob.files.process_batch_of_files")
    def test_folder_blobs_basic(self, mock_process_batch, mock_estimate_batch):
        self.create_test_folder(num_files=3)

        mock_estimate_batch.return_value = 2

        mock_process_batch.return_value = [
            (jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), "file1.mrc"),
            (jnp.array([[7.0, 8.0, 9.0]]), "file2.mrc"),
        ]

        result_df = cb.folder_blobs(
            self.test_dir + "/",
            file_type="mrc",
            blob_downscale=4.0,
            target_memory_gb=2.0,
        )

        assert isinstance(result_df, pd.DataFrame)
        expected_columns = [
            "File Location",
            "Center Y (nm)",
            "Center X (nm)",
            "Size (nm)",
        ]
        assert list(result_df.columns) == expected_columns

    def test_folder_blobs_empty_folder(self):
        result_df = cb.folder_blobs(self.test_dir + "/", file_type="mrc")

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 0

    @parameterized.parameters(
        {"exponential": True, "gblur": 2},
        {"logarizer": True, "background": 5},
        {"apply_filter": 3},
    )
    def test_folder_blobs_preprocessing_options(self, kwargs):
        self.create_test_folder(num_files=1)

        with patch("cryoblob.files.process_batch_of_files") as mock_process:
            mock_process.return_value = [(jnp.array([]), "test.mrc")]

            cb.folder_blobs(self.test_dir + "/", file_type="mrc", **kwargs)

            call_args = mock_process.call_args
            preprocessing_config = call_args[0][1]

            assert isinstance(preprocessing_config, PreprocessingConfig)
            for key, value in kwargs.items():
                assert getattr(preprocessing_config, key) == value

    def test_folder_blobs_invalid_preprocessing_params(self):
        with self.assertRaises(ValueError) as context:
            cb.folder_blobs(
                self.test_dir + "/",
                exponential=True,
                logarizer=True,
            )
        
        assert "Invalid preprocessing parameters" in str(context.exception)

    def test_folder_blobs_preprocessing_config_validation(self):
        self.create_test_folder(num_files=1)
        
        with patch("cryoblob.files.process_batch_of_files") as mock_process:
            mock_process.return_value = [(jnp.array([]), "test.mrc")]
            
            result_df = cb.folder_blobs(
                self.test_dir + "/",
                exponential=True,
                gblur=5,
                background=10
            )
            
            call_args = mock_process.call_args
            preprocessing_config = call_args[0][1]
            
            assert isinstance(preprocessing_config, PreprocessingConfig)
            assert preprocessing_config.exponential == True
            assert preprocessing_config.gblur == 5
            assert preprocessing_config.background == 10


class TestMemoryManagement(chex.TestCase):

    @patch("mrcfile.open")
    def test_estimate_batch_size(self, mock_mrcfile):
        mock_mrc = MagicMock()
        mock_mrc.data.shape = (1000, 1000)
        mock_mrc.data.dtype = np.float32
        mock_mrcfile.return_value.__enter__.return_value = mock_mrc

        if hasattr(cb, "estimate_batch_size"):
            batch_size = cb.estimate_batch_size("test.mrc", target_memory_gb=4.0)
            assert isinstance(batch_size, int)
            assert batch_size > 0

    @patch("cryoblob.files.device_get")
    @patch("cryoblob.files.device_put")
    def test_memory_clearing(self, mock_device_put, mock_device_get):
        mock_device_put.side_effect = lambda x: x
        mock_device_get.side_effect = lambda x: np.array(x)

        test_dir = tempfile.mkdtemp()
        try:
            filepath = os.path.join(test_dir, "test.mrc")
            data = np.random.rand(50, 50).astype(np.float32)

            with mrcfile.new(filepath, overwrite=True) as mrc:
                mrc.set_data(data)
                mrc.voxel_size = (1.0, 1.0, 1.0)
                mrc.update_header_from_data()

            preprocessing_config = PreprocessingConfig()
            cb.process_single_file(filepath, preprocessing_config, blob_downscale=1.0)

            assert mock_device_put.called
            assert mock_device_get.called

        finally:
            import shutil
            shutil.rmtree(test_dir)


class TestDataFrameOutput(chex.TestCase):

    def test_dataframe_columns(self):
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = []

            df = cb.folder_blobs("dummy_folder/", file_type="mrc")

            expected_columns = [
                "File Location",
                "Center Y (nm)",
                "Center X (nm)",
                "Size (nm)",
            ]
            assert list(df.columns) == expected_columns

    def test_dataframe_dtypes(self):
        test_data = {
            "File Location": ["file1.mrc", "file1.mrc", "file2.mrc"],
            "Center Y (nm)": [10.5, 20.3, 15.7],
            "Center X (nm)": [5.2, 15.8, 25.1],
            "Size (nm)": [2.1, 3.5, 2.8],
        }
        df = pd.DataFrame(test_data)

        assert df["File Location"].dtype == "object"
        assert np.issubdtype(df["Center Y (nm)"].dtype, np.floating)
        assert np.issubdtype(df["Center X (nm)"].dtype, np.floating)
        assert np.issubdtype(df["Size (nm)"].dtype, np.floating)


class TestPreprocessingConfigIntegration(chex.TestCase):
    """Test integration between preprocessing config and file processing functions."""

    def test_preprocessing_config_creation_from_kwargs(self):
        kwargs = {
            "exponential": True,
            "gblur": 3,
            "background": 7,
            "apply_filter": 5
        }
        
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = []
            
            df = cb.folder_blobs("dummy/", **kwargs)
            assert isinstance(df, pd.DataFrame)

    def test_preprocessing_config_validation_in_folder_blobs(self):
        with self.assertRaises(ValueError):
            cb.folder_blobs(
                "dummy/",
                exponential=True,
                logarizer=True,
            )

    def test_preprocessing_config_defaults(self):
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = []
            
            with patch("cryoblob.files.process_batch_of_files") as mock_process:
                mock_process.return_value = []
                
                cb.folder_blobs("dummy/")
                
                if mock_process.called:
                    call_args = mock_process.call_args
                    config = call_args[0][1]
                    
                    assert config.exponential == False
                    assert config.logarizer == False
                    assert config.gblur == 0
                    assert config.background == 0
                    assert config.apply_filter == 0


if __name__ == "__main__":
    from absl.testing import absltest

    absltest.main()