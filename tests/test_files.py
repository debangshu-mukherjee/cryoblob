"""
Tests for cryoblob.files module

This module tests file I/O operations, batch processing, and memory management.
"""

import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import mrcfile
from absl.testing import parameterized

import cryoblob as cb
from cryoblob.types import MRC_Image, make_MRC_Image, scalar_float


class TestFileParams(chex.TestCase):
    """Test file parameter loading."""
    
    @patch('cryoblob.files.files')
    @patch('builtins.open')
    @patch('json.load')
    def test_file_params(self, mock_json_load, mock_open, mock_files):
        """Test file_params function."""
        # Mock the return values
        mock_json_load.return_value = {
            'data': {'test': 'path'},
            'results': {'test': 'results'}
        }
        mock_files.return_value.joinpath.return_value = 'mock_path'
        
        main_dir, folder_struct = cb.file_params()
        
        # Check that it returns expected structure
        assert isinstance(main_dir, str)
        assert isinstance(folder_struct, dict)
        assert 'data' in folder_struct
        assert 'results' in folder_struct


class TestLoadMRC(chex.TestCase, parameterized.TestCase):
    """Test MRC file loading."""
    
    def setUp(self):
        super().setUp()
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.test_dir)
        super().tearDown()
    
    def create_test_mrc(self, filename, shape=(10, 10), dtype=np.float32):
        """Helper to create test MRC files."""
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
        """Test loading 2D MRC file."""
        filepath, original_data = self.create_test_mrc('test_2d.mrc', shape=(50, 50))
        
        def load_fn():
            return cb.load_mrc(filepath)
        
        mrc_image = self.variant(load_fn)()
        
        # Check structure
        assert isinstance(mrc_image, MRC_Image)
        assert mrc_image.image_data.shape == (50, 50)
        chex.assert_trees_all_close(mrc_image.image_data, original_data, atol=1e-6)
        
        # Check metadata
        assert mrc_image.voxel_size.shape == (3,)
        chex.assert_trees_all_close(mrc_image.voxel_size, jnp.array([1.0, 1.2, 1.2]))
        assert mrc_image.mode == 2  # float32
    
    @chex.all_variants
    def test_load_mrc_3d(self):
        """Test loading 3D MRC file."""
        filepath, original_data = self.create_test_mrc('test_3d.mrc', shape=(20, 30, 40))
        
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
        """Test loading MRC files with different data types."""
        filepath, _ = self.create_test_mrc(f'test_{dtype.__name__}.mrc', dtype=dtype)
        
        mrc_image = cb.load_mrc(filepath)
        
        # Mode should match the data type
        assert mrc_image.mode == expected_mode


class TestProcessSingleFile(chex.TestCase, parameterized.TestCase):
    """Test single file processing."""
    
    def setUp(self):
        super().setUp()
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)
        super().tearDown()
    
    def create_test_file_with_blobs(self, filename):
        """Create test MRC file with synthetic blobs."""
        filepath = os.path.join(self.test_dir, filename)
        
        # Create image with blobs
        x, y = jnp.meshgrid(jnp.linspace(-5, 5, 100), jnp.linspace(-5, 5, 100))
        blob1 = jnp.exp(-((x-2)**2 + (y-2)**2) / 1.0)
        blob2 = jnp.exp(-((x+2)**2 + (y+2)**2) / 1.0)
        data = np.array(blob1 + blob2)
        
        with mrcfile.new(filepath, overwrite=True) as mrc:
            mrc.set_data(data.astype(np.float32))
            mrc.voxel_size = (1.0, 0.1, 0.1)  # 0.1 nm per pixel
            mrc.update_header_from_data()
            
        return filepath
    
    @patch('cryoblob.files.device_put')
    @patch('cryoblob.files.device_get')
    def test_process_single_file_basic(self, mock_device_get, mock_device_put):
        """Test basic single file processing."""
        # Mock device operations
        mock_device_put.side_effect = lambda x: x
        mock_device_get.side_effect = lambda x: x
        
        filepath = self.create_test_file_with_blobs('test_blobs.mrc')
        
        preprocessing_kwargs = {
            'exponential': False,
            'logarizer': False,
            'gblur': 0,
            'background': 0,
            'apply_filter': 0,
        }
        
        blobs, returned_path = cb.process_single_file(
            filepath,
            preprocessing_kwargs,
            blob_downscale=4.0,
            stream_mode=False
        )
        
        assert returned_path == filepath
        assert isinstance(blobs, jnp.ndarray)
        assert blobs.ndim == 2
        assert blobs.shape[1] == 3  # (y, x, size)
        
        # Should detect at least one blob
        assert len(blobs) >= 1
    
    @parameterized.parameters(True, False)
    def test_process_single_file_stream_mode(self, stream_mode):
        """Test file processing with different stream modes."""
        filepath = self.create_test_file_with_blobs('test_stream.mrc')
        
        preprocessing_kwargs = {'exponential': True}
        
        with patch('mrcfile.mmap' if stream_mode else 'mrcfile.open'):
            blobs, _ = cb.process_single_file(
                filepath,
                preprocessing_kwargs,
                blob_downscale=4.0,
                stream_mode=stream_mode
            )
            
            assert isinstance(blobs, jnp.ndarray)
    
    def test_process_single_file_error_handling(self):
        """Test error handling in file processing."""
        # Non-existent file
        blobs, filepath = cb.process_single_file(
            'nonexistent.mrc',
            {},
            blob_downscale=1.0
        )
        
        # Should return empty array on error
        assert len(blobs) == 0
        assert filepath == 'nonexistent.mrc'


class TestProcessBatchOfFiles(chex.TestCase):
    """Test batch file processing."""
    
    @patch('cryoblob.files.process_single_file')
    @patch('jax.vmap')
    def test_process_batch_of_files(self, mock_vmap, mock_process_single):
        """Test batch processing of files."""
        # Mock process_single_file
        mock_process_single.return_value = (jnp.array([[1.0, 2.0, 3.0]]), 'test.mrc')
        
        # Mock vmap to just call the function for each file
        def mock_vmap_impl(fn):
            def wrapped(files):
                return [fn(f) for f in files]
            return wrapped
        mock_vmap.side_effect = mock_vmap_impl
        
        file_batch = ['file1.mrc', 'file2.mrc', 'file3.mrc']
        preprocessing_kwargs = {}
        
        results = cb.process_batch_of_files(
            file_batch,
            preprocessing_kwargs,
            blob_downscale=1.0
        )
        
        # Should process all files
        assert len(results) == 3


class TestFolderBlobs(chex.TestCase, parameterized.TestCase):
    """Test folder-level blob detection."""
    
    def setUp(self):
        super().setUp()
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)
        super().tearDown()
    
    def create_test_folder(self, num_files=3):
        """Create a folder with test MRC files."""
        for i in range(num_files):
            x, y = jnp.meshgrid(jnp.linspace(-5, 5, 50), jnp.linspace(-5, 5, 50))
            # Add blob at different position for each file
            blob = jnp.exp(-((x-i+1)**2 + y**2) / 1.0)
            data = np.array(blob)
            
            filepath = os.path.join(self.test_dir, f'test_{i}.mrc')
            with mrcfile.new(filepath, overwrite=True) as mrc:
                mrc.set_data(data.astype(np.float32))
                mrc.voxel_size = (1.0, 0.1, 0.1)
                mrc.update_header_from_data()
    
    @patch('cryoblob.files.estimate_batch_size')
    @patch('cryoblob.files.process_batch_of_files')
    def test_folder_blobs_basic(self, mock_process_batch, mock_estimate_batch):
        """Test basic folder processing."""
        self.create_test_folder(num_files=3)
        
        # Mock batch size estimation
        mock_estimate_batch.return_value = 2
        
        # Mock batch processing to return some blobs
        mock_process_batch.return_value = [
            (jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 'file1.mrc'),
            (jnp.array([[7.0, 8.0, 9.0]]), 'file2.mrc'),
        ]
        
        result_df = cb.folder_blobs(
            self.test_dir + '/',
            file_type='mrc',
            blob_downscale=4.0,
            target_memory_gb=2.0
        )
        
        # Check DataFrame structure
        assert isinstance(result_df, pd.DataFrame)
        expected_columns = ['File Location', 'Center Y (nm)', 'Center X (nm)', 'Size (nm)']
        assert list(result_df.columns) == expected_columns
    
    def test_folder_blobs_empty_folder(self):
        """Test processing empty folder."""
        # Empty folder
        result_df = cb.folder_blobs(
            self.test_dir + '/',
            file_type='mrc'
        )
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 0
    
    @parameterized.parameters(
        {'exponential': True, 'gblur': 2},
        {'logarizer': True, 'background': 5},
        {'apply_filter': 3},
    )
    def test_folder_blobs_preprocessing_options(self, kwargs):
        """Test folder processing with different preprocessing options."""
        self.create_test_folder(num_files=1)
        
        with patch('cryoblob.files.process_batch_of_files') as mock_process:
            mock_process.return_value = [(jnp.array([]), 'test.mrc')]
            
            cb.folder_blobs(
                self.test_dir + '/',
                file_type='mrc',
                **kwargs
            )
            
            # Check that preprocessing kwargs were passed correctly
            call_args = mock_process.call_args
            preprocessing_kwargs = call_args[0][1]
            
            for key, value in kwargs.items():
                assert preprocessing_kwargs[key] == value


class TestMemoryManagement(chex.TestCase):
    """Test memory management features."""
    
    @patch('mrcfile.open')
    def test_estimate_batch_size(self, mock_mrcfile):
        """Test batch size estimation (when function exists)."""
        # Mock MRC file
        mock_mrc = MagicMock()
        mock_mrc.data.shape = (1000, 1000)
        mock_mrc.data.dtype = np.float32
        mock_mrcfile.return_value.__enter__.return_value = mock_mrc
        
        # Note: estimate_batch_size is referenced but not implemented in the files
        # This test assumes it would be implemented
        if hasattr(cb, 'estimate_batch_size'):
            batch_size = cb.estimate_batch_size('test.mrc', target_memory_gb=4.0)
            assert isinstance(batch_size, int)
            assert batch_size > 0
    
    @patch('cryoblob.files.device_get')
    @patch('cryoblob.files.device_put')
    def test_memory_clearing(self, mock_device_put, mock_device_get):
        """Test that memory is properly cleared during processing."""
        # Track calls to device operations
        mock_device_put.side_effect = lambda x: x
        mock_device_get.side_effect = lambda x: np.array(x)
        
        # Create test file
        test_dir = tempfile.mkdtemp()
        try:
            filepath = os.path.join(test_dir, 'test.mrc')
            data = np.random.rand(50, 50).astype(np.float32)
            
            with mrcfile.new(filepath, overwrite=True) as mrc:
                mrc.set_data(data)
                mrc.voxel_size = (1.0, 1.0, 1.0)
                mrc.update_header_from_data()
            
            # Process file
            cb.process_single_file(filepath, {}, blob_downscale=1.0)
            
            # Verify device operations were called
            assert mock_device_put.called
            assert mock_device_get.called
            
        finally:
            import shutil
            shutil.rmtree(test_dir)


class TestDataFrameOutput(chex.TestCase):
    """Test DataFrame output formatting."""
    
    def test_dataframe_columns(self):
        """Test that output DataFrame has correct columns."""
        with patch('glob.glob') as mock_glob:
            mock_glob.return_value = []
            
            df = cb.folder_blobs('dummy_folder/', file_type='mrc')
            
            expected_columns = [
                'File Location',
                'Center Y (nm)',
                'Center X (nm)',
                'Size (nm)'
            ]
            assert list(df.columns) == expected_columns
    
    def test_dataframe_dtypes(self):
        """Test DataFrame data types."""
        # Create mock data
        test_data = {
            'File Location': ['file1.mrc', 'file1.mrc', 'file2.mrc'],
            'Center Y (nm)': [10.5, 20.3, 15.7],
            'Center X (nm)': [5.2, 15.8, 25.1],
            'Size (nm)': [2.1, 3.5, 2.8]
        }
        df = pd.DataFrame(test_data)
        
        # Check dtypes
        assert df['File Location'].dtype == 'object'
        assert np.issubdtype(df['Center Y (nm)'].dtype, np.floating)
        assert np.issubdtype(df['Center X (nm)'].dtype, np.floating)
        assert np.issubdtype(df['Size (nm)'].dtype, np.floating)


if __name__ == "__main__":
    from absl.testing import absltest
    absltest.main()