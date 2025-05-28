"""
Module: files
---------------------------

Contains the codes for interfacing with data files.
One goal here is to separate the Python code from
the JAX code. Thus most of the necessary outward
facing code, which is necessarily in Python, is here.

Functions
---------
- `file_params`:
    Get the parameters for the file organization.
- `load_mrc`:
    Reads an MRC-format cryo-EM file, extracting image data and metadata.
- `process_single_file`:
    Process a single file for blob detection with memory optimization.
- `process_batch_of_files`:
    Process a batch of files in parallel with memory optimization.
- `folder_blobs`:
    Process a folder of images for blob detection with memory optimization.
- `estimate_batch_size`:
    Estimate optimal batch size for processing MRC files based on available memory.
- `estimate_memory_usage`:
    Estimate memory usage in GB for processing a single MRC file.
- `get_optimal_batch_size`:
    Get optimal batch size by sampling multiple files from the list.
"""

import glob
import json
import os
from importlib.resources import files

import jax
import jax.numpy as jnp
import mrcfile
import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Dict, List, Literal, Optional, Tuple
from jax import device_get, device_put, vmap
from jaxtyping import Array, Float, jaxtyped
from tqdm.auto import tqdm

from cryoblob.blobs import blob_list_log
from cryoblob.types import (MRC_Image, make_MRC_Image, scalar_float,
                            scalar_int, scalar_num)

jax.config.update("jax_enable_x64", True)


def file_params() -> Tuple[str, dict]:
    """
    Description
    -----------
    Run this at the beginning to generate the dict
    This gives both the absolute and relative paths
    on how the files are organized.

    Returns
    -------
    - `main_directory` (str):
        the main directory where the package is located.
    - `folder_structure` (dict):
        where the files and data are stored, as read
        from the organization.json file.
    """
    pkg_directory: str = os.path.dirname(__file__)
    listring: List = pkg_directory.split("/")[1:-2]
    listring.append("")
    listring.insert(0, "")
    main_directory: str = "/".join(listring)
    folder_structure: dict = json.load(
        open(files("cryoblob.params").joinpath("organization.json"))
    )
    return (main_directory, folder_structure)


def load_mrc(filepath: str) -> MRC_Image:
    """
    Description
    -----------
    Reads an MRC-format cryo-EM file from the specified path, extracting
    image data and relevant metadata. All numeric data are converted into
    JAX arrays and wrapped into a structured `MRC_Image` PyTree, compatible
    with JAX's functional programming paradigm.

    Parameters
    ----------
    - `filepath` (str):
        Path to the MRC file to be loaded.

    Returns
    -------
    `MRC_Image` (A PyTree containing):
        - `image_data`: Image array (2D or 3D).
        - `voxel_size`: Array containing voxel dimensions in
            Å (Z, Y, X).
        - `origin`: Array indicating the origin coordinates from the
            header (Z, Y, X).
        - `data_min`: Minimum pixel value.
        - `data_max`: Maximum pixel value.
        - `data_mean`: Mean pixel value.
        - `mode`: Integer code representing data type
            (e.g., 0=int8, 1=int16, 2=float32).

    Examples
    --------
    >>> mrc_image = load_mrc("example.mrc")
    >>> print(mrc_image.voxel_size)
    Array([1.2, 1.2, 1.2], dtype=float32)

    Notes
    -----
    - This function uses the `mrcfile` library for parsing MRC files.
    - The resulting PyTree structure (`MRC_Image`) is explicitly
        designed for use in JAX-based image processing pipelines.

    """
    with mrcfile.open(filepath, permissive=True) as mrc:
        data = jnp.array(mrc.data)
        voxel_size = jnp.array(
            [
                float(mrc.voxel_size.z),
                float(mrc.voxel_size.y),
                float(mrc.voxel_size.x),
            ]
        )
        origin = jnp.array(
            [
                float(mrc.header.origin.z),
                float(mrc.header.origin.y),
                float(mrc.header.origin.x),
            ]
        )
        data_min = jnp.array(mrc.header.dmin)
        data_max = jnp.array(mrc.header.dmax)
        data_mean = jnp.array(mrc.header.dmean)
        mode = jnp.array(mrc.header.mode)
    MRC_data: MRC_Image = make_MRC_Image(
        image_data=data,
        voxel_size=voxel_size,
        origin=origin,
        data_min=data_min,
        data_max=data_max,
        data_mean=data_mean,
        mode=mode,
    )
    return MRC_data


def process_single_file(
    file_path: str,
    preprocessing_kwargs: Dict,
    blob_downscale: float,
    stream_mode: bool = True,
) -> Tuple[Float[Array, "n 3"], str]:
    """
    Process a single file for blob detection with memory optimization.

    Parameters
    ----------
    - `file_path` (str):
        Path to the image file
    - `preprocessing_kwargs` (Dict):
        Preprocessing parameters
    - `blob_downscale` (float):
        Downscaling factor for blob detection
    - `stream_mode` (bool):
        Whether to use streaming for large files

    Returns
    -------
    - `scaled_blobs` (Float[Array, "n 3"]):
        Array of blob coordinates and sizes
    - `file_path` (str):
        Original file path

    Notes
    -----
    Uses streaming mode for large files to reduce memory usage.
    Immediately releases file handles after reading.
    """
    try:
        if stream_mode:
            with mrcfile.mmap(file_path, mode="r") as data_mrc:
                x_calib = jnp.asarray(data_mrc.voxel_size.x)
                y_calib = jnp.asarray(data_mrc.voxel_size.y)
                im_data = jnp.asarray(data_mrc.data, dtype=jnp.float64)
        else:
            with mrcfile.open(file_path) as data_mrc:
                x_calib = jnp.asarray(data_mrc.voxel_size.x)
                y_calib = jnp.asarray(data_mrc.voxel_size.y)
                im_data = jnp.asarray(data_mrc.data, dtype=jnp.float64)

        im_data = device_put(im_data)
        preprocessed_imdata = preprocessing(
            image_orig=im_data, return_params=False, **preprocessing_kwargs
        )
        del im_data
        blob_list = blob_list_log(preprocessed_imdata, downscale=blob_downscale)
        del preprocessed_imdata
        scaled_blobs = jnp.concatenate(
            [
                (blob_list[:, 0] * y_calib)[:, None],
                (blob_list[:, 1] * x_calib)[:, None],
                (blob_list[:, 2] * ((y_calib**2 + x_calib**2) ** 0.5))[:, None],
            ],
            axis=1,
        )
        return scaled_blobs, file_path

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return jnp.array([]), file_path


def process_batch_of_files(
    file_batch: List[str], preprocessing_kwargs: Dict, blob_downscale: float
) -> List[Tuple[Float[Array, "n 3"], str]]:
    """
    Process a batch of files in parallel with memory optimization.

    Parameters
    ----------
    - `file_batch` (List[str]):
        List of file paths to process
    - `preprocessing_kwargs` (Dict):
        Preprocessing parameters
    - `blob_downscale` (float):
        Downscaling factor

    Returns
    -------
    - `results` (List[Tuple[Array, str]]):
        List of (blobs, file_path) tuples
    """
    batch_process_fn = vmap(
        lambda x: process_single_file(x, preprocessing_kwargs, blob_downscale)
    )
    return batch_process_fn(jnp.array(file_batch))


def folder_blobs(
    folder_location: str,
    file_type: Literal["mrc"] | None = "mrc",
    blob_downscale: float | None = 7,
    target_memory_gb: float = 4.0,
    stream_large_files: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Process a folder of images for blob detection with memory optimization.

    Parameters
    ----------
    - `folder_location` (str):
        Path to folder containing images
    - `file_type` (str, optional):
        File type to process. Default is "mrc"
    - `blob_downscale` (float, optional):
        Downscaling factor. Default is 7
    - `target_memory_gb` (float, optional):
        Target GPU memory usage in GB. Default is 4.0
    - `stream_large_files` (bool, optional):
        Whether to use streaming for large files. Default is True
    - `**kwargs`:
        Additional preprocessing parameters

    Returns
    -------
    - `blob_dataframe` (pd.DataFrame):
        DataFrame containing blob information

    Memory Management
    ----------------
    - Uses batch processing to control memory usage
    - Automatically adjusts batch size based on available memory
    - Clears device memory between batches
    - Streams large files if needed
    - Efficiently handles intermediate results
    """
    default_kwargs = {
        "exponential": False,
        "logarizer": False,
        "gblur": 0,
        "background": 0,
        "apply_filter": 0,
    }
    preprocessing_kwargs = {**default_kwargs, **kwargs}
    file_list = glob.glob(folder_location + "*." + file_type)
    if not file_list:
        return pd.DataFrame(
            columns=["File Location", "Center Y (nm)", "Center X (nm)", "Size (nm)"]
        )
    batch_size = estimate_batch_size(file_list[0], target_memory_gb)
    all_blobs = []
    all_files = []
    with tqdm(total=len(file_list), desc="Processing files") as pbar:
        for i in range(0, len(file_list), batch_size):
            batch_files = file_list[i : i + batch_size]
            batch_results = process_batch_of_files(
                batch_files, preprocessing_kwargs, blob_downscale
            )
            for blobs, file_path in batch_results:
                if len(blobs) > 0:
                    cpu_blobs = device_get(blobs)
                    all_blobs.append(cpu_blobs)
                    all_files.extend([file_path] * len(cpu_blobs))
            pbar.update(len(batch_files))
    if all_blobs:
        combined_blobs = np.concatenate(all_blobs, axis=0)
        blob_dataframe = pd.DataFrame(
            data=np.column_stack((all_files, combined_blobs)),
            columns=["File Location", "Center Y (nm)", "Center X (nm)", "Size (nm)"],
        )
    else:
        blob_dataframe = pd.DataFrame(
            columns=["File Location", "Center Y (nm)", "Center X (nm)", "Size (nm)"]
        )

    return blob_dataframe


@jaxtyped(typechecker=beartype)
def estimate_batch_size(
    sample_file_path: str,
    target_memory_gb: Optional[scalar_float] = 4.0,
    safety_factor: Optional[scalar_float] = 0.7,
    processing_overhead: Optional[scalar_float] = 3.0,
) -> scalar_int:
    """
    Description
    -----------
    Estimate optimal batch size for processing MRC files based on available memory
    and file characteristics. This function analyzes a sample file to estimate
    memory requirements and calculates the maximum number of files that can be
    processed simultaneously without exceeding memory limits.

    Parameters
    ----------
    - `sample_file_path` (str):
        Path to a representative MRC file for size estimation
    - `target_memory_gb` (scalar_float, optional):
        Target GPU memory usage in GB. Default is 4.0
    - `safety_factor` (scalar_float, optional):
        Safety factor to prevent memory overflow (0.0-1.0).
        Default is 0.7 (use 70% of target memory)
    - `processing_overhead` (scalar_float, optional):
        Memory overhead multiplier for processing operations.
        Default is 3.0 (processing uses 3x the raw data size)

    Returns
    -------
    - `batch_size` (scalar_int):
        Recommended batch size for processing

    Notes
    -----
    The estimation considers:
    - Raw file size in memory (dtype conversion)
    - Preprocessing operations (filtering, transformations)
    - Blob detection memory requirements
    - JAX compilation overhead
    - Intermediate array storage

    Memory estimation formula:
    ```
    per_file_memory = file_size * processing_overhead
    available_memory = target_memory_gb * safety_factor * 1e9
    batch_size = max(1, available_memory // per_file_memory)
    ```

    Examples
    --------
    >>> batch_size = estimate_batch_size("sample.mrc", target_memory_gb=8.0)
    >>> print(f"Recommended batch size: {batch_size}")
    """
    try:
        file_size_bytes: scalar_float = float(os.path.getsize(sample_file_path))

        with mrcfile.open(sample_file_path, mode="r", permissive=True) as mrc:
            data_shape: tuple = mrc.data.shape
            data_dtype: str = str(mrc.data.dtype)

            dtype_bytes: scalar_int
            if "float64" in data_dtype:
                dtype_bytes = 8
            elif "float32" in data_dtype:
                dtype_bytes = 4
            elif "int32" in data_dtype:
                dtype_bytes = 4
            elif "int16" in data_dtype:
                dtype_bytes = 2
            elif "int8" in data_dtype or "uint8" in data_dtype:
                dtype_bytes = 1
            else:
                dtype_bytes = 4

            array_elements: scalar_int = int(jnp.prod(jnp.array(data_shape)))
            base_memory_bytes: scalar_float = float(array_elements * dtype_bytes)

            jax_memory_bytes: scalar_float = float(array_elements * 8)

            per_file_memory: scalar_float = jax_memory_bytes * processing_overhead

            target_memory_bytes: scalar_float = target_memory_gb * 1e9
            available_memory: scalar_float = target_memory_bytes * safety_factor

            estimated_batch_size: scalar_float = available_memory / per_file_memory
            batch_size: scalar_int = max(1, int(jnp.floor(estimated_batch_size)))

            min_batch_size: scalar_int = 1
            max_batch_size: scalar_int = 50

            final_batch_size: scalar_int = max(
                min_batch_size, min(batch_size, max_batch_size)
            )

            return final_batch_size

    except Exception as e:
        print(f"Warning: Could not estimate batch size for {sample_file_path}: {e}")
        print("Using conservative batch size of 2")
        return 2


@jaxtyped(typechecker=beartype)
def estimate_memory_usage(
    file_path: str,
    include_preprocessing: Optional[bool] = True,
    include_blob_detection: Optional[bool] = True,
) -> scalar_float:
    """
    Description
    -----------
    Estimate memory usage in GB for processing a single MRC file.

    Parameters
    ----------
    - `file_path` (str):
        Path to MRC file
    - `include_preprocessing` (bool, optional):
        Include memory for preprocessing operations. Default is True
    - `include_blob_detection` (bool, optional):
        Include memory for blob detection. Default is True

    Returns
    -------
    - `memory_gb` (scalar_float):
        Estimated memory usage in GB
    """
    try:
        with mrcfile.open(file_path, mode="r", permissive=True) as mrc:
            data_shape: tuple = mrc.data.shape

            array_elements: scalar_int = int(jnp.prod(jnp.array(data_shape)))
            base_memory: scalar_float = float(array_elements * 8)

            total_memory: scalar_float = base_memory

            if include_preprocessing:
                preprocessing_memory: scalar_float = base_memory * 2.0
                total_memory += preprocessing_memory

            if include_blob_detection:
                typical_scales: scalar_int = 10
                downscale_factor: scalar_float = 4.0

                downscaled_elements: scalar_float = array_elements / (
                    downscale_factor**2
                )
                scale_space_memory: scalar_float = (
                    downscaled_elements * typical_scales * 8
                )
                total_memory += scale_space_memory

            memory_gb: scalar_float = total_memory / 1e9

            return memory_gb

    except Exception as e:
        print(f"Warning: Could not estimate memory for {file_path}: {e}")
        return 1.0


@jaxtyped(typechecker=beartype)
def get_optimal_batch_size(
    file_list: list[str],
    target_memory_gb: Optional[scalar_float] = 4.0,
    sample_fraction: Optional[scalar_float] = 0.1,
) -> scalar_int:
    """
    Description
    -----------
    Get optimal batch size by sampling multiple files from the list.

    Parameters
    ----------
    - `file_list` (list[str]):
        List of file paths to process
    - `target_memory_gb` (scalar_float, optional):
        Target memory usage in GB. Default is 4.0
    - `sample_fraction` (scalar_float, optional):
        Fraction of files to sample for estimation. Default is 0.1

    Returns
    -------
    - `batch_size` (scalar_int):
        Optimal batch size
    """
    if not file_list:
        return 1

    num_samples: scalar_int = max(1, int(len(file_list) * sample_fraction))
    sample_indices: list = jnp.linspace(
        0, len(file_list) - 1, num_samples, dtype=int
    ).tolist()

    batch_sizes: list = []

    for idx in sample_indices:
        try:
            batch_size: scalar_int = estimate_batch_size(
                file_list[idx], target_memory_gb=target_memory_gb
            )
            batch_sizes.append(batch_size)
        except Exception:
            continue

    if not batch_sizes:
        return 2

    optimal_batch_size: scalar_int = int(min(batch_sizes))

    return max(1, optimal_batch_size)
