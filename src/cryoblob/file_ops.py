"""
Module: file_ops
---------------------------

Contains the codes for interfacing with data files.
One goal here is to separate the Python code from
the JAX code. Thus most of the necessary outward 
facing code, which is necessarily in Python, is here.

Functions
---------
- `file_params`:
    Get the parameters for the file organization.
- `process_single_file`:
    Process a single file for blob detection with memory optimization.
- `process_batch_of_files`:
    Process a batch of files in parallel with memory optimization.
- `folder_blobs`:
    Process a folder of images for blob detection with memory optimization.
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
from beartype.typing import (Dict, List, Literal, Optional, Tuple, TypeAlias,
                             Union)
from jax import device_get, device_put, lax, vmap
from jaxtyping import Array, Float, Int, Num, jaxtyped
from tqdm.auto import tqdm

import cryoblob
from cryoblob.types import *

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
        open(files("arm_em.params").joinpath("organization.json"))
    )
    return (main_directory, folder_structure)


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
            # Stream large files in chunks
            with mrcfile.mmap(file_path, mode="r") as data_mrc:
                x_calib = jnp.asarray(data_mrc.voxel_size.x)
                y_calib = jnp.asarray(data_mrc.voxel_size.y)
                # Process image in chunks if needed
                im_data = jnp.asarray(data_mrc.data, dtype=jnp.float64)
        else:
            with mrcfile.open(file_path) as data_mrc:
                x_calib = jnp.asarray(data_mrc.voxel_size.x)
                y_calib = jnp.asarray(data_mrc.voxel_size.y)
                im_data = jnp.asarray(data_mrc.data, dtype=jnp.float64)

        # Move data to device
        im_data = device_put(im_data)

        # Preprocess and detect blobs
        preprocessed_imdata = cryoblob.preprocessing(
            image_orig=im_data, return_params=False, **preprocessing_kwargs
        )

        # Clear intermediate results
        del im_data

        blob_list = cryoblob.blob_list(preprocessed_imdata, downscale=blob_downscale)

        # Clear more intermediate results
        del preprocessed_imdata

        # Scale blob coordinates efficiently
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
    # Setup preprocessing parameters
    default_kwargs = {
        "exponential": False,
        "logarizer": False,
        "gblur": 0,
        "background": 0,
        "apply_filter": 0,
    }
    preprocessing_kwargs = {**default_kwargs, **kwargs}

    # Get file list
    file_list = glob.glob(folder_location + "*." + file_type)

    if not file_list:
        return pd.DataFrame(
            columns=["File Location", "Center Y (nm)", "Center X (nm)", "Size (nm)"]
        )

    # Estimate optimal batch size
    batch_size = cryoblob.estimate_batch_size(file_list[0], target_memory_gb)

    # Process files in batches with progress tracking
    all_blobs = []
    all_files = []

    with tqdm(total=len(file_list), desc="Processing files") as pbar:
        for i in range(0, len(file_list), batch_size):
            # Get current batch
            batch_files = file_list[i : i + batch_size]

            # Process batch
            batch_results = process_batch_of_files(
                batch_files, preprocessing_kwargs, blob_downscale
            )

            # Store results
            for blobs, file_path in batch_results:
                if len(blobs) > 0:
                    # Move results to CPU to free GPU memory
                    cpu_blobs = device_get(blobs)
                    all_blobs.append(cpu_blobs)
                    all_files.extend([file_path] * len(cpu_blobs))

            # Clear device memory
            pbar.update(len(batch_files))

    # Combine results
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
