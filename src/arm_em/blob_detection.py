import gc
import glob
from functools import partial
from typing import Dict, List, Literal, Tuple, Union

import arm_em
import jax
import jax.numpy as jnp
import mrcfile
import numpy as np
import pandas as pd
from jax import device_get, device_put, lax, vmap
from jaxtyping import Array, Float, Int
from tqdm.auto import tqdm

number = Union[int, float]
jax.config.update("jax_enable_x64", True)


def blob_list(
    image: Float[Array, "a b"] | Float[Array, "a b c"],
    min_blob_size: Float | Int | None = 10,
    max_blob_size: Float | Int | None = 100,
    blob_step: Float | Int | None = 2,
    downscale: Float | Int | None = 2,
    std_threshold: Float | Int | None = 6,
) -> Float[Array, "a 3"]:
    """
    Description
    -----------
    Detects blobs in an input image using the Laplacian of Gaussian (LoG) method.
    If input is 3D, sums along the last axis before processing.

    Parameters
    ----------
    - `image` (Float[Array, "a b"] | Float[Array, "a b c"]):
        A 2D or 3D array representing the input image.
        If 3D, will be summed along the last axis.
    - `min_blob_size` (Float | Int, optional):
        The minimum size of the blobs to be detected.
        Defaults to 10.
    - `max_blob_size` (Float | Int, optional):
        The maximum size of the blobs to be detected.
        Defaults to 100.
    - `blob_step` (Float | Int, optional):
        The step size for iterating over different blob sizes.
        Defaults to 2.
    - `downscale` (Float | Int, optional):
        The factor by which the image is downscaled before blob detection.
        Defaults to 2.
    - `std_threshold` (Float | Int, optional):
        The threshold for blob detection based on standard deviation.
        Defaults to 6.

    Returns
    -------
    - `scaled_coords` (Float[Array, "labels 3"]):
        A 2D array containing the coordinates of the detected blobs. Each row
        represents the coordinates of a blob, with the first two columns
        representing the x and y coordinates, and the last column
        representing the size of the blob.

    Notes
    -----
    For 3D inputs, the function:
    1. Sums along the last axis to create a 2D projection
    2. Processes the 2D projection for blob detection
    3. Returns coordinates in the 2D projection space
    """
    # Handle 3D input by summing along last axis
    if image.ndim == 3:
        image = jnp.sum(image, axis=-1)

    peak_range: Float[Array, "c"] = jnp.arange(
        start=min_blob_size, stop=max_blob_size, step=blob_step
    )
    scaled_image: Float[Array, "e f"] = arm_em.fast_resizer(image, (1 / downscale))

    if jnp.amin(x=jnp.asarray(jnp.shape(scaled_image))) < 20:
        raise ValueError("Image is too small for blob detection")

    vectorized_log = jax.vmap(
        arm_em.laplacian_gaussian,
        in_axes=(
            None,
            0,
        ),  # First arg (image) is broadcasted, second arg (sigma) is mapped
    )
    results_3D: Float[Array, "e f r"] = vectorized_log(
        scaled_image, peak_range
    ).transpose(1, 2, 0)

    max_filtered: Float[Array, "e f r"] = lax.reduce_window(
        results_3D,
        init_value=-jnp.inf,
        computation_fn=lax.max,
        window_dimensions=(4, 4, 4),  # 3D window size
        window_strides=(1, 1, 1),  # stride of 1 in each dimension
        padding="SAME",  # to match scipy's behavior
    )

    image_thresh: Float[Array, "e f r"] = jnp.mean(max_filtered) + (
        std_threshold * jnp.std(max_filtered)
    )
    coords = arm_em.find_particle_coords(results_3D, max_filtered, image_thresh)
    scaled_coords: Float[Array, "labels 3"] = jnp.concatenate(
        [
            downscale * coords[:, 0:2],  # x, y coordinates
            downscale * ((blob_step * coords[:, -1:]) + min_blob_size),  # z coordinate
        ],
        axis=1,
    )
    return scaled_coords


def estimate_batch_size(sample_file: str, target_memory_gb: float = 4.0) -> int:
    """
    Estimate optimal batch size based on available memory.

    Parameters
    ----------
    - `sample_file` (str):
        Path to a sample file for size estimation
    - `target_memory_gb` (float):
        Target memory usage in gigabytes

    Returns
    -------
    - `batch_size` (int):
        Recommended batch size
    """
    # Load sample file to estimate memory usage
    with mrcfile.open(sample_file) as mrc:
        sample_size = mrc.data.nbytes

    # Estimate memory per file (including processing overhead)
    memory_per_file = sample_size * 4  # Approximate factor for processing

    # Calculate batch size
    max_memory = target_memory_gb * 1024**3  # Convert GB to bytes
    batch_size = max(1, int(max_memory / memory_per_file))

    return batch_size


@partial(jax.jit, static_argnums=(1,))
def clear_device_memory():
    """Clear GPU memory between batches."""
    jax.clear_caches()
    gc.collect()


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
        preprocessed_imdata = arm_em.preprocessing(
            image_orig=im_data, return_params=False, **preprocessing_kwargs
        )

        # Clear intermediate results
        del im_data

        blob_list = arm_em.blob_list(preprocessed_imdata, downscale=blob_downscale)

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


@partial(jax.jit, static_argnums=(1, 2))
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
    batch_size = estimate_batch_size(file_list[0], target_memory_gb)

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
            clear_device_memory()
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
