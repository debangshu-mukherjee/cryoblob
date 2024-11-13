import glob
from typing import Optional, Union

import jax
import jax.numpy as jnp
import mrcfile
import numpy as np
import pandas as pd
from jaxtyping import Array, Float, Int, Shape
from tqdm.auto import trange

import arm_em


def blob_list(
    image: Float[Array, "a b"],
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

    Parameters
    ----------
    - `image` (Float[Array, "a b"]):
        A 2D array representing the input image.
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
    - `scaled_coords` (NDArray[Shape["*, 3"], Float]):
        A 2D array containing the coordinates of the detected blobs. Each row
        represents the coordinates of a blob, with the first two columns
        representing the x and y coordinates, and the last column
        representing the size of the blob.
    """
    peak_range: Float[Array, "c"] = jnp.arange(
        start=min_blob_size, stop=max_blob_size, step=blob_step
    )
    scaled_image: Float[Array, "e f"] = arm_em.fast_resizer(image, (1 / downscale))
    if jnp.amin(x=jnp.asarray(jnp.shape(scaled_image))) < 20:
        raise ValueError("Image is too small for blob detection")
    results_3D: Float[Array, "d e f"] = jnp.zeros(
        shape=(scaled_image.shape[0], scaled_image.shape[1], len(peak_range)),
        dtype=jnp.float64,
    )
    for ii in range(len(peak_range)):
        results_3D[:, :, ii] = arm_em.laplacian_gaussian(
            scaled_image, standard_deviation=peak_range[ii]
        )

    max_filtered: NDArray[Shape["*, *, *"], Float] = xnd.maximum_filter(
        results_3D, size=(4, 4, 4)
    )
    image_thresh: NDArray[Shape["*, *, *"], Float] = xp.mean(max_filtered) + (
        std_threshold * xp.std(max_filtered)
    )
    labels, num_labels = xnd.label(max_filtered > image_thresh)
    coords: NDArray[Shape["*, 3"], Float] = xp.asarray(
        xnd.center_of_mass(
            results_3D, labels=labels, index=xp.arange(1, num_labels + 1)
        )
    )
    scaled_coords: NDArray[Shape["*, 3"], Float] = xp.zeros_like(coords)
    scaled_coords[:, 0:2] = downscale * coords[:, 0:2]
    scaled_coords[:, -1] = downscale * ((blob_step * coords[:, -1]) + min_blob_size)
    return scaled_coords


def folder_blobs(
    folder_location: str,
    file_type: Optional[str] = "mrc",
    blob_downscale: Optional[Union[Float, Int]] = 7,
    **kwargs,
) -> DataFrame:
    """
    Returns a pandas DataFrame containing information about the blobs found in
    the files in the specified folder.

    Args:
        - `folder_location` (str):
            The path to the folder containing the files.
        - `file_type` (str, optional):
            The type of files to search for in the folder. Default is "mrc".
        - `blob_downscale` (float or int, optional):
            The downscale factor used for blob detection. Default is 7.
        - `**kwargs`:
            Additional keyword arguments to pass for preprocessing the image
            data passed to `arm_em.preprocessing`. If no kwargs are passed,
            then the default will be that `summed_im` will not be preprocessed.
            The arguments are:
            - `exponential` (Bool, optional):
                A boolean indicating whether to apply an exponential function to the image.
                Default is True.
            - `logarizer` (Bool, optional):
                A boolean indicating whether to apply a log function to the image.
                Default is False.
            - `gblur` (Int, optional):
                The standard deviation of the Gaussian filter.
                Default is 2.
            - `background` (Int, optional):
                The standard deviation of the Gaussian filter for background subtraction.
                Default is 0.
            - `apply_filter` (Int, optional):
                If greater than 1, a Wiener filter is applied to the image.

    Returns:
        - `blob_dataframe` (DataFrame):
            A DataFrame containing the file location, center coordinates (Y and X),
            and size of each blob found in the specified folder.
    """
    default_kwargs = {
        "exponential": False,
        "logarizer": False,
        "gblur": 0,
        "background": 0,
        "apply_filter": 0,
    }
    preprocessing_kwargs = {**default_kwargs, **kwargs}
    file_list: NDArray[Shape["*"], str] = np.asarray(
        glob.glob(folder_location + "*." + file_type), dtype="str"
    )
    blobs: NDArray[Shape["1, 4"], Object] = np.zeros(shape=(1, 4), dtype=object)
    for im_file in trange(len(file_list)):
        data_mrc = mrcfile.open(file_list[im_file])
        if file_type == "mrc":
            x_calib: Float | Int = xp.asarray(data_mrc.voxel_size.x)
            y_calib: Float | Int = xp.asarray(data_mrc.voxel_size.y)
        else:
            x_calib: Float | Int = 1
            y_calib: Float | Int = 1
        im_data: NDArray[Shape["*, *"], Float] = xp.asarray(
            data_mrc.data, dtype=xp.float64
        )
        preprocessed_imdata: NDArray[Shape["*, *"], Float] = arm_em.preprocessing(
            image_orig=im_data, return_params=False, **preprocessing_kwargs
        )
        this_blob_list: NDArray[Shape["*, 3"], Float] = arm_em.blob_list(
            preprocessed_imdata, downscale=blob_downscale
        )
        this_blob_list[:, 0] = this_blob_list[:, 0] * y_calib
        this_blob_list[:, 1] = this_blob_list[:, 0] * x_calib
        this_blob_list[:, 2] = this_blob_list[:, 0] * (
            ((y_calib**2) + (x_calib**2)) ** 0.5
        )
        num_blobs: Int = this_blob_list.shape[0]
        blob_file: NDArray[Shape["*, 1"], str] = (
            np.repeat(file_list[im_file], num_blobs)
        ).reshape(-1, 1)
        this_blob_set: NDArray[Shape["*, 4"], Object] = np.asarray(
            np.hstack((blob_file, this_blob_list.get())), dtype=object
        )
        blobs = np.vstack((blobs, this_blob_set))
    blobs = blobs[1:, :]
    blob_dataframe: DataFrame[
        S[
            "File Location: str, Center Y (nm): Float, Center Y (nm): Float, Size (nm): Float"
        ]
    ] = pd.DataFrame(
        data=blobs,
        columns=["File Location", "Center Y (nm)", "Center X (nm)", "Size (nm)"],
    )
    return blob_dataframe


def folder_blobs_3D(
    folder_location: str,
    file_type: Optional[str] = "mrc",
    blob_downscale: Optional[Union[Float, Int]] = 7,
    **kwargs,
) -> DataFrame:
    """
    Returns a pandas DataFrame containing information about the blobs found in
    the files in the specified folder.

    Args:
        - `folder_location` (str):
            The path to the folder containing the files.
        - `file_type` (str, optional):
            The type of files to search for in the folder. Default is "mrc".
        - `blob_downscale` (float or int, optional):
            The downscale factor used for blob detection. Default is 7.
        - `**kwargs`:
            Additional keyword arguments to pass for preprocessing the image
            data passed to `arm_em.preprocessing`. If no kwargs are passed,
            then the default will be that `summed_im` will not be preprocessed.
            The arguments are:
            - `exponential` (Bool, optional):
                A boolean indicating whether to apply an exponential function to the image.
                Default is True.
            - `logarizer` (Bool, optional):
                A boolean indicating whether to apply a log function to the image.
                Default is False.
            - `gblur` (Int, optional):
                The standard deviation of the Gaussian filter.
                Default is 2.
            - `background` (Int, optional):
                The standard deviation of the Gaussian filter for background subtraction.
                Default is 0.
            - `apply_filter` (Int, optional):
                If greater than 1, a Wiener filter is applied to the image.

    Returns:
        - `blob_dataframe` (DataFrame):
            A DataFrame containing the file location, center coordinates (Y and X),
            and size of each blob found in the specified folder.
    """
    default_kwargs = {
        "exponential": False,
        "logarizer": False,
        "gblur": 0,
        "background": 0,
        "apply_filter": 0,
    }
    preprocessing_kwargs = {**default_kwargs, **kwargs}
    file_list: NDArray[Shape["*"], str] = np.asarray(
        glob.glob(folder_location + "*." + file_type), dtype="str"
    )
    blobs: NDArray[Shape["1, 4"], Object] = np.zeros(shape=(1, 4), dtype=object)
    for im_file in trange(len(file_list)):
        data_mrc = mrcfile.open(file_list[im_file])
        if file_type == "mrc":
            x_calib: Float | Int = xp.asarray(data_mrc.voxel_size.x)
            y_calib: Float | Int = xp.asarray(data_mrc.voxel_size.y)
        else:
            x_calib: Float | Int = 1
            y_calib: Float | Int = 1
        im_data: NDArray[Shape["*, *, *"], Float] = np.asarray(
            data_mrc.data, dtype=np.float64
        )
        # We load the data in NumPy, and then sum it first before
        # converting it to CuPy, to avoid out of memory errors.
        summed_imdata: NDArray[Shape["*, *"], Float] = np.sum(im_data, axis=0)
        summed_imdata = (summed_imdata - np.min(summed_imdata)) / (
            np.max(summed_imdata) - np.min(summed_imdata)
        )
        summed_imdata = xp.asarray(summed_imdata)
        preprocessed_summed: NDArray[Shape["*, *"], Float] = arm_em.preprocessing(
            image_orig=summed_imdata, return_params=False, **preprocessing_kwargs
        )
        this_blob_list: NDArray[Shape["*, 3"], Float] = arm_em.blob_list(
            preprocessed_summed, downscale=blob_downscale
        )
        this_blob_list[:, 0] = this_blob_list[:, 0] * y_calib
        this_blob_list[:, 1] = this_blob_list[:, 0] * x_calib
        this_blob_list[:, 2] = this_blob_list[:, 0] * (
            ((y_calib**2) + (x_calib**2)) ** 0.5
        )
        num_blobs: Int = this_blob_list.shape[0]
        blob_file: NDArray[Shape["*, 1"], str] = (
            np.repeat(file_list[im_file], num_blobs)
        ).reshape(-1, 1)
        this_blob_set: NDArray[Shape["*, 4"], Object] = np.asarray(
            np.hstack((blob_file, this_blob_list.get())), dtype=object
        )
        blobs = np.vstack((blobs, this_blob_set))
    blobs = blobs[1:, :]
    blob_dataframe: DataFrame[
        S[
            "File Location: str, Center Y (nm): Float, Center Y (nm): Float, Size (nm): Float"
        ]
    ] = pd.DataFrame(
        data=blobs,
        columns=["File Location", "Center Y (nm)", "Center X (nm)", "Size (nm)"],
    )
    return blob_dataframe
