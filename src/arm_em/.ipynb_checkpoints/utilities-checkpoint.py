import json
import os
from importlib.resources import files
from typing import List, Optional, Tuple, Union

try:
    import cucim.skimage.exposure as xexpose
    import cupy as xp
    import cupyx.scipy.ndimage as xnd
    import cupyx.scipy.signal as xsig
except ImportError:
    import skimage.exposure as xexpose
    import numpy as xp
    import scipy.ndimage as xnd
    import scipy.signal as xsig
import numpy as np
from nptyping import Bool, Float, Int, NDArray, Shape


def file_params() -> Tuple[str, dict]:
    """
    Run this at the beginning to generate the dict
    This gives both the absolute and relative paths
    on how the files are organized.

    Returns:
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


def laplacian_gaussian(
    image: NDArray[Shape["*, *"], Float],
    standard_deviation: Optional[Union[Int, Float]] = 3,
    hist_stretch: Optional[Bool] = True,
    sampling: Optional[Union[Float, Int]] = 1,
    normalized: Optional[Bool] = True,
) -> NDArray[Shape["*, *"], Float]:
    """
    Applies Laplacian of Gaussian (LoG) filtering to an input image.

    Args:
        - `image` (cupy.ndarray):
            An input image represented as a 2D CuPy array.
        - `standard_deviation` (int, optional):
            The standard deviation of the Gaussian filter.
            Default is 3.
        - `hist_stretch` (bool, optional):
            A boolean indicating whether to perform histogram stretching on the image.
            Default is True.
        - `sampling` (float, optional):
            The downsampling factor for the image.
            Default is 1.

    Returns:
        - `filtered` (NDArray[Shape["*, *"], Float]):
            The laplacian of gaussian filtered image.
    """
    image: NDArray[Shape["*, *"], Float] = xp.asarray(image.astype(xp.float64))
    if sampling != 1:
        sampled_image: NDArray[Shape["*, *"], Float] = xnd.zoom(image, sampling)
    else:
        sampled_image: NDArray[Shape["*, *"], Float] = xp.copy(image)
    if hist_stretch:
        sampled_image = xexpose.equalize_hist(sampled_image)
    gauss_image: NDArray[Shape["*, *"], Float] = xnd.gaussian_filter(
        sampled_image, standard_deviation
    )
    laplacian: NDArray[Shape["3, 3"], Float] = xp.asarray(
        (
            (0.0, 1.0, 0.0),
            (1.0, -4.0, 1.0),
            (0.0, 1.0, 0.0),
        ),
        dtype=np.float64,
    )
    filtered: NDArray[Shape["*, *"], Float] = xsig.convolve2d(
        gauss_image, laplacian, mode="same", boundary="symm", fillvalue=0
    )
    if normalized:
        filtered = filtered * standard_deviation
    return filtered


def preprocessing(
    image_orig: NDArray[Shape["*, *"], Float],
    return_params: Optional[Bool] = False,
    exponential: Optional[Bool] = True,
    logarizer: Optional[Bool] = False,
    gblur: Optional[Int] = 2,
    background: Optional[Int] = 0,
    apply_filter: Optional[Int] = 0,
) -> NDArray[Shape["*, *"], Float]:
    """
    Pre-processing of low SNR images to
    improve contrast of blobs.

    Args:
        - `image_orig` (NDArray[Shape["*, *"], Float]):
            An input image represented as a 2D CuPy array.
        - `return_params` (Bool, optional):
            A boolean indicating whether to return the processing parameters.
            Default is False.
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
        - `image_proc` (NDArray[Shape["*, *"], Float]):
            The pre-processed image
    """
    processing_params: dict = {
        "exponential": exponential,
        "logarizer": logarizer,
        "gblur": gblur,
        "background": background,
        "apply_filter": apply_filter,
    }
    image_proc: NDArray[Shape["*, *"], Float] = (image_orig - xp.amin(image_orig)) / (
        xp.amax(image_orig) - xp.amin(image_orig)
    )
    if exponential:
        image_proc = xp.exp(image_proc)
    if logarizer:
        image_proc = xp.log(image_proc)
    if gblur > 0:
        image_proc = xnd.gaussian_filter(image_proc, gblur)
    if background > 0:
        image_proc = image_proc - xnd.gaussian_filter(image_proc, background)
    if apply_filter > 0:
        image_proc = xsig.wiener(image_proc, mysize=apply_filter)
    if return_params:
        return image_proc, processing_params
    else:
        return image_proc
