"""
Module: types
---------------------------
A single location for storing commonly
used type aliases and PyTrees along with
factory functions for creating them.

Types
-----
- `scalar_float`:
    Zero dimensional floating point number
- `scalar_int`:
    Zero dimensional integer.
- `scalar_num`:
    Zero dimensional number, that can either be a
    floating point number or an integer.
- `non_jax_number`:
    A number that is not a JAX array. This is because
    even single number are stored as 0D JAX arrays.

PyTrees
-------
- `MRC_Image`:
    A PyTree structure for MRC images.
    Contains the image data and metadata.
- `PreprocessingConfig`:
    PyTree for image preprocessing parameters
- `BlobDetectionConfig`:
    PyTree for blob detection parameters
- `FileProcessingConfig`:
    PyTree for file processing and batch operations
- `MRCMetadata`:
    PyTree for MRC file metadata
- `RidgeDetectionConfig`:
    PyTree for ridge detection parameters
- `WatershedConfig`:
    PyTree for watershed segmentation parameters
- `EnhancedBlobDetectionConfig`:
    PyTree for enhanced blob detection combining multiple methods
- `HessianBlobConfig`:
    PyTree for Hessian-based blob detection
- `AdaptiveFilterConfig`:
    PyTree for adaptive filtering parameters
"""

from beartype import beartype
from beartype.typing import NamedTuple, TypeAlias, Union
import jax
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, Integer, Num, Bool, Int, jaxtyped

scalar_float: TypeAlias = Union[float, Float[Array, ""]]
scalar_int: TypeAlias = Union[int, Integer[Array, ""]]
scalar_num: TypeAlias = Union[int, float, Num[Array, ""]]
non_jax_number: TypeAlias = Union[int, float]


@jaxtyped(typechecker=beartype)
@register_pytree_node_class
class MRC_Image(NamedTuple):
    """
    Description
    -----------
    A JAX-compatible data structure representing an MRC image file.

    Attributes
    ----------
    - `image_data` (Num[Array, "H W"] | Num[Array, "D H W"])
        The image data array from the MRC file. Either 2D or 3D.
    - `voxel_size` (Float[Array, "3"]):
        The voxel size (Å/pixel) in the order (Z, Y, X).
    - `origin` (Float[Array, "3"]):
        Origin coordinates from the MRC file header (Z, Y, X).
    - `data_min` (scalar_float)
        Minimum value of image data (as stored in header).
    - `data_max` (scalar_float)
        Maximum value of image data (as stored in header).
    - `data_mean` (scalar_float)
        Mean value of image data (as stored in header).
    - `mode` (scalar_int)
        Data type mode from MRC header (e.g., 0: int8, 2: float32).
    """

    image_data: Union[Num[Array, "H W"], Num[Array, "D H W"]]
    voxel_size: Float[Array, "3"]
    origin: Float[Array, "3"]
    data_min: scalar_float
    data_max: scalar_float
    data_mean: scalar_float
    mode: scalar_int

    def tree_flatten(self):
        children = (
            self.image_data,
            self.voxel_size,
            self.origin,
            self.data_min,
            self.data_max,
            self.data_mean,
            self.mode,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class PreprocessingConfig(NamedTuple):
    """
    PyTree for image preprocessing parameters.
    
    Attributes
    ----------
    exponential : bool
        Apply exponential function to enhance contrast
    logarizer : bool
        Apply logarithmic transformation
    gblur : int
        Gaussian blur sigma, 0 means no blur
    background : int
        Background subtraction sigma, 0 means no subtraction
    apply_filter : int
        Wiener filter kernel size, 0 means no filter
    """
    exponential: Bool[Array, ""]
    logarizer: Bool[Array, ""]
    gblur: Int[Array, ""]
    background: Int[Array, ""]
    apply_filter: Int[Array, ""]


class BlobDetectionConfig(NamedTuple):
    """
    PyTree for blob detection parameters.
    
    Attributes
    ----------
    min_sigma : float
        Minimum sigma for Laplacian of Gaussian
    max_sigma : float
        Maximum sigma for Laplacian of Gaussian
    num_sigma : int
        Number of sigma values to test
    threshold : float
        Detection threshold
    exclude_border : int
        Pixels to exclude from border
    """
    min_sigma: Float[Array, ""]
    max_sigma: Float[Array, ""]
    num_sigma: Int[Array, ""]
    threshold: Float[Array, ""]
    exclude_border: Int[Array, ""]


class FileProcessingConfig(NamedTuple):
    """
    PyTree for file processing and batch operations.
    
    Attributes
    ----------
    batch_size : int
        Number of files to process in parallel
    memory_limit_gb : float
        Memory limit in GB
    """
    batch_size: Int[Array, ""]
    memory_limit_gb: Float[Array, ""]


class MRCMetadata(NamedTuple):
    """
    PyTree for MRC file metadata.
    
    Attributes
    ----------
    nx : int
        Number of columns (fastest changing)
    ny : int
        Number of rows
    nz : int
        Number of sections (slowest changing)
    mode : int
        Data type (0=int8, 1=int16, 2=float32, etc.)
    dmin : float
        Minimum density value
    dmax : float
        Maximum density value
    dmean : float
        Mean density value
    """
    nx: Int[Array, ""]
    ny: Int[Array, ""]
    nz: Int[Array, ""]
    mode: Int[Array, ""]
    dmin: Float[Array, ""]
    dmax: Float[Array, ""]
    dmean: Float[Array, ""]


class AdaptiveFilterConfig(NamedTuple):
    """
    PyTree for adaptive filtering parameters.
    
    Attributes
    ----------
    kernel_size : int
        Size of the filter kernel
    noise_estimate : float
        Initial noise estimate
    iterations : int
        Number of adaptation iterations
    """
    kernel_size: Int[Array, ""]
    noise_estimate: Float[Array, ""]
    iterations: Int[Array, ""]


class RidgeDetectionConfig(NamedTuple):
    """
    PyTree for ridge detection parameters.
    
    Attributes
    ----------
    min_scale : float
        Minimum scale for ridge detection
    max_scale : float
        Maximum scale for ridge detection
    scale_step : float
        Step size for scale space
    threshold : float
        Detection threshold
    """
    min_scale: Float[Array, ""]
    max_scale: Float[Array, ""]
    scale_step: Float[Array, ""]
    threshold: Float[Array, ""]


class WatershedConfig(NamedTuple):
    """
    PyTree for watershed segmentation parameters.
    
    Attributes
    ----------
    min_distance : int
        Minimum distance between markers
    threshold_abs : float
        Absolute threshold for markers (optional, use -1 for None)
    compactness : float
        Compactness parameter for watershed
    """
    min_distance: Int[Array, ""]
    threshold_abs: Float[Array, ""]  # Use -1 to indicate None
    compactness: Float[Array, ""]


class HessianBlobConfig(NamedTuple):
    """
    PyTree for Hessian-based blob detection.
    
    Attributes
    ----------
    min_sigma : float
        Minimum sigma for scale space
    max_sigma : float
        Maximum sigma for scale space
    num_sigma : int
        Number of scales to test
    threshold : float
        Detection threshold
    """
    min_sigma: Float[Array, ""]
    max_sigma: Float[Array, ""]
    num_sigma: Int[Array, ""]
    threshold: Float[Array, ""]


class EnhancedBlobDetectionConfig(NamedTuple):
    """
    PyTree for enhanced multi-method blob detection.
    
    Attributes
    ----------
    min_blob_size : float
        Minimum expected blob size
    max_blob_size : float
        Maximum expected blob size
    detection_threshold : float
        Overall detection threshold
    use_ridge_detection : bool
        Enable ridge detection for elongated objects
    use_watershed : bool
        Enable watershed for overlapping blobs
    """
    min_blob_size: Float[Array, ""]
    max_blob_size: Float[Array, ""]
    detection_threshold: Float[Array, ""]
    use_ridge_detection: Bool[Array, ""]
    use_watershed: Bool[Array, ""]


# Register all PyTrees with JAX
jax.tree_util.register_pytree_node(
    PreprocessingConfig,
    lambda x: (list(x), None),
    lambda _, x: PreprocessingConfig(*x)
)

jax.tree_util.register_pytree_node(
    BlobDetectionConfig,
    lambda x: (list(x), None),
    lambda _, x: BlobDetectionConfig(*x)
)

jax.tree_util.register_pytree_node(
    FileProcessingConfig,
    lambda x: (list(x), None),
    lambda _, x: FileProcessingConfig(*x)
)

jax.tree_util.register_pytree_node(
    MRCMetadata,
    lambda x: (list(x), None),
    lambda _, x: MRCMetadata(*x)
)

jax.tree_util.register_pytree_node(
    AdaptiveFilterConfig,
    lambda x: (list(x), None),
    lambda _, x: AdaptiveFilterConfig(*x)
)

jax.tree_util.register_pytree_node(
    RidgeDetectionConfig,
    lambda x: (list(x), None),
    lambda _, x: RidgeDetectionConfig(*x)
)

jax.tree_util.register_pytree_node(
    WatershedConfig,
    lambda x: (list(x), None),
    lambda _, x: WatershedConfig(*x)
)

jax.tree_util.register_pytree_node(
    HessianBlobConfig,
    lambda x: (list(x), None),
    lambda _, x: HessianBlobConfig(*x)
)

jax.tree_util.register_pytree_node(
    EnhancedBlobDetectionConfig,
    lambda x: (list(x), None),
    lambda _, x: EnhancedBlobDetectionConfig(*x)
)