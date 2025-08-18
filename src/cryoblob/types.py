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
- `scalar_bool`:
    Zero dimensional boolean.
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
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Bool, Float, Int, Integer, Num, jaxtyped

scalar_float: TypeAlias = Union[float, Float[Array, ""]]
scalar_int: TypeAlias = Union[int, Integer[Array, ""]]
scalar_num: TypeAlias = Union[int, float, Num[Array, ""]]
scalar_bool: TypeAlias = Union[bool, Bool[Array, ""]]
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


@register_pytree_node_class
class PreprocessingConfig(NamedTuple):
    """
    Description
    -----------
    A JAX-compatible data structure for image preprocessing parameters.

    Attributes
    ----------
    - `exponential` (Bool[Array, ""]):
        Apply exponential function to enhance contrast.
    - `logarizer` (Bool[Array, ""]):
        Apply logarithmic transformation.
    - `gblur` (Int[Array, ""]):
        Gaussian blur sigma, 0 means no blur.
    - `background` (Int[Array, ""]):
        Background subtraction sigma, 0 means no subtraction.
    - `apply_filter` (Int[Array, ""]):
        Wiener filter kernel size, 0 means no filter.
    """

    exponential: Bool[Array, ""]
    logarizer: Bool[Array, ""]
    gblur: Int[Array, ""]
    background: Int[Array, ""]
    apply_filter: Int[Array, ""]

    def tree_flatten(self):
        children = (
            self.exponential,
            self.logarizer,
            self.gblur,
            self.background,
            self.apply_filter,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class BlobDetectionConfig(NamedTuple):
    """
    Description
    -----------
    A JAX-compatible data structure for blob detection parameters.

    Attributes
    ----------
    - `min_sigma` (Float[Array, ""]):
        Minimum sigma for Laplacian of Gaussian.
    - `max_sigma` (Float[Array, ""]):
        Maximum sigma for Laplacian of Gaussian.
    - `num_sigma` (Int[Array, ""]):
        Number of sigma values to test.
    - `threshold` (Float[Array, ""]):
        Detection threshold.
    - `exclude_border` (Int[Array, ""]):
        Pixels to exclude from border.
    """

    min_sigma: Float[Array, ""]
    max_sigma: Float[Array, ""]
    num_sigma: Int[Array, ""]
    threshold: Float[Array, ""]
    exclude_border: Int[Array, ""]

    def tree_flatten(self):
        children = (
            self.min_sigma,
            self.max_sigma,
            self.num_sigma,
            self.threshold,
            self.exclude_border,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class FileProcessingConfig(NamedTuple):
    """
    Description
    -----------
    A JAX-compatible data structure for file processing and batch operations.

    Attributes
    ----------
    - `batch_size` (Int[Array, ""]):
        Number of files to process in parallel.
    - `memory_limit_gb` (Float[Array, ""]):
        Memory limit in GB.
    """

    batch_size: Int[Array, ""]
    memory_limit_gb: Float[Array, ""]

    def tree_flatten(self):
        children = (
            self.batch_size,
            self.memory_limit_gb,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class MRCMetadata(NamedTuple):
    """
    Description
    -----------
    A JAX-compatible data structure for MRC file metadata.

    Attributes
    ----------
    - `nx` (Int[Array, ""]):
        Number of columns (fastest changing).
    - `ny` (Int[Array, ""]):
        Number of rows.
    - `nz` (Int[Array, ""]):
        Number of sections (slowest changing).
    - `mode` (Int[Array, ""]):
        Data type (0=int8, 1=int16, 2=float32, etc.).
    - `dmin` (Float[Array, ""]):
        Minimum density value.
    - `dmax` (Float[Array, ""]):
        Maximum density value.
    - `dmean` (Float[Array, ""]):
        Mean density value.
    """

    nx: Int[Array, ""]
    ny: Int[Array, ""]
    nz: Int[Array, ""]
    mode: Int[Array, ""]
    dmin: Float[Array, ""]
    dmax: Float[Array, ""]
    dmean: Float[Array, ""]

    def tree_flatten(self):
        children = (
            self.nx,
            self.ny,
            self.nz,
            self.mode,
            self.dmin,
            self.dmax,
            self.dmean,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class AdaptiveFilterConfig(NamedTuple):
    """
    Description
    -----------
    A JAX-compatible data structure for adaptive filtering parameters.

    Attributes
    ----------
    - `kernel_size` (Int[Array, ""]):
        Size of the filter kernel.
    - `noise_estimate` (Float[Array, ""]):
        Initial noise estimate.
    - `iterations` (Int[Array, ""]):
        Number of adaptation iterations.
    """

    kernel_size: Int[Array, ""]
    noise_estimate: Float[Array, ""]
    iterations: Int[Array, ""]

    def tree_flatten(self):
        children = (
            self.kernel_size,
            self.noise_estimate,
            self.iterations,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class RidgeDetectionConfig(NamedTuple):
    """
    Description
    -----------
    A JAX-compatible data structure for ridge detection parameters.

    Attributes
    ----------
    - `min_scale` (Float[Array, ""]):
        Minimum scale for ridge detection.
    - `max_scale` (Float[Array, ""]):
        Maximum scale for ridge detection.
    - `scale_step` (Float[Array, ""]):
        Step size for scale space.
    - `threshold` (Float[Array, ""]):
        Detection threshold.
    """

    min_scale: Float[Array, ""]
    max_scale: Float[Array, ""]
    scale_step: Float[Array, ""]
    threshold: Float[Array, ""]

    def tree_flatten(self):
        children = (
            self.min_scale,
            self.max_scale,
            self.scale_step,
            self.threshold,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class WatershedConfig(NamedTuple):
    """
    Description
    -----------
    A JAX-compatible data structure for watershed segmentation parameters.

    Attributes
    ----------
    - `min_distance` (Int[Array, ""]):
        Minimum distance between markers.
    - `threshold_abs` (Float[Array, ""]):
        Absolute threshold for markers (use -1 for None).
    - `compactness` (Float[Array, ""]):
        Compactness parameter for watershed.
    """

    min_distance: Int[Array, ""]
    threshold_abs: Float[Array, ""]  # Use -1 to indicate None
    compactness: Float[Array, ""]

    def tree_flatten(self):
        children = (
            self.min_distance,
            self.threshold_abs,
            self.compactness,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class HessianBlobConfig(NamedTuple):
    """
    Description
    -----------
    A JAX-compatible data structure for Hessian-based blob detection.

    Attributes
    ----------
    - `min_sigma` (Float[Array, ""]):
        Minimum sigma for scale space.
    - `max_sigma` (Float[Array, ""]):
        Maximum sigma for scale space.
    - `num_sigma` (Int[Array, ""]):
        Number of scales to test.
    - `threshold` (Float[Array, ""]):
        Detection threshold.
    """

    min_sigma: Float[Array, ""]
    max_sigma: Float[Array, ""]
    num_sigma: Int[Array, ""]
    threshold: Float[Array, ""]

    def tree_flatten(self):
        children = (
            self.min_sigma,
            self.max_sigma,
            self.num_sigma,
            self.threshold,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class EnhancedBlobDetectionConfig(NamedTuple):
    """
    Description
    -----------
    A JAX-compatible data structure for enhanced multi-method blob detection.

    Attributes
    ----------
    - `min_blob_size` (Float[Array, ""]):
        Minimum expected blob size.
    - `max_blob_size` (Float[Array, ""]):
        Maximum expected blob size.
    - `detection_threshold` (Float[Array, ""]):
        Overall detection threshold.
    - `use_ridge_detection` (Bool[Array, ""]):
        Enable ridge detection for elongated objects.
    - `use_watershed` (Bool[Array, ""]):
        Enable watershed for overlapping blobs.
    """

    min_blob_size: Float[Array, ""]
    max_blob_size: Float[Array, ""]
    detection_threshold: Float[Array, ""]
    use_ridge_detection: Bool[Array, ""]
    use_watershed: Bool[Array, ""]

    def tree_flatten(self):
        children = (
            self.min_blob_size,
            self.max_blob_size,
            self.detection_threshold,
            self.use_ridge_detection,
            self.use_watershed,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
