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

Factory Functions
----------------
- `make_MRC_Image`:
    Factory function to create an MRC_Image instance.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, TypeAlias, Union
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, Integer, Num, jaxtyped

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


@jaxtyped(typechecker=beartype)
def make_MRC_Image(
    image_data: Union[Num[Array, "H W"], Num[Array, "D H W"]],
    voxel_size: Float[Array, "3"],
    origin: Float[Array, "3"],
    data_min: scalar_float,
    data_max: scalar_float,
    data_mean: scalar_float,
    mode: scalar_int,
) -> MRC_Image:
    """
    Description
    -----------
    Factory function to create an MRC_Image instance.

    Parameters
    ----------
    - `image_data` (Num[Array, "H W"] | Num[Array, "D H W"]):
        The image data array from the MRC file. Can be 2D or 3D.
    - `voxel_size` (Float[Array, "3"]):
        Voxel size in the order (Z, Y, X).
    - `origin` (Float[Array, "3"]):
        Origin coordinates from the MRC file header (Z, Y, X).
    - `data_min` (scalar_float):
        Minimum value of image data (as stored in header).
    - `data_max` (scalar_float):
        Maximum value of image data (as stored in header).
    - `data_mean` (scalar_float):
        Mean value of image data (as stored in header).
    - `mode` (scalar_int):
        Data type mode from MRC header (e.g., 0: int8, 2: float32).

    Returns
    -------
    - `MRC_Image`:
        An instance of the MRC_Image PyTree structure.
    """
    return MRC_Image(
        image_data=image_data,
        voxel_size=voxel_size,
        origin=origin,
        data_min=data_min,
        data_max=data_max,
        data_mean=data_mean,
        mode=mode,
    )
