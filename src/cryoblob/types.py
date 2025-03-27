"""
Module: types
---------------------------

A single location for storing commonly
used type aliases and PyTrees.

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
"""
from beartype.typing import TypeAlias, Union
from jaxtyping import Array, Float, Integer, Num

scalar_float: TypeAlias = Union[float, Float[Array, ""]]
scalar_int: TypeAlias = Union[int, Integer[Array, ""]]
scalar_num: TypeAlias = Union[int, float, Num[Array, ""]]
non_jax_number: TypeAlias = Union[int, float]
