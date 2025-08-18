import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from cryoblob.types import MRC_Image, scalar_float, scalar_int, scalar_num
from jax import tree_util


class TestTypeAliases(parameterized.TestCase):

    def test_scalar_float_accepts_python_float(self):
        val: scalar_float = 3.14
        assert isinstance(val, float)

    def test_scalar_float_accepts_jax_array(self):
        val: scalar_float = jnp.array(3.14)
        assert isinstance(val, jnp.ndarray)
        assert val.ndim == 0

    def test_scalar_int_accepts_python_int(self):
        val: scalar_int = 42
        assert isinstance(val, int)

    def test_scalar_int_accepts_jax_array(self):
        val: scalar_int = jnp.array(42)
        assert isinstance(val, jnp.ndarray)
        assert val.ndim == 0

    def test_scalar_num_accepts_all_types(self):
        vals: list[scalar_num] = [
            42,
            3.14,
            jnp.array(42),
            jnp.array(3.14),
        ]
        for val in vals:
            assert isinstance(val, (int, float, jnp.ndarray))
