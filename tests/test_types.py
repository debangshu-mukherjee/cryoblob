import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax import tree_util

from cryoblob.types import (MRC_Image, make_MRC_Image, scalar_float,
                            scalar_int, scalar_num)


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


class TestMRCImage(chex.TestCase):

    def setUp(self):
        super().setUp()
        self.sample_2d_data = {
            "image_data": jnp.ones((100, 100)),
            "voxel_size": jnp.array([1.0, 1.2, 1.2]),
            "origin": jnp.array([0.0, 0.0, 0.0]),
            "data_min": jnp.array(0.0),
            "data_max": jnp.array(1.0),
            "data_mean": jnp.array(0.5),
            "mode": jnp.array(2),
        }

        self.sample_3d_data = {
            "image_data": jnp.ones((50, 100, 100)),
            "voxel_size": jnp.array([1.5, 1.2, 1.2]),
            "origin": jnp.array([10.0, 20.0, 30.0]),
            "data_min": jnp.array(-1.0),
            "data_max": jnp.array(2.0),
            "data_mean": jnp.array(0.7),
            "mode": jnp.array(2),
        }

    @chex.all_variants
    def test_make_mrc_image_2d(self):
        def create_mrc():
            return make_MRC_Image(**self.sample_2d_data)

        mrc_image = self.variant(create_mrc)()

        assert isinstance(mrc_image, MRC_Image)
        assert mrc_image.image_data.shape == (100, 100)
        chex.assert_trees_all_close(
            mrc_image.voxel_size, self.sample_2d_data["voxel_size"]
        )
        chex.assert_trees_all_close(mrc_image.origin, self.sample_2d_data["origin"])
        chex.assert_trees_all_close(mrc_image.data_min, self.sample_2d_data["data_min"])
        chex.assert_trees_all_close(mrc_image.data_max, self.sample_2d_data["data_max"])
        chex.assert_trees_all_close(
            mrc_image.data_mean, self.sample_2d_data["data_mean"]
        )
        assert mrc_image.mode == self.sample_2d_data["mode"]

    @chex.all_variants
    def test_make_mrc_image_3d(self):
        def create_mrc():
            return make_MRC_Image(**self.sample_3d_data)

        mrc_image = self.variant(create_mrc)()

        assert isinstance(mrc_image, MRC_Image)
        assert mrc_image.image_data.shape == (50, 100, 100)
        chex.assert_trees_all_close(
            mrc_image.voxel_size, self.sample_3d_data["voxel_size"]
        )
        chex.assert_trees_all_close(mrc_image.origin, self.sample_3d_data["origin"])

    def test_mrc_image_is_pytree(self):
        mrc_image = make_MRC_Image(**self.sample_2d_data)
        leaves, treedef = tree_util.tree_flatten(mrc_image)
        reconstructed = tree_util.tree_unflatten(treedef, leaves)
        assert len(leaves) == 7
        assert isinstance(reconstructed, MRC_Image)
        chex.assert_trees_all_equal(reconstructed, mrc_image)

    @chex.all_variants
    def test_mrc_image_processing(self):
        mrc_image = make_MRC_Image(**self.sample_2d_data)

        def process_mrc(mrc: MRC_Image) -> scalar_float:
            return jnp.sum(mrc.image_data) * mrc.data_mean

        result = self.variant(process_mrc)(mrc_image)
        expected = jnp.sum(mrc_image.image_data) * mrc_image.data_mean
        chex.assert_trees_all_close(result, expected)

    def test_mrc_image_vmap_compatible(self):
        scales = jnp.array([0.5, 1.0, 2.0])

        def create_scaled_mrc(scale: scalar_float) -> MRC_Image:
            data = self.sample_2d_data.copy()
            data["image_data"] = data["image_data"] * scale
            data["data_max"] = data["data_max"] * scale
            data["data_mean"] = data["data_mean"] * scale
            return make_MRC_Image(**data)

        mrc_images = jax.vmap(create_scaled_mrc)(scales)

        def get_mean(mrc: MRC_Image) -> scalar_float:
            return mrc.data_mean

        means = jax.vmap(get_mean)(mrc_images)
        expected_means = self.sample_2d_data["data_mean"] * scales
        chex.assert_trees_all_close(means, expected_means)

    def test_mrc_image_tree_map(self):
        mrc_image = make_MRC_Image(**self.sample_2d_data)

        def scale_by_two(x):
            if isinstance(x, jnp.ndarray):
                return x * 2
            return x

        scaled_mrc = jax.tree_map(scale_by_two, mrc_image)

        chex.assert_trees_all_close(scaled_mrc.image_data, mrc_image.image_data * 2)
        chex.assert_trees_all_close(scaled_mrc.voxel_size, mrc_image.voxel_size * 2)
        chex.assert_trees_all_close(scaled_mrc.data_mean, mrc_image.data_mean * 2)

    def test_mrc_image_immutability(self):
        mrc_image = make_MRC_Image(**self.sample_2d_data)
        with self.assertRaises(AttributeError):
            mrc_image.data_mean = 0.8

    @parameterized.parameters(
        (jnp.float32,),
        (jnp.float64,),
        (jnp.int32,),
    )
    def test_mrc_image_with_different_dtypes(self, dtype):
        image_data = jnp.ones((50, 50), dtype=dtype)
        mrc_image = make_MRC_Image(
            image_data=image_data,
            voxel_size=jnp.array([1.0, 1.0, 1.0]),
            origin=jnp.zeros(3),
            data_min=0.0,
            data_max=1.0,
            data_mean=0.5,
            mode=2,
        )
        assert mrc_image.image_data.dtype == dtype

    @parameterized.parameters(
        ((10, 10),),
        ((20, 30),),
        ((5, 10, 15),),
    )
    def test_mrc_image_various_shapes(self, shape):
        mrc_image = make_MRC_Image(
            image_data=jnp.ones(shape),
            voxel_size=jnp.array([1.0, 1.0, 1.0]),
            origin=jnp.zeros(3),
            data_min=0.0,
            data_max=1.0,
            data_mean=1.0,
            mode=2,
        )
        assert mrc_image.image_data.shape == shape


class TestTypeValidation(chex.TestCase):

    def test_make_mrc_image_type_validation(self):
        make_MRC_Image(
            image_data=jnp.ones((10, 10)),
            voxel_size=jnp.array([1.0, 1.0, 1.0]),
            origin=jnp.zeros(3),
            data_min=0.0,
            data_max=1.0,
            data_mean=0.5,
            mode=2,
        )


if __name__ == "__main__":
    pytest.main([__file__])
