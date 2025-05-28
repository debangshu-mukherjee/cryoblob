import matplotlib

matplotlib.use("Agg")

from unittest.mock import Mock, patch

import chex
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib_scalebar.scalebar as sb
import numpy as np
from absl.testing import parameterized

import cryoblob as cb
from cryoblob.types import make_MRC_Image


class TestPlotMRC(chex.TestCase, parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.sample_data = jnp.linspace(0, 100, 100).reshape(10, 10)
        self.mrc_image = make_MRC_Image(
            image_data=self.sample_data,
            voxel_size=jnp.array([1.0, 2.0, 2.0]),
            origin=jnp.zeros(3),
            data_min=jnp.min(self.sample_data),
            data_max=jnp.max(self.sample_data),
            data_mean=jnp.mean(self.sample_data),
            mode=2,
        )

        self.uniform_data = jnp.ones((20, 20)) * 50
        self.uniform_mrc = make_MRC_Image(
            image_data=self.uniform_data,
            voxel_size=jnp.array([1.0, 1.0, 1.0]),
            origin=jnp.zeros(3),
            data_min=50.0,
            data_max=50.0,
            data_mean=50.0,
            mode=2,
        )

    def tearDown(self):
        plt.close("all")
        super().tearDown()

    @patch("matplotlib.pyplot.show")
    def test_plot_mrc_basic(self, mock_show):
        cb.plot_mrc(self.mrc_image)

        mock_show.assert_called_once()

        assert len(plt.get_fignums()) > 0

        fig = plt.gcf()
        ax = plt.gca()

        assert fig.get_size_inches()[0] == 15
        assert fig.get_size_inches()[1] == 15

        assert not ax.axison

        assert len(ax.images) == 1

        artists = ax.get_children()
        scalebar_found = any(isinstance(artist, sb.ScaleBar) for artist in artists)
        assert scalebar_found

    @parameterized.parameters(
        ((10, 10), "viridis", "plain"),
        ((20, 20), "plasma", "log"),
        ((15, 15), "inferno", "exp"),
        ((5, 5), "magma", "plain"),
    )
    @patch("matplotlib.pyplot.show")
    def test_plot_mrc_parameters(self, image_size, cmap, mode, mock_show):
        cb.plot_mrc(self.mrc_image, image_size=image_size, cmap=cmap, mode=mode)

        fig = plt.gcf()
        ax = plt.gca()

        assert fig.get_size_inches()[0] == image_size[0]
        assert fig.get_size_inches()[1] == image_size[1]

        img = ax.images[0]
        assert img.get_cmap().name == cmap

    @patch("matplotlib.pyplot.show")
    def test_plot_mrc_mode_plain(self, mock_show):
        cb.plot_mrc(self.mrc_image, mode="plain")

        ax = plt.gca()
        img = ax.images[0]
        img_data = img.get_array()

        assert np.min(img_data) >= 0
        assert np.max(img_data) <= 1

    @patch("matplotlib.pyplot.show")
    def test_plot_mrc_mode_log(self, mock_show):
        cb.plot_mrc(self.mrc_image, mode="log")

        ax = plt.gca()
        img = ax.images[0]
        img_data = img.get_array()

        assert np.min(img_data) >= 0

    @patch("matplotlib.pyplot.show")
    def test_plot_mrc_mode_exp(self, mock_show):
        cb.plot_mrc(self.mrc_image, mode="exp")

        ax = plt.gca()
        img = ax.images[0]
        img_data = img.get_array()

        assert np.min(img_data) >= 1

    def test_plot_mrc_invalid_mode(self):
        with self.assertRaises(ValueError) as context:
            cb.plot_mrc(self.mrc_image, mode="invalid")

        assert "Invalid mode" in str(context.exception)

    @patch("matplotlib.pyplot.show")
    def test_plot_mrc_scalebar_properties(self, mock_show):
        cb.plot_mrc(self.mrc_image)

        ax = plt.gca()

        scalebar = None
        for artist in ax.get_children():
            if isinstance(artist, sb.ScaleBar):
                scalebar = artist
                break

        assert scalebar is not None

        assert scalebar.dx == 20.0
        assert scalebar.units == "nm"
        assert scalebar.location == "lower right"
        assert scalebar.color == "white"

    @patch("matplotlib.pyplot.show")
    def test_plot_mrc_uniform_data(self, mock_show):
        cb.plot_mrc(self.uniform_mrc)

        ax = plt.gca()
        assert len(ax.images) == 1

    @chex.all_variants
    def test_plot_mrc_with_jax_transformations(self):
        def transform_mrc(mrc):
            new_data = mrc.image_data * 2
            return make_MRC_Image(
                image_data=new_data,
                voxel_size=mrc.voxel_size,
                origin=mrc.origin,
                data_min=jnp.min(new_data),
                data_max=jnp.max(new_data),
                data_mean=jnp.mean(new_data),
                mode=mrc.mode,
            )

        transformed_mrc = self.variant(transform_mrc)(self.mrc_image)

        with patch("matplotlib.pyplot.show"):
            cb.plot_mrc(transformed_mrc)

        assert len(plt.get_fignums()) > 0

    @patch("matplotlib.pyplot.show")
    def test_plot_mrc_origin_lower(self, mock_show):
        cb.plot_mrc(self.mrc_image)

        ax = plt.gca()
        img = ax.images[0]

        assert img.origin == "lower"

    @parameterized.parameters(
        (jnp.float32,),
        (jnp.float64,),
        (jnp.int32,),
    )
    @patch("matplotlib.pyplot.show")
    def test_plot_mrc_different_dtypes(self, dtype, mock_show):
        data = self.sample_data.astype(dtype)
        mrc = make_MRC_Image(
            image_data=data,
            voxel_size=jnp.array([1.0, 1.0, 1.0]),
            origin=jnp.zeros(3),
            data_min=jnp.min(data),
            data_max=jnp.max(data),
            data_mean=jnp.mean(data),
            mode=2,
        )

        cb.plot_mrc(mrc)

        assert len(plt.get_fignums()) > 0

    @patch("matplotlib.pyplot.show")
    def test_plot_mrc_large_voxel_size(self, mock_show):
        large_voxel_mrc = make_MRC_Image(
            image_data=self.sample_data,
            voxel_size=jnp.array([10.0, 50.0, 50.0]),
            origin=jnp.zeros(3),
            data_min=jnp.min(self.sample_data),
            data_max=jnp.max(self.sample_data),
            data_mean=jnp.mean(self.sample_data),
            mode=2,
        )

        cb.plot_mrc(large_voxel_mrc)

        ax = plt.gca()
        scalebar = None
        for artist in ax.get_children():
            if isinstance(artist, sb.ScaleBar):
                scalebar = artist
                break

        assert scalebar is not None
        assert scalebar.dx == 500.0

    def test_plot_function_imports(self):
        assert hasattr(cb, "plot_mrc")

        import matplotlib

        backend = matplotlib.get_backend()
        assert backend is not None

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.show")
    def test_plot_mrc_figure_creation(self, mock_show, mock_subplots):
        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.imshow = Mock()
        mock_ax.add_artist = Mock()
        mock_ax.axis = Mock()
        mock_fig.tight_layout = Mock()

        mock_subplots.return_value = (mock_fig, mock_ax)

        cb.plot_mrc(self.mrc_image)

        mock_subplots.assert_called_once_with(figsize=(15, 15))
        mock_ax.imshow.assert_called_once()
        mock_ax.add_artist.assert_called_once()
        mock_ax.axis.assert_called_once_with("off")
        mock_fig.tight_layout.assert_called_once()
        mock_show.assert_called_once()


class TestPlotIntegration(chex.TestCase):

    def setUp(self):
        super().setUp()
        x, y = jnp.meshgrid(jnp.linspace(-5, 5, 100), jnp.linspace(-5, 5, 100))
        self.complex_data = jnp.exp(-(x**2 + y**2) / 2) * 100

        self.complex_mrc = make_MRC_Image(
            image_data=self.complex_data,
            voxel_size=jnp.array([1.0, 1.5, 1.5]),
            origin=jnp.array([10.0, 20.0, 30.0]),
            data_min=jnp.min(self.complex_data),
            data_max=jnp.max(self.complex_data),
            data_mean=jnp.mean(self.complex_data),
            mode=2,
        )

    def tearDown(self):
        plt.close("all")
        super().tearDown()

    @patch("matplotlib.pyplot.show")
    def test_plot_after_preprocessing(self, mock_show):
        processed_data = cb.preprocessing(self.complex_data, exponential=True, gblur=2)

        processed_mrc = make_MRC_Image(
            image_data=processed_data,
            voxel_size=self.complex_mrc.voxel_size,
            origin=self.complex_mrc.origin,
            data_min=jnp.min(processed_data),
            data_max=jnp.max(processed_data),
            data_mean=jnp.mean(processed_data),
            mode=self.complex_mrc.mode,
        )

        cb.plot_mrc(processed_mrc, mode="log")

        assert len(plt.get_fignums()) > 0

    @patch("matplotlib.pyplot.show")
    def test_plot_different_image_shapes(self, mock_show):
        shapes = [(50, 100), (100, 50), (200, 200), (10, 300)]

        for shape in shapes:
            plt.close("all")

            data = jnp.ones(shape)
            mrc = make_MRC_Image(
                image_data=data,
                voxel_size=jnp.array([1.0, 1.0, 1.0]),
                origin=jnp.zeros(3),
                data_min=0.0,
                data_max=1.0,
                data_mean=1.0,
                mode=2,
            )

            cb.plot_mrc(mrc)

            assert len(plt.get_fignums()) > 0


if __name__ == "__main__":
    from absl.testing import absltest

    absltest.main()
