"""Tests for visualisation utilities."""

import pytest
import torch

from photokan.layers import PhotoKAN
from photokan.utils.visualization import plot_kan_graph


class TestPlotKanGraph:
    @pytest.fixture()
    def small_model(self):
        torch.manual_seed(42)
        return PhotoKAN(
            layer_sizes=[2, 4, 1],
            activation="sine",
            backend="cpu",
            noise_sim=False,
            n_basis=4,
        )

    def test_returns_matplotlib_figure(self, small_model):
        import matplotlib.figure

        fig = plot_kan_graph(small_model)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_figure_has_axes(self, small_model):
        fig = plot_kan_graph(small_model)
        assert len(fig.get_axes()) > 0
