"""Tests for SplineEdgeActivation."""
import torch
import pytest
from photokan.activations import SplineEdgeActivation


class TestSplineEdgeActivation:

    def test_output_shape(self):
        act = SplineEdgeActivation(grid_size=5, spline_order=3)
        x = torch.randn(16)
        assert act(x).shape == (16,)

    def test_2d_input(self):
        act = SplineEdgeActivation()
        x = torch.randn(8, 5)
        assert act(x).shape == (8, 5)

    def test_gradient_flows(self):
        act = SplineEdgeActivation()
        x = torch.randn(8, requires_grad=True)
        act(x).sum().backward()
        assert x.grad is not None
        assert act.coefficients.grad is not None

    def test_clamping(self):
        """Values outside grid_range should not raise."""
        act = SplineEdgeActivation(grid_range=(-1.0, 1.0))
        x = torch.tensor([-5.0, 0.0, 5.0])
        y = act(x)
        assert not torch.isnan(y).any()

    def test_n_basis_alias(self):
        act = SplineEdgeActivation(n_basis=7)
        assert act.grid_size == 7

    def test_lut_output(self):
        act = SplineEdgeActivation(grid_size=3)
        lut = act.to_lut(n_points=32)
        assert lut["y"].shape == (32,)
