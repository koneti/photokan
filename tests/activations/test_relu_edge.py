"""Tests for ReLUEdgeActivation."""
import torch
import pytest
from photokan.activations import ReLUEdgeActivation


class TestReLUEdgeActivation:

    def test_output_shape(self):
        act = ReLUEdgeActivation(n_segments=4)
        x = torch.randn(20)
        assert act(x).shape == (20,)

    def test_gradient_flows(self):
        act = ReLUEdgeActivation()
        x = torch.randn(8, requires_grad=True)
        act(x).sum().backward()
        assert x.grad is not None

    def test_n_basis_alias(self):
        act = ReLUEdgeActivation(n_basis=6)
        assert act.n_segments == 6

    def test_output_finite(self):
        act = ReLUEdgeActivation()
        x = torch.linspace(-5, 5, 100)
        y = act(x)
        assert torch.isfinite(y).all()
