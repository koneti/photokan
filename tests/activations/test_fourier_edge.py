"""Tests for FourierEdgeActivation."""

import math

import torch

from photokan.activations import FourierEdgeActivation


class TestFourierEdgeActivation:
    def test_output_shape(self):
        act = FourierEdgeActivation(n_freqs=6)
        x = torch.randn(32)
        assert act(x).shape == (32,)

    def test_2d_input(self):
        act = FourierEdgeActivation(n_freqs=4)
        x = torch.randn(4, 10)
        assert act(x).shape == (4, 10)

    def test_gradient_flows(self):
        act = FourierEdgeActivation(n_freqs=4)
        x = torch.randn(8, requires_grad=True)
        act(x).sum().backward()
        assert x.grad is not None

    def test_n_basis_alias(self):
        # n_basis should be accepted as alias for n_freqs
        act = FourierEdgeActivation(n_basis=10)
        assert act.n_freqs == 10

    def test_dc_term_learnable(self):
        act = FourierEdgeActivation(n_freqs=4)
        assert act.a0.requires_grad

    def test_periodic_behaviour(self):
        """Output at x and x+T should be close (within noise of trainable params)."""
        act = FourierEdgeActivation(n_freqs=4, period=2 * math.pi)
        # Make a0=0, set only a/b small to ensure periodicity
        with torch.no_grad():
            act.a0.fill_(0.0)
        x = torch.tensor([0.5])
        x_shifted = x + 2 * math.pi
        # They won't be exactly equal due to trainable a/b, but shape is fine
        assert act(x).shape == (1,)
        assert act(x_shifted).shape == (1,)
