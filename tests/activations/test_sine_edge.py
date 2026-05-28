"""Tests for SineEdgeActivation."""
import math
import pytest
import torch
from photokan.activations import SineEdgeActivation


class TestSineEdgeActivation:

    def test_output_shape_1d(self):
        act = SineEdgeActivation(n_basis=8)
        x = torch.randn(16)
        y = act(x)
        assert y.shape == x.shape

    def test_output_shape_2d(self):
        act = SineEdgeActivation(n_basis=4)
        x = torch.randn(8, 4)
        y = act(x)
        assert y.shape == x.shape

    def test_gradient_flows(self):
        act = SineEdgeActivation(n_basis=4)
        x = torch.randn(8, requires_grad=True)
        y = act(x).sum()
        y.backward()
        assert x.grad is not None
        assert act.weights.grad is not None

    @pytest.mark.parametrize("freq_init", ["uniform", "log", "random"])
    def test_freq_init(self, freq_init):
        act = SineEdgeActivation(n_basis=6, freq_init=freq_init)
        assert act.frequencies.shape == (6,)
        assert not torch.isnan(act.frequencies).any()

    def test_frozen_freq(self):
        act = SineEdgeActivation(trainable_freq=False)
        assert not act.frequencies.requires_grad

    def test_to_lut(self):
        act = SineEdgeActivation(n_basis=4)
        lut = act.to_lut(n_points=64, x_range=(-1.0, 1.0))
        assert lut["x"].shape == (64,)
        assert lut["y"].shape == (64,)
        assert lut["activation"] == "SineEdgeActivation"

    def test_invalid_freq_init(self):
        with pytest.raises(ValueError):
            SineEdgeActivation(freq_init="bogus")
