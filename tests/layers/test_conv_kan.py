"""Tests for PhotoConvKAN."""
import pytest
import torch
from photokan.layers import PhotoConvKAN


class TestPhotoConvKAN:

    def test_output_shape_same_padding(self):
        layer = PhotoConvKAN(2, 4, kernel_size=3, padding=1, n_basis=3,
                              activation="relu", backend="cpu", noise_sim=False)
        x = torch.randn(2, 2, 8, 8)
        y = layer(x)
        assert y.shape == (2, 4, 8, 8)

    def test_output_shape_no_padding(self):
        layer = PhotoConvKAN(1, 2, kernel_size=3, padding=0, n_basis=3,
                              activation="relu", backend="cpu", noise_sim=False)
        x = torch.randn(2, 1, 10, 10)
        y = layer(x)
        assert y.shape == (2, 2, 8, 8)

    def test_output_shape_stride2(self):
        layer = PhotoConvKAN(2, 4, kernel_size=3, stride=2, padding=1, n_basis=3,
                              activation="relu", backend="cpu", noise_sim=False)
        x = torch.randn(2, 2, 8, 8)
        y = layer(x)
        assert y.shape == (2, 4, 4, 4)

    def test_gradient_flows(self):
        layer = PhotoConvKAN(1, 2, kernel_size=3, padding=1, n_basis=3,
                              activation="relu", backend="cpu", noise_sim=False)
        x = torch.randn(2, 1, 6, 6, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None

    def test_output_finite(self):
        layer = PhotoConvKAN(2, 2, kernel_size=3, padding=1, n_basis=3,
                              activation="sine", backend="cpu", noise_sim=False)
        x = torch.randn(2, 2, 8, 8)
        assert torch.isfinite(layer(x)).all()
