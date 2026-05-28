"""Tests for PhotoKANLayer."""

import pytest
import torch

from photokan.layers import PhotoKANLayer


class TestPhotoKANLayer:
    @pytest.mark.parametrize("activation", ["sine", "fourier", "spline", "relu"])
    def test_output_shape(self, activation):
        layer = PhotoKANLayer(
            4, 3, activation=activation, backend="cpu", noise_sim=False, n_basis=4
        )
        x = torch.randn(8, 4)
        y = layer(x)
        assert y.shape == (8, 3)

    def test_gradient_flows(self):
        layer = PhotoKANLayer(3, 2, activation="sine", backend="cpu", noise_sim=False, n_basis=4)
        x = torch.randn(5, 3, requires_grad=True)
        y = layer(x).sum()
        y.backward()
        assert x.grad is not None

    def test_parameter_count(self):
        layer = PhotoKANLayer(2, 3, activation="sine", n_basis=4, backend="cpu", noise_sim=False)
        counts = layer.parameter_count()
        assert counts["n_edges"] == 6  # 2*3
        assert counts["edge_params"] > 0

    def test_wrong_backend_raises(self):
        """Requesting qpal when NPU absent should raise PhotonicBackendError."""
        from photokan.backends import PhotonicBackendError

        layer = PhotoKANLayer(2, 2, activation="sine", backend="qpal", noise_sim=False, n_basis=4)
        x = torch.randn(4, 2)
        with pytest.raises((PhotonicBackendError, RuntimeError)):
            layer(x)

    def test_noise_sim_enabled(self):
        layer_noisy = PhotoKANLayer(
            2, 2, activation="sine", backend="cpu", noise_sim=True, n_basis=4
        )
        layer_clean = PhotoKANLayer(
            2, 2, activation="sine", backend="cpu", noise_sim=False, n_basis=4
        )
        # Copy weights so only noise differs
        layer_clean.load_state_dict(layer_noisy.state_dict())
        x = torch.randn(32, 2)
        # Noisy and clean outputs may differ; just check shapes
        assert layer_noisy(x).shape == (32, 2)
        assert layer_clean(x).shape == (32, 2)
