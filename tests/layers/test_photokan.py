"""Tests for PhotoKAN full model."""

import pytest
import torch

from photokan.layers import PhotoKAN


class TestPhotoKAN:
    def _make_model(self, **kwargs):
        return PhotoKAN(
            layer_sizes=[4, 8, 4, 1],
            activation="sine",
            backend="cpu",
            noise_sim=False,
            n_basis=4,
            **kwargs,
        )

    def test_output_shape(self):
        model = self._make_model()
        x = torch.randn(16, 4)
        y = model(x)
        assert y.shape == (16, 1)

    def test_single_hidden_layer(self):
        model = PhotoKAN([2, 4, 1], backend="cpu", noise_sim=False, n_basis=4)
        x = torch.randn(8, 2)
        assert model(x).shape == (8, 1)

    def test_too_short_layer_sizes(self):
        with pytest.raises(ValueError):
            PhotoKAN([4])

    def test_gradient_flows(self):
        model = self._make_model()
        x = torch.randn(8, 4, requires_grad=True)
        model(x).sum().backward()
        assert x.grad is not None

    def test_parameter_count(self):
        model = self._make_model()
        counts = model.parameter_count()
        assert "total" in counts
        assert counts["n_layers"] == 3
        assert counts["total"] > 0

    def test_estimate_ops(self):
        model = self._make_model()
        ops = model.estimate_ops()
        assert "n_params" in ops
        assert "mlp_equivalent_params" in ops

    def test_standard_pytorch_training_step(self):
        """A full gradient step should work without errors."""
        model = self._make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(16, 4)
        target = torch.randn(16, 1)
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert loss.item() > 0

    def test_compose_with_sequential(self):
        """PhotoKANLayer should compose with nn.Sequential."""
        import torch.nn as nn

        from photokan.layers import PhotoKANLayer

        net = nn.Sequential(
            PhotoKANLayer(4, 8, backend="cpu", noise_sim=False, n_basis=4),
            PhotoKANLayer(8, 1, backend="cpu", noise_sim=False, n_basis=4),
        )
        x = torch.randn(5, 4)
        assert net(x).shape == (5, 1)
