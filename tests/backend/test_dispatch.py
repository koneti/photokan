"""Tests for the edge dispatch layer."""

import pytest
import torch

from photokan.activations import SineEdgeActivation
from photokan.backend.dispatch import apply_edge
from photokan.backends.errors import PhotonicBackendError


class TestApplyEdge:
    @pytest.fixture()
    def activation(self):
        torch.manual_seed(42)
        return SineEdgeActivation(n_basis=4)

    @pytest.fixture()
    def input_tensor(self):
        torch.manual_seed(0)
        return torch.randn(8)

    def test_cpu_backend_returns_correct_shape(self, activation, input_tensor):
        out = apply_edge(input_tensor, activation, backend_mode="cpu")
        assert out.shape == input_tensor.shape

    def test_noise_disabled_matches_raw_activation(self, activation, input_tensor):
        noise_config = {"enabled": False, "snr_db": 14.0, "bit_depth": 6, "phase_noise_rad": 0.0}
        out_dispatch = apply_edge(
            input_tensor, activation, backend_mode="cpu", noise_config=noise_config
        )
        out_raw = activation(input_tensor)
        assert torch.allclose(out_dispatch, out_raw, atol=1e-5)

    def test_unavailable_vendor_raises_backend_error(self, activation, input_tensor):
        # "lightmatter" is registered but hardware is not present
        with pytest.raises(PhotonicBackendError, match="hardware is not available"):
            apply_edge(input_tensor, activation, backend_mode="lightmatter")

    def test_deprecated_qpal_name_raises_appropriate_error(self, activation, input_tensor):
        with pytest.raises(PhotonicBackendError, match="deprecated"):
            apply_edge(input_tensor, activation, backend_mode="qpal")

    def test_auto_backend_falls_back_to_cpu(self, activation, input_tensor):
        out = apply_edge(input_tensor, activation, backend_mode="auto")
        assert out.shape == input_tensor.shape

    def test_gradients_flow_through_dispatch(self, activation):
        x = torch.randn(4, requires_grad=True)
        out = apply_edge(x, activation, backend_mode="cpu")
        out.sum().backward()
        assert x.grad is not None
