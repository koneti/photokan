"""Tests for SimBackend and NoiseModel — updated for 0.2.0 API."""
import torch
import pytest
from photokan.backend import SimBackend, NoiseModel
from photokan.activations import SineEdgeActivation


class TestNoiseModel:

    def test_noise_changes_output(self):
        nm = NoiseModel(snr_db=6.0, bit_depth=4, enabled=True)
        x = torch.linspace(-1, 1, 64)
        noisy = nm.apply(x.clone())
        assert not torch.allclose(x, noisy, atol=1e-3)

    def test_disabled_noise_is_identity(self):
        nm = NoiseModel(enabled=False)
        x = torch.randn(32)
        assert torch.allclose(nm.apply(x), x)

    def test_output_shape_preserved(self):
        nm = NoiseModel()
        x = torch.randn(4, 8)
        assert nm.apply(x).shape == (4, 8)


class TestSimBackend:

    def test_forward_shape(self):
        act = SineEdgeActivation(n_basis=4)
        x = torch.randn(16)
        y = SimBackend.forward(x, act)
        assert y.shape == (16,)

    def test_forward_without_noise_matches_activation(self):
        act = SineEdgeActivation(n_basis=4)
        x = torch.randn(16)
        y_sim = SimBackend.forward(x, act, noise_config={"enabled": False,
                                                          "snr_db": 14.0,
                                                          "bit_depth": 6,
                                                          "phase_noise_rad": 0.0})
        y_act = act(x)
        assert torch.allclose(y_sim, y_act, atol=1e-5)

    def test_hardware_profile_npu1(self):
        result = SimBackend.set_hardware_profile("npu1")
        assert isinstance(result, dict)

    def test_hardware_profile_npu2(self):
        result = SimBackend.set_hardware_profile("npu2")
        assert isinstance(result, dict)

    def test_hardware_profile_ideal(self):
        result = SimBackend.set_hardware_profile("ideal")
        assert isinstance(result, dict)

    def test_invalid_profile_raises(self):
        with pytest.raises((ValueError, KeyError)):
            SimBackend.set_hardware_profile("nonexistent")

    def test_get_transfer_function(self):
        act = SineEdgeActivation(n_basis=4)
        x, y_ideal, y_noisy = SimBackend.get_transfer_function(act, n_points=32)
        assert x.shape == (32,)
        assert y_ideal.shape == (32,)
        assert y_noisy.shape == (32,)
