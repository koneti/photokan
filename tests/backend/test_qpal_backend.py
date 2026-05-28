"""
Tests for QANTBackend (formerly QPALBackend) — Phase 2.
NPU-dependent tests are skipped when hardware is unavailable.
"""

import pytest
import torch

from photokan.activations import ReLUEdgeActivation, SineEdgeActivation
from photokan.backends.errors import PhotonicBackendError
from photokan.backends.qant.backend import QANTBackend, QPALDeviceInfo, _op_type


class TestQPALDeviceInfo:
    def test_unavailable_info(self):
        info = QANTBackend.get_device_info()
        assert isinstance(info, QPALDeviceInfo)
        if not QANTBackend.is_available():
            assert info.available is False

    def test_str_not_available(self):
        info = QPALDeviceInfo(available=False)
        assert "not available" in str(info)


class TestOpRegistry:
    def test_known_activations(self):
        for ActClass, expected_op in [
            (SineEdgeActivation, "sine_waveguide"),
            (ReLUEdgeActivation, "relu_piecewise"),
        ]:
            act = ActClass(n_basis=4)
            assert _op_type(act) == expected_op

    def test_unknown_activation_raises(self):
        from photokan.activations.base import EdgeActivation

        class UnknownAct(EdgeActivation):
            def forward(self, x):
                return x

        with pytest.raises(PhotonicBackendError, match="No Q.PAL op registered"):
            _op_type(UnknownAct())


@pytest.mark.npu
class TestQANTBackendNPU:
    """These tests only run when Q.ANT NPU hardware is available."""

    @pytest.fixture(autouse=True)
    def require_npu(self):
        if not QANTBackend.is_available():
            pytest.skip("Q.ANT NPU not available")

    def test_nonlinear_forward_shape(self):
        act = SineEdgeActivation(n_basis=6)
        x = torch.randn(16)
        y = QANTBackend.nonlinear_forward(x, act)
        assert y.shape == (16,)

    def test_gradient_matches_autograd(self):
        act = SineEdgeActivation(n_basis=4)
        x = torch.randn(8)
        result = QANTBackend.validate_gradient(act, x)
        assert result["passed"], f"Adjoint gradient mismatch: max_abs={result['max_abs_err']:.2e}"

    def test_estimate_flops_npu(self):
        from photokan.layers import PhotoKANLayer

        layer = PhotoKANLayer(4, 4, activation="sine", backend="qant", n_basis=4)
        flops = QANTBackend.estimate_flops(layer)
        assert "total_ops" in flops


class TestQANTBackendSoftwareFallback:
    """Software fallback for estimate_flops when NPU unavailable."""

    def test_estimate_flops_returns_dict(self):
        from photokan.layers import PhotoKANLayer

        layer = PhotoKANLayer(3, 3, activation="sine", backend="cpu", noise_sim=False, n_basis=4)
        flops = QANTBackend.estimate_flops(layer)
        assert "total_ops" in flops
        assert flops["n_edges"] == 9

    def test_validate_gradient_skips_gracefully(self):
        if QANTBackend.is_available():
            pytest.skip("NPU available — would run actual validation")
        act = SineEdgeActivation(n_basis=4)
        result = QANTBackend.validate_gradient(act, torch.randn(4))
        assert result["passed"] is None
        assert "skipped" in result["reason"]
