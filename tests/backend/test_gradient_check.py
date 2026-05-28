"""Tests for gradient validation utilities."""
import pytest
import torch
from photokan.utils.gradient_check import gradcheck_activation, gradcheck_layer
from photokan.activations import (
    SineEdgeActivation, FourierEdgeActivation,
    SplineEdgeActivation, ReLUEdgeActivation,
)
from photokan.layers import PhotoKANLayer


class TestGradcheckActivation:

    @pytest.mark.parametrize("ActClass,n_basis", [
        (SineEdgeActivation,    4),
        (FourierEdgeActivation, 4),
        (SplineEdgeActivation,  4),
        (ReLUEdgeActivation,    4),
    ])
    def test_gradcheck_passes(self, ActClass, n_basis):
        act    = ActClass(n_basis=n_basis)
        result = gradcheck_activation(act, eps=1e-4, atol=1e-3)
        assert result["passed"], (
            f"{ActClass.__name__} gradcheck failed: {result.get('error', '')}"
        )

    def test_result_has_expected_keys(self):
        act    = SineEdgeActivation(n_basis=4)
        result = gradcheck_activation(act)
        assert "passed" in result
        assert "activation" in result
        assert "n_params" in result


class TestGradcheckLayer:

    def test_sine_layer_gradcheck(self):
        layer  = PhotoKANLayer(2, 2, activation="sine", backend="cpu",
                                noise_sim=False, n_basis=4)
        result = gradcheck_layer(layer, atol=1e-2)
        assert result["passed"], f"Layer gradcheck failed: max_err={result['max_err']:.2e}"

    def test_relu_layer_gradcheck(self):
        layer  = PhotoKANLayer(2, 2, activation="relu", backend="cpu",
                                noise_sim=False, n_basis=4)
        result = gradcheck_layer(layer, atol=1e-2)
        assert result["passed"], f"ReLU layer gradcheck: max_err={result['max_err']:.2e}"

    def test_result_structure(self):
        layer  = PhotoKANLayer(2, 2, activation="sine", backend="cpu",
                                noise_sim=False, n_basis=4)
        result = gradcheck_layer(layer)
        assert "passed" in result
        assert "max_err" in result
        assert "atol" in result
