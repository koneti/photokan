"""Tests for LUTCompiler."""

import numpy as np
import pytest

from photokan.activations import (
    FourierEdgeActivation,
    ReLUEdgeActivation,
    SineEdgeActivation,
    SplineEdgeActivation,
)
from photokan.backends.errors import PhotonicCompilerError
from photokan.compiler import LUTCompiler, LUTEntry


class TestLUTEntry:
    def _make_entry(self, n=64):
        act = SineEdgeActivation(n_basis=4)
        return LUTCompiler(n_points=n).compile_activation(act, validate=False)

    def test_dequantise_shape(self):
        entry = self._make_entry(n=64)
        recon = entry.dequantise()
        assert recon.shape == (64,)

    def test_dequantise_dtype(self):
        entry = self._make_entry()
        assert entry.dequantise().dtype == np.float32

    def test_values_int8_dtype(self):
        entry = self._make_entry()
        assert entry.values_int8.dtype == np.int8

    def test_to_dict_keys(self):
        entry = self._make_entry()
        d = entry.to_dict()
        for key in (
            "values_int8",
            "x_min",
            "x_max",
            "scale",
            "zero_point",
            "activation_type",
            "mse_error",
        ):
            assert key in d

    def test_mse_non_negative(self):
        entry = self._make_entry()
        assert entry.mse_error >= 0


class TestLUTCompiler:
    @pytest.fixture
    def compiler(self):
        return LUTCompiler(n_points=128, x_range=(-2.0, 2.0), max_mse=1e-2)

    @pytest.mark.parametrize(
        "ActClass",
        [
            SineEdgeActivation,
            FourierEdgeActivation,
            SplineEdgeActivation,
            ReLUEdgeActivation,
        ],
    )
    def test_compile_activation_all_types(self, compiler, ActClass):
        act = ActClass(n_basis=4)
        entry = compiler.compile_activation(act, validate=False)
        assert isinstance(entry, LUTEntry)
        assert entry.values_int8.shape == (128,)

    def test_reconstruction_accuracy(self, compiler):
        """LUT reconstruction should be close to original activation."""
        act = SineEdgeActivation(n_basis=4)
        entry = compiler.compile_activation(act, validate=True)
        assert entry.mse_error < compiler.max_mse

    def test_validation_raises_on_tight_tolerance(self):
        compiler = LUTCompiler(n_points=4, max_mse=1e-20)  # impossibly tight
        act = SineEdgeActivation(n_basis=8)
        with pytest.raises(PhotonicCompilerError):
            compiler.compile_activation(act, validate=True)

    def test_compile_layer(self, compiler):
        from photokan.layers import PhotoKANLayer

        layer = PhotoKANLayer(3, 2, activation="sine", backend="cpu", noise_sim=False, n_basis=4)
        luts = compiler.compile_layer(layer, validate=False)
        assert len(luts) == 6  # 3 * 2 edges

    def test_compile_model(self, compiler):
        from photokan.layers import PhotoKAN

        model = PhotoKAN([2, 4, 1], activation="sine", backend="cpu", noise_sim=False, n_basis=4)
        luts = compiler.compile_model(model, validate=False)
        assert len(luts) == 2  # 2 layers
        assert len(luts[0]) == 8  # 2*4 edges
        assert len(luts[1]) == 4  # 4*1 edges

    def test_scale_and_zp_valid(self, compiler):
        act = ReLUEdgeActivation(n_segments=4)
        entry = compiler.compile_activation(act, validate=False)
        assert np.isfinite(entry.scale)
        assert -128 <= entry.zero_point <= 127
