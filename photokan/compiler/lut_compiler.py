# photokan/compiler/lut_compiler.py
"""
LUT Compiler — converts trained edge activations to int8 lookup tables.

Each activation φ(x) is sampled on a uniform grid, quantised to int8,
and stored as a compact segment-wise table for photonic deployment.
The compiler validates that LUT accuracy stays within tolerance before
accepting the result.
"""

from __future__ import annotations

import numpy as np
import torch

from ..backends.errors import PhotonicCompilerError


class LUTEntry:
    """One compiled LUT: metadata + quantised values."""

    def __init__(
        self,
        values_int8: np.ndarray,
        x_min: float,
        x_max: float,
        scale: float,
        zero_point: int,
        activation_type: str,
        mse_error: float,
    ):
        self.values_int8 = values_int8  # [n_points] int8
        self.x_min = x_min
        self.x_max = x_max
        self.scale = scale  # dequant: y = (q - zp) * scale
        self.zero_point = zero_point
        self.activation_type = activation_type
        self.mse_error = mse_error  # reconstruction MSE

    def dequantise(self) -> np.ndarray:
        """Reconstruct float32 values from int8 LUT."""
        return (self.values_int8.astype(np.float32) - self.zero_point) * self.scale

    def to_dict(self) -> dict:
        return {
            "values_int8": self.values_int8.tolist(),
            "x_min": self.x_min,
            "x_max": self.x_max,
            "scale": self.scale,
            "zero_point": self.zero_point,
            "activation_type": self.activation_type,
            "mse_error": self.mse_error,
        }


class LUTCompiler:
    """
    Converts EdgeActivation instances to quantised LUT entries.

    Args:
        n_points   : Number of LUT entries (default 256).
        x_range    : Input domain (default (-2.0, 2.0)).
        max_mse    : Maximum allowable reconstruction MSE (default 1e-4).
        bit_depth  : Quantisation bit depth (default 8 → int8).
    """

    def __init__(
        self,
        n_points: int = 256,
        x_range: tuple[float, float] = (-2.0, 2.0),
        max_mse: float = 1e-4,
        bit_depth: int = 8,
    ):
        self.n_points = n_points
        self.x_range = x_range
        self.max_mse = max_mse
        self.bit_depth = bit_depth
        self._q_min = -(2 ** (bit_depth - 1))  # -128 for int8
        self._q_max = (2 ** (bit_depth - 1)) - 1  #  127 for int8

    def compile_activation(
        self,
        activation,
        validate: bool = True,
    ) -> LUTEntry:
        """
        Sample, quantise, and optionally validate a single edge activation.

        Args:
            activation: Trained EdgeActivation instance.
            validate  : If True, raise PhotonicCompilerError when MSE > max_mse.

        Returns:
            LUTEntry with int8-quantised lookup table.
        """
        # Sample the activation on a uniform grid
        x = torch.linspace(self.x_range[0], self.x_range[1], self.n_points)
        with torch.no_grad():
            y = activation(x).float()

        y_np = y.numpy()

        # Compute affine quantisation parameters
        y_min = float(y_np.min())
        y_max = float(y_np.max())
        y_range = max(y_max - y_min, 1e-8)

        scale = y_range / (self._q_max - self._q_min)
        zero_point = int(round(self._q_min - y_min / scale))
        zero_point = int(np.clip(zero_point, self._q_min, self._q_max))

        # Quantise
        q = np.round(y_np / scale + zero_point).astype(np.int32)
        q = np.clip(q, self._q_min, self._q_max).astype(np.int8)

        # Reconstruction MSE
        y_recon = (q.astype(np.float32) - zero_point) * scale
        mse = float(np.mean((y_np - y_recon) ** 2))

        if validate and mse > self.max_mse:
            raise PhotonicCompilerError(
                f"LUT MSE {mse:.2e} exceeds max_mse={self.max_mse:.2e} "
                f"for {type(activation).__name__}. "
                f"Increase n_points or relax max_mse."
            )

        return LUTEntry(
            values_int8=q,
            x_min=self.x_range[0],
            x_max=self.x_range[1],
            scale=scale,
            zero_point=zero_point,
            activation_type=type(activation).__name__,
            mse_error=mse,
        )

    def compile_layer(
        self,
        layer,
        validate: bool = True,
    ) -> list[LUTEntry]:
        """
        Compile all edge activations in a PhotoKANLayer.

        Returns:
            List of LUTEntry in edge-index order.
        """
        return [self.compile_activation(act, validate=validate) for act in layer.edge_activations]

    def compile_model(
        self,
        model,
        validate: bool = True,
    ) -> list[list[LUTEntry]]:
        """
        Compile all layers in a PhotoKAN model.

        Returns:
            List of lists: luts[layer_idx][edge_idx].
        """
        return [self.compile_layer(layer, validate=validate) for layer in model.layers]
