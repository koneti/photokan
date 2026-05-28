# photokan/activations/base.py
"""Abstract base class for all KAN edge activation functions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class EdgeActivation(ABC, nn.Module):
    """
    Base class for KAN edge activation functions.

    Each subclass represents a learnable nonlinear function φ(x)
    placed on a single KAN edge. Photonic-native variants map
    directly to Q.ANT NPU waveguide operations.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute φ(x).

        Args:
            x: Input tensor of any shape.

        Returns:
            Output tensor of the same shape as x.
        """
        ...

    def to_lut(
        self,
        n_points: int = 256,
        x_range: tuple[float, float] = (-1.0, 1.0),
    ) -> dict:
        """
        Pre-compute a lookup table for AOT compilation.

        Returns a dict with keys: x, y, x_range, n_points,
        activation (class name), params (all param tensors).
        """
        x = torch.linspace(*x_range, n_points)
        with torch.no_grad():
            y = self.forward(x)
        return {
            "x": x.numpy(),
            "y": y.numpy(),
            "x_range": x_range,
            "n_points": n_points,
            "activation": type(self).__name__,
            "params": {k: v.detach().numpy() for k, v in self.named_parameters()},
        }

    def extra_repr(self) -> str:
        return f"n_params={sum(p.numel() for p in self.parameters())}"
