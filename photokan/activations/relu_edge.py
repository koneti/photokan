# photokan/activations/relu_edge.py
"""
ReLU-based KAN edge activation — fastest variant.

φ(x) = Σ_s  w_s · ReLU(a_s · x + b_s)

Piecewise-linear approximation of arbitrary functions.
~20× faster than spline; lower expressiveness but hardware-efficient
on all backends. Suitable for edge inference and quantisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import EdgeActivation


class ReLUEdgeActivation(EdgeActivation):
    """
    Piecewise-linear KAN edge activation via learnable ReLU segments.

    Parameters
    ----------
    n_segments : int
        Number of piecewise segments (default 4).
        Total params = 3 × n_segments (w, a, b per segment).
    """

    def __init__(
        self,
        n_segments: int = 4,
        n_basis: int | None = None,  # alias for n_segments
        **kwargs,
    ):
        super().__init__()
        if n_basis is not None:
            n_segments = n_basis
        self.n_segments = n_segments

        # Per-segment: output weight, input scale, input bias
        self.w = nn.Parameter(torch.randn(n_segments) * 0.1)
        self.a = nn.Parameter(torch.ones(n_segments))
        self.b = nn.Parameter(torch.linspace(-1.0, 1.0, n_segments))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [...] arbitrary shape.

        Returns:
            [...] same shape.
        """
        x_exp = x.unsqueeze(-1)  # [..., 1]
        segments = torch.relu(self.a * x_exp + self.b)  # [..., n_segments]
        return (self.w * segments).sum(dim=-1)  # [...]

    def extra_repr(self) -> str:
        return f"n_segments={self.n_segments}"
