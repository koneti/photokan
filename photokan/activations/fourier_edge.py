# photokan/activations/fourier_edge.py
"""
Full Fourier-basis KAN edge activation.

φ(x) = a_0 + Σ_k [a_k · cos(k·x/T) + b_k · sin(k·x/T)]

More expressive than SineEdge; best for periodic or multi-frequency
targets. Slightly higher parameter count.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from .base import EdgeActivation


class FourierEdgeActivation(EdgeActivation):
    """
    Learnable Fourier basis activation for KAN edges.

    Parameters
    ----------
    n_freqs : int
        Number of Fourier modes (default 8).
    period : float
        Fundamental period T (default 2π).
    """

    def __init__(
        self,
        n_freqs: int = 8,
        period: float = 2 * math.pi,
        n_basis: int | None = None,   # alias so layer factory works uniformly
        **kwargs,
    ):
        super().__init__()
        # Allow n_basis as an alias for n_freqs
        if n_basis is not None and n_freqs == 8:
            n_freqs = n_basis
        self.n_freqs = n_freqs
        self.period = period

        # a_0 (DC), a_k (cosine), b_k (sine)
        self.a0 = nn.Parameter(torch.zeros(1))
        self.a = nn.Parameter(torch.randn(n_freqs) * 0.1)  # cosine coeffs
        self.b = nn.Parameter(torch.randn(n_freqs) * 0.1)  # sine coeffs

        # Pre-compute integer frequency indices (not trained)
        self.register_buffer(
            "k", torch.arange(1, n_freqs + 1, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [...] arbitrary shape.

        Returns:
            [...] same shape.
        """
        # Angular frequency: 2π·k / T
        omega = 2 * math.pi * self.k / self.period         # [n_freqs]
        x_exp = x.unsqueeze(-1)                            # [..., 1]
        arg = omega * x_exp                                # [..., n_freqs]
        out = (
            self.a0
            + (self.a * torch.cos(arg)).sum(dim=-1)
            + (self.b * torch.sin(arg)).sum(dim=-1)
        )
        return out

    def extra_repr(self) -> str:
        return f"n_freqs={self.n_freqs}, period={self.period:.4f}"
