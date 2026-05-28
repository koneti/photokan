# photokan/activations/sine_edge.py
"""
Photonic-native sine-basis KAN edge activation.

φ(x) = Σ_k  w_k · sin(f_k · x + p_k)

Maps directly to TFLN waveguide phase response on the Q.ANT NPU.
Proven comparable or superior accuracy to B-spline at a fraction
of the compute cost for periodic/quasi-periodic targets.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import EdgeActivation


class SineEdgeActivation(EdgeActivation):
    """
    Learnable sine basis activation for KAN edges.

    Parameters
    ----------
    n_basis : int
        Number of sine components (default 8).
    freq_init : str
        How to initialise frequencies: 'uniform', 'log', or 'random'.
    trainable_freq : bool
        Whether frequencies are learnable (default True).
    trainable_phase : bool
        Whether phases are learnable (default True).
    """

    def __init__(
        self,
        n_basis: int = 8,
        freq_init: str = "uniform",
        trainable_freq: bool = True,
        trainable_phase: bool = True,
        **kwargs,  # absorb unused kwargs from layer factory
    ):
        super().__init__()
        self.n_basis = n_basis

        # --- frequency initialisation ---
        if freq_init == "uniform":
            freqs = torch.linspace(1.0, float(n_basis), n_basis)
        elif freq_init == "log":
            freqs = torch.logspace(0.0, math.log10(max(n_basis, 2)), n_basis)
        elif freq_init == "random":
            freqs = torch.rand(n_basis) * n_basis + 1.0
        else:
            raise ValueError(
                f"Unknown freq_init '{freq_init}'. Choose 'uniform', 'log', or 'random'."
            )

        self.frequencies = nn.Parameter(freqs, requires_grad=trainable_freq)
        self.phases = nn.Parameter(torch.zeros(n_basis), requires_grad=trainable_phase)
        self.weights = nn.Parameter(torch.randn(n_basis) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [...] arbitrary shape.

        Returns:
            [...] same shape as x.
        """
        x_exp = x.unsqueeze(-1)  # [..., 1]
        basis = torch.sin(self.frequencies * x_exp + self.phases)  # [..., n_basis]
        return (basis * self.weights).sum(dim=-1)  # [...]

    def extra_repr(self) -> str:
        return (
            f"n_basis={self.n_basis}, "
            f"trainable_freq={self.frequencies.requires_grad}, "
            f"trainable_phase={self.phases.requires_grad}"
        )
