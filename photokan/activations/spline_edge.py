# photokan/activations/spline_edge.py
"""
B-spline KAN edge activation — the original KAN formulation.

φ(x) = Σ_i  c_i · B_i^k(x)

Highest precision for non-periodic functions; slower than Sine/Fourier.
Compilable to int8 LUT for photonic deployment.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import EdgeActivation


def _b_spline_basis(
    x: torch.Tensor,
    grid: torch.Tensor,
    order: int,
) -> torch.Tensor:
    """
    Compute B-spline basis functions B_i^k(x) via the Cox–de Boor recursion.

    Args:
        x    : [...] input values (clamped to grid range by caller).
        grid : [n_knots] knot positions (uniform).
        order: spline order k.

    Returns:
        [..., n_basis] where n_basis = n_knots - order - 1.
    """
    x = x.unsqueeze(-1)  # [..., 1]

    # Order-0 (piecewise constant) basis
    basis = (x >= grid[:-1]) & (x < grid[1:])  # [..., n_knots-1]
    basis = basis.float()

    # Clamp last point to include right boundary
    last_interval = x == grid[-1]
    basis[..., -1] = basis[..., -1] + last_interval.squeeze(-1).float()

    # Cox-de Boor recursion
    for k in range(1, order + 1):
        n = basis.shape[-1] - 1  # number of order-k bases
        left_num = x - grid[:n]  # [..., n]
        left_den = grid[k : k + n] - grid[:n]
        right_num = grid[k + 1 : k + 1 + n] - x
        right_den = grid[k + 1 : k + 1 + n] - grid[1 : n + 1]

        # Safe division (avoid 0/0 → 0)
        left = torch.where(left_den != 0, left_num / left_den, torch.zeros_like(left_num))
        right = torch.where(right_den != 0, right_num / right_den, torch.zeros_like(right_num))

        basis = left * basis[..., :n] + right * basis[..., 1:]

    return basis  # [..., n_basis]


class SplineEdgeActivation(EdgeActivation):
    """
    B-spline KAN edge activation.

    Parameters
    ----------
    grid_size : int
        Number of interior knot intervals (default 5).
    spline_order : int
        Polynomial order k (default 3 → cubic).
    grid_range : tuple
        (min, max) input domain (default (-1, 1)).
    """

    def __init__(
        self,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        n_basis: int | None = None,  # alias: overrides grid_size if set
        **kwargs,
    ):
        super().__init__()
        if n_basis is not None:
            grid_size = n_basis

        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range

        # Uniform knot grid with extended boundary knots
        n_knots = grid_size + 2 * spline_order + 1
        grid = torch.linspace(grid_range[0], grid_range[1], n_knots)
        self.register_buffer("grid", grid)

        # Number of basis functions
        n_coeff = grid_size + spline_order
        self.coefficients = nn.Parameter(torch.randn(n_coeff) * 0.1)

        # Optional residual: w·SiLU(x) for stability (like pykan)
        self.residual_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [...] arbitrary shape.

        Returns:
            [...] same shape.
        """
        # Clamp to grid range
        x_clamped = x.clamp(self.grid_range[0], self.grid_range[1])

        # Evaluate B-spline basis
        basis = _b_spline_basis(x_clamped, self.grid, self.spline_order)
        # [..., n_coeff]

        spline_out = (basis * self.coefficients).sum(dim=-1)
        residual = self.residual_weight * torch.nn.functional.silu(x)

        return spline_out + residual

    def extra_repr(self) -> str:
        return f"grid_size={self.grid_size}, order={self.spline_order}, range={self.grid_range}"
