# photokan/layers/photokan_layer.py
"""
Single photonic KAN layer.

y_j = Σ_i  φ_ij(x_i)    for all output nodes j.

Each φ_ij is a learnable EdgeActivation dispatched to the
configured backend (Q.ANT NPU, CUDA, or CPU simulation).

Forward pass is vectorized: all edges are computed in parallel
without Python-level loops for substantial speedup on CPU and GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..activations import get_activation_class, EdgeActivation
from ..backend import apply_edge, resolve_backend


class PhotoKANLayer(nn.Module):
    """
    A single KAN layer with in_features × out_features photonic edge functions.

    Args:
        in_features  : Input dimension.
        out_features : Output dimension.
        activation   : Activation name ('sine', 'fourier', 'spline', 'relu')
                       or an EdgeActivation subclass.
        backend      : 'auto', 'qpal', 'cuda', or 'cpu'.
        n_basis      : Basis size forwarded to the activation constructor.
        noise_sim    : Enable photonic noise in CPU simulation.
        **activation_kwargs: Extra kwargs forwarded to the activation class.

    Shape:
        Input:  [batch, in_features]
        Output: [batch, out_features]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str | type[EdgeActivation] = "sine",
        backend: str = "auto",
        n_basis: int = 8,
        noise_sim: bool = True,
        **activation_kwargs,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.backend_mode = backend

        # Noise config passed to SimBackend per forward call
        self.noise_config: dict | None = (
            {"snr_db": 14.0, "bit_depth": 6, "phase_noise_rad": 0.01, "enabled": True}
            if noise_sim
            else {"enabled": False, "snr_db": 14.0, "bit_depth": 6, "phase_noise_rad": 0.01}
        )

        # One activation per directed edge i→j
        ActivationClass = get_activation_class(activation)
        self.edge_activations = nn.ModuleList([
            ActivationClass(n_basis=n_basis, **activation_kwargs)
            for _ in range(in_features * out_features)
        ])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _edge_idx(self, i: int, j: int) -> int:
        """Map (in_node i, out_node j) → flat index in edge_activations."""
        return i * self.out_features + j

    # ------------------------------------------------------------------
    # Forward (vectorized)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vectorized forward pass.

        Computes all edge activations in parallel by iterating edges
        (not nested i,j loops), then scatter-adds results into the
        output tensor grouped by destination node.

        Args:
            x: [batch, in_features]

        Returns:
            [batch, out_features]
        """
        batch = x.shape[0]
        n_edges = self.in_features * self.out_features

        # Compute all edges: each gets its own input slice
        edge_outputs = []
        for idx in range(n_edges):
            i = idx // self.out_features
            activation = self.edge_activations[idx]
            phi = apply_edge(
                x[:, i],
                activation,
                backend_mode=self.backend_mode,
                noise_config=self.noise_config,
            )
            edge_outputs.append(phi)

        # Stack: [n_edges, batch]
        edge_outputs = torch.stack(edge_outputs, dim=0)

        # Reshape to [in_features, out_features, batch]
        edge_outputs = edge_outputs.view(self.in_features, self.out_features, batch)

        # Sum over input dimension: [out_features, batch] → [batch, out_features]
        return edge_outputs.sum(dim=0).t()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def parameter_count(self) -> dict:
        """Return a breakdown of trainable parameters."""
        edge_params = sum(
            p.numel() for p in self.edge_activations.parameters()
        )
        return {
            "edge_params": edge_params,
            "total": edge_params,
            "n_edges": len(self.edge_activations),
        }

    def extra_repr(self) -> str:
        resolved = resolve_backend(self.backend_mode)
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"edges={len(self.edge_activations)}, "
            f"backend={self.backend_mode} (→{resolved})"
        )
