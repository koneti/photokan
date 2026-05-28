# photokan/layers/photokan.py
"""
Full PhotoKAN model — stacked PhotoKANLayers with training utilities.
"""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn

from .photokan_layer import PhotoKANLayer
from ..backend import available_backends


class PhotoKAN(nn.Module):
    """
    Photonic Kolmogorov-Arnold Network.

    Stacks multiple PhotoKANLayer instances according to layer_sizes.

    Args:
        layer_sizes  : List of integers defining the width of each layer,
                       e.g. [4, 16, 16, 1] → input dim 4, output dim 1.
        activation   : Edge activation type for all layers.
        backend      : Hardware backend ('auto', 'qpal', 'cuda', 'cpu').
        symbolic     : If True, enables post-training symbolic regression.
        **layer_kwargs: Forwarded to each PhotoKANLayer.

    Shape:
        Input:  [batch, layer_sizes[0]]
        Output: [batch, layer_sizes[-1]]
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "sine",
        backend: str = "auto",
        symbolic: bool = False,
        **layer_kwargs: Any,
    ):
        super().__init__()

        if len(layer_sizes) < 2:
            raise ValueError(
                "layer_sizes must have at least 2 elements (input and output)."
            )

        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.backend_mode = backend
        self.symbolic_mode = symbolic

        self.layers = nn.ModuleList([
            PhotoKANLayer(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i + 1],
                activation=activation,
                backend=backend,
                **layer_kwargs,
            )
            for i in range(len(layer_sizes) - 1)
        ])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def parameter_count(self) -> dict:
        """Summarise parameter counts across all layers."""
        total = sum(p.numel() for p in self.parameters())
        edge_total = sum(
            layer.parameter_count()["edge_params"]
            for layer in self.layers
        )
        return {
            "total": total,
            "edge_params": edge_total,
            "n_layers": len(self.layers),
            "layer_sizes": self.layer_sizes,
        }

    def estimate_ops(self) -> dict:
        """
        Rough op-count estimate vs an equivalent MLP.

        Based on published KAN benchmarks: ~43% fewer params,
        ~46% fewer ops vs equivalent MLP with same I/O dimensions.
        """
        total_edges = sum(
            layer.in_features * layer.out_features
            for layer in self.layers
        )
        n_params = sum(p.numel() for p in self.parameters())

        # Equivalent MLP width ≈ average hidden width of this KAN
        hidden_widths = self.layer_sizes[1:-1]
        mlp_hidden = (
            int(sum(hidden_widths) / len(hidden_widths))
            if hidden_widths
            else self.layer_sizes[-1]
        )
        mlp_layers = len(self.layers)
        mlp_params = (
            self.layer_sizes[0] * mlp_hidden
            + (mlp_layers - 1) * mlp_hidden * mlp_hidden
            + mlp_hidden * self.layer_sizes[-1]
        )

        return {
            "n_params": n_params,
            "n_edges": total_edges,
            "mlp_equivalent_params": mlp_params,
            "param_ratio": round(n_params / max(mlp_params, 1), 3),
        }

    def symbolic_regression(self) -> dict[str, Any]:
        """
        Extract symbolic formulas from learned edge activations.

        Returns:
            Dict mapping '(layer, i→j)' strings to sympy expressions
            (or string summaries when sympy is unavailable).
        """
        from ..utils.symbolic import symbolic_regress_activation

        results = {}
        for l_idx, layer in enumerate(self.layers):
            for j in range(layer.out_features):
                for i in range(layer.in_features):
                    edge_idx = layer._edge_idx(i, j)
                    activation = layer.edge_activations[edge_idx]
                    key = f"L{l_idx}:({i}→{j})"
                    results[key] = symbolic_regress_activation(activation)
        return results

    def plot_activations(self, x_range: tuple = (-2.0, 2.0)):
        """
        Plot all learned edge activation functions.

        Returns:
            matplotlib.Figure
        """
        from ..utils.visualization import plot_activation_grid

        return plot_activation_grid(self, x_range=x_range)

    def compile(self, output_path: str, **kwargs):
        """
        Compile this model to a photonic deployment bundle (.npu).

        Delegates to PhotonicCompiler.
        """
        from ..compiler import PhotonicCompiler

        compiler = PhotonicCompiler()
        return compiler.compile(self, output_path, **kwargs)

    def extra_repr(self) -> str:
        return (
            f"layers={self.layer_sizes}, "
            f"activation={self.activation_name}, "
            f"backend={self.backend_mode}"
        )
