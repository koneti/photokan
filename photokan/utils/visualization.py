# photokan/utils/visualization.py
"""Visualisation utilities for PhotoKAN models."""

from __future__ import annotations

import numpy as np
import torch


def plot_kan_graph(model):
    """
    Visualise the KAN computation graph with edge function thumbnails.
    """
    import matplotlib.pyplot as plt

    layer_sizes = model.layer_sizes
    n_layers = len(layer_sizes) - 1

    fig_w = max(8, n_layers * 4)
    fig_h = max(4, max(layer_sizes) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.5, n_layers + 0.5)
    ax.set_ylim(-0.5, max(layer_sizes) - 0.5)
    ax.axis("off")
    ax.set_title("PhotoKAN Computation Graph", fontsize=14, fontweight="bold")

    x_np = np.linspace(-2, 2, 50)
    x_t = torch.tensor(x_np, dtype=torch.float32)

    for l, layer in enumerate(model.layers):
        in_n = layer.in_features
        out_n = layer.out_features

        for i in range(in_n):
            for j in range(out_n):
                # Node positions
                x1 = l
                y1 = i + (max(layer_sizes) - in_n) / 2
                x2 = l + 1
                y2 = j + (max(layer_sizes) - out_n) / 2

                activation = layer.edge_activations[layer._edge_idx(i, j)]
                with torch.no_grad():
                    y_vals = activation(x_t).numpy()

                # Draw edge as line
                ax.plot([x1, x2], [y1, y2], "k-", alpha=0.15, linewidth=0.8)

                # Mini sparkline inset
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                y_norm = (y_vals - y_vals.min()) / (np.ptp(y_vals) + 1e-8) * 0.4 - 0.2
                x_norm = np.linspace(mx - 0.2, mx + 0.2, len(y_norm))
                ax.plot(x_norm, my + y_norm, "b-", alpha=0.6, linewidth=1.2)

        # Draw nodes
        for i in range(layer_sizes[l]):
            y_pos = i + (max(layer_sizes) - layer_sizes[l]) / 2
            ax.plot(l, y_pos, "o", markersize=14, color="steelblue", zorder=5)
            ax.text(l, y_pos, str(i), ha="center", va="center", color="white", fontsize=8, zorder=6)

    # Draw output nodes
    for i in range(layer_sizes[-1]):
        y_pos = i + (max(layer_sizes) - layer_sizes[-1]) / 2
        ax.plot(n_layers, y_pos, "o", markersize=14, color="tomato", zorder=5)
        ax.text(
            n_layers, y_pos, str(i), ha="center", va="center", color="white", fontsize=8, zorder=6
        )

    fig.tight_layout()
    return fig


def plot_activation_grid(model, x_range: tuple = (-2.0, 2.0)):
    """Plot all edge activations in a grid."""
    import matplotlib.pyplot as plt

    x_t = torch.linspace(*x_range, 100)
    activations = []
    labels = []

    for l_idx, layer in enumerate(model.layers):
        for j in range(layer.out_features):
            for i in range(layer.in_features):
                act = layer.edge_activations[layer._edge_idx(i, j)]
                with torch.no_grad():
                    y = act(x_t).numpy()
                activations.append(y)
                labels.append(f"L{l_idx}:({i}→{j})")

    n = len(activations)
    cols = min(8, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    x_np = x_t.numpy()
    for ax, y, label in zip(axes, activations, labels):
        ax.plot(x_np, y, linewidth=1.5)
        ax.set_title(label, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)

    for ax in axes[len(activations) :]:
        ax.axis("off")

    fig.suptitle("PhotoKAN Edge Activations", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
