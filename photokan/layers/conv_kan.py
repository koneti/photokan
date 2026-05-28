# photokan/layers/conv_kan.py
"""
PhotoConvKAN — Photonic convolutional KAN layer (v1.1).

Extends KAN to 2D feature maps by applying independent photonic edge
activations at each spatial location, then summing across input channels.
Equivalent to a depthwise-separable structure where the channel mixing
uses KAN (nonlinear) rather than linear weights.

Shape:
    Input:  [B, C_in, H, W]
    Output: [B, C_out, H_out, W_out]

Each (C_in, C_out) pair shares one EdgeActivation, applied identically
at every spatial position — the activation parameters are position-invariant,
matching the translational equivariance of standard CNNs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..activations import get_activation_class
from ..backend import apply_edge


class PhotoConvKAN(nn.Module):
    """
    Photonic convolutional KAN layer.

    Args:
        in_channels  : Number of input feature map channels.
        out_channels : Number of output feature map channels.
        kernel_size  : Spatial kernel size (default 3).
        stride       : Convolution stride (default 1).
        padding      : Zero-padding (default 1 for same-size output with k=3).
        activation   : KAN edge activation type.
        backend      : Hardware backend.
        n_basis      : Activation basis size.
        noise_sim    : Enable photonic noise.
        groups       : Channel grouping (1 = standard, in_channels = depthwise).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: str = "sine",
        backend: str = "auto",
        n_basis: int = 8,
        noise_sim: bool = True,
        groups: int = 1,
        **activation_kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.backend_mode = backend

        self.noise_config: dict | None = (
            {"snr_db": 14.0, "bit_depth": 6, "phase_noise_rad": 0.01, "enabled": True}
            if noise_sim
            else {"enabled": False, "snr_db": 14.0, "bit_depth": 6, "phase_noise_rad": 0.01}
        )

        # im2col weight matrix: shape [out_ch, in_ch//groups, kH, kW]
        # We replace the linear weight with KAN activations per (in_ch, out_ch) pair.
        # One activation per (in_ch//groups × kH × kW) input element per out_ch.
        self.in_per_group = in_channels // groups
        n_in_weights = self.in_per_group * kernel_size * kernel_size

        ActivationClass = get_activation_class(activation)
        # [out_channels, n_in_weights] activations
        self.edge_activations = nn.ModuleList(
            [
                ActivationClass(n_basis=n_basis, **activation_kwargs)
                for _ in range(out_channels * n_in_weights)
            ]
        )

        # Bias term per output channel
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W]

        Returns:
            [B, C_out, H_out, W_out]
        """
        B, C_in, H, W = x.shape

        # Unfold input into patches: [B, C_in * kH * kW, L] where L = H_out * W_out
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )  # [B, C_in * kH * kW, L]

        L = patches.shape[-1]
        n_in = self.in_per_group * self.kernel_size * self.kernel_size

        # Reshape for processing: [B*L, n_in]
        # patches: [B, n_in, L] → [B*L, n_in]
        patches = patches.permute(0, 2, 1).reshape(B * L, n_in)  # [B*L, n_in]

        # Apply KAN edge activations for each output channel
        out_channels_list = []
        for oc in range(self.out_channels):
            channel_out = torch.zeros(B * L, device=x.device, dtype=x.dtype)
            for ic in range(n_in):
                act_idx = oc * n_in + ic
                act = self.edge_activations[act_idx]
                phi = apply_edge(
                    patches[:, ic],
                    act,
                    backend_mode=self.backend_mode,
                    noise_config=self.noise_config,
                )
                channel_out = channel_out + phi
            channel_out = channel_out + self.bias[oc]
            out_channels_list.append(channel_out)

        # Stack: [B*L, C_out] → [B, C_out, L]
        out = torch.stack(out_channels_list, dim=1)  # [B*L, C_out]
        out = out.reshape(B, L, self.out_channels).permute(0, 2, 1)  # [B, C_out, L]

        # Fold back to spatial: compute output H/W
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(B, self.out_channels, H_out, W_out)

        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, out={self.out_channels}, "
            f"k={self.kernel_size}, stride={self.stride}, "
            f"backend={self.backend_mode}"
        )
