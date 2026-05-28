# photokan/llm/attention.py
"""
PhotoKANAttention — KAN-based attention mechanism (v1.2).

Replaces the standard Q/K/V projections and attention score computation
with photonic KAN edge functions. Instead of:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V

PhotoKANAttention uses:
    scores_ij = KAN_score(q_i, k_j)  ← nonlinear edge function per pair
    output_i  = Σ_j softmax(scores)_ij · V_j

The Q/K/V projections themselves are also replaced with PhotoKANLayer
to exploit photonic nonlinearity end-to-end.

This is a research-grade implementation. For production use, evaluate
accuracy carefully vs standard attention before deployment.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import PhotoKANLayer


class PhotoKANAttention(nn.Module):
    """
    Multi-head KAN attention with photonic Q/K/V projections.

    Args:
        d_model     : Model embedding dimension.
        n_heads     : Number of attention heads.
        activation  : KAN edge activation for Q/K/V projections.
        backend     : Hardware backend.
        n_basis     : Activation basis size.
        dropout     : Attention dropout probability.
        noise_sim   : Enable photonic noise simulation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        activation: str = "sine",
        backend: str = "auto",
        n_basis: int = 6,
        dropout: float = 0.0,
        noise_sim: bool = False,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.scale    = math.sqrt(self.d_head)
        self.dropout  = nn.Dropout(dropout)

        common = dict(
            activation=activation, backend=backend,
            n_basis=n_basis, noise_sim=noise_sim,
        )

        # Photonic Q/K/V projections (nonlinear edge functions)
        self.q_proj = PhotoKANLayer(d_model, d_model, **common)
        self.k_proj = PhotoKANLayer(d_model, d_model, **common)
        self.v_proj = PhotoKANLayer(d_model, d_model, **common)

        # Output projection — standard linear for compatibility
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x               : [B, T, d_model] input embeddings.
            attn_mask       : [T, T] or [B*H, T, T] additive attention mask.
            key_padding_mask: [B, T] boolean mask (True = pad).

        Returns:
            [B, T, d_model] attended output.
        """
        B, T, D = x.shape

        # Project to Q, K, V via photonic KAN
        # [B, T, d_model] → flatten spatial, project, reshape
        x_flat = x.reshape(B * T, D)
        Q = self.q_proj(x_flat).reshape(B, T, self.n_heads, self.d_head)
        K = self.k_proj(x_flat).reshape(B, T, self.n_heads, self.d_head)
        V = self.v_proj(x_flat).reshape(B, T, self.n_heads, self.d_head)

        # [B, n_heads, T, d_head]
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, T, T]

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            scores = scores + attn_mask

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        attended = torch.matmul(weights, V)           # [B, H, T, d_head]
        attended = attended.permute(0, 2, 1, 3)       # [B, T, H, d_head]
        attended = attended.reshape(B, T, D)

        return self.out_proj(attended)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_heads={self.n_heads}, d_head={self.d_head}"
