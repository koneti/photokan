"""Tests for PhotoKANAttention."""
import pytest
import torch
from photokan.llm.attention import PhotoKANAttention


class TestPhotoKANAttention:

    @pytest.fixture
    def attn(self):
        return PhotoKANAttention(
            d_model=16, n_heads=2, activation="relu",
            backend="cpu", n_basis=4, noise_sim=False,
        )

    def test_output_shape(self, attn):
        x = torch.randn(2, 5, 16)
        y = attn(x)
        assert y.shape == (2, 5, 16)

    def test_with_causal_mask(self, attn):
        T = 5
        x    = torch.randn(2, T, 16)
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        y    = attn(x, attn_mask=mask)
        assert y.shape == (2, T, 16)

    def test_gradient_flows(self, attn):
        x = torch.randn(2, 3, 16, requires_grad=True)
        attn(x).sum().backward()
        assert x.grad is not None

    def test_key_padding_mask(self, attn):
        x    = torch.randn(2, 4, 16)
        mask = torch.tensor([[False, False, True, True],
                              [False, True,  True, True]])
        y = attn(x, key_padding_mask=mask)
        assert y.shape == (2, 4, 16)
        assert torch.isfinite(y).all()
