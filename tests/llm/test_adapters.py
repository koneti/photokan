"""Tests for PhotoLoRA adapters."""
import pytest
import torch
import torch.nn as nn
import photokan.llm as pkl
from photokan.llm.adapters import PhotoLoRALinear


class _TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16)
        self.v_proj = nn.Linear(16, 16)
        self.ffn    = nn.Linear(16, 8)
    def forward(self, x):
        return self.ffn(self.q_proj(x) + self.v_proj(x))


class TestPhotoLoRALinear:

    def test_output_shape(self):
        base  = nn.Linear(8, 4)
        lora  = PhotoLoRALinear(base, rank=2, n_basis=4, noise_sim=False)
        x     = torch.randn(5, 8)
        assert lora(x).shape == (5, 4)

    def test_base_weights_frozen(self):
        base = nn.Linear(8, 4)
        lora = PhotoLoRALinear(base, rank=2, n_basis=4, noise_sim=False)
        for p in lora.base.parameters():
            assert not p.requires_grad

    def test_adapter_trainable(self):
        base = nn.Linear(8, 4)
        lora = PhotoLoRALinear(base, rank=2, n_basis=4, noise_sim=False)
        trainable = [p for p in lora.adapter.parameters() if p.requires_grad]
        assert len(trainable) > 0


class TestAddPhotoLoRA:

    def test_adds_adapters(self):
        model = _TinyTransformer()
        photo = pkl.add_photo_lora(model, rank=2, n_basis=4,
                                    target_modules=["q_proj", "v_proj"],
                                    noise_sim=False)
        lora_count = sum(1 for m in photo.modules() if isinstance(m, PhotoLoRALinear))
        assert lora_count == 2

    def test_output_shape_preserved(self):
        model = _TinyTransformer()
        photo = pkl.add_photo_lora(model, rank=2, n_basis=4,
                                    target_modules=["q_proj"], noise_sim=False)
        x = torch.randn(3, 16)
        assert photo(x).shape == (3, 8)

    def test_only_adapter_trainable(self):
        model = _TinyTransformer()
        photo = pkl.add_photo_lora(model, rank=2, n_basis=4,
                                    target_modules=["q_proj"], noise_sim=False)
        for name, p in photo.named_parameters():
            if "base" in name:
                assert not p.requires_grad, f"{name} should be frozen"
