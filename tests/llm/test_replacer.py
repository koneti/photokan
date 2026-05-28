"""Tests for LLM MLP replacement."""

import pytest
import torch
import torch.nn as nn

import photokan.llm as pkl
from photokan.layers import PhotoKAN


class _SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))

    def forward(self, x):
        return x + self.mlp(x)


class _SimpleTwoMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp0 = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))
        self.mlp1 = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))

    def forward(self, x):
        return self.mlp1(self.mlp0(x))


class TestReplaceMLPWithPhotoKAN:
    def test_replaces_sequential_mlp(self):
        model = _SimpleTwoMLP()
        photo = pkl.replace_mlp_with_photokan(model, n_basis=4, noise_sim=False)
        kan_count = sum(1 for m in photo.modules() if isinstance(m, PhotoKAN))
        assert kan_count >= 1

    def test_output_shape_preserved(self):
        model = _SimpleTwoMLP()
        photo = pkl.replace_mlp_with_photokan(model, n_basis=4, noise_sim=False)
        x = torch.randn(4, 8)
        assert photo(x).shape == (4, 8)

    def test_no_mlp_found_warns(self):
        model = nn.Linear(4, 4)
        with pytest.warns(UserWarning, match="No MLP/FFN modules detected"):
            pkl.replace_mlp_with_photokan(model)

    def test_compile_photokan_layers(self, tmp_path):
        model = _SimpleTwoMLP()
        photo = pkl.replace_mlp_with_photokan(model, n_basis=4, noise_sim=False)
        bundles = pkl.compile_photokan_layers(
            photo, str(tmp_path), n_lut_points=32, max_lut_mse=1e-1
        )
        assert len(bundles) >= 1
