# photokan/llm/__init__.py
"""
photokan.llm — LLM integration (Phase 3).

Provides tools to integrate PhotoKAN into transformer models:
  - replace_mlp_with_photokan: swap FFN layers for photonic KAN
  - add_photo_lora: insert PhotoKAN low-rank adapters
  - compile_photokan_layers: AOT compile all KAN layers in a model
  - PhotoKANAttention: full KAN-based attention mechanism
"""

from .adapters import PhotoLoRALinear, add_photo_lora
from .attention import PhotoKANAttention
from .replacer import compile_photokan_layers, replace_mlp_with_photokan

__all__ = [
    "PhotoKANAttention",
    "PhotoLoRALinear",
    "add_photo_lora",
    "compile_photokan_layers",
    "replace_mlp_with_photokan",
]
