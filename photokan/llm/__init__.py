# photokan/llm/__init__.py
"""
photokan.llm — LLM integration (Phase 3).

Provides tools to integrate PhotoKAN into transformer models:
  - replace_mlp_with_photokan: swap FFN layers for photonic KAN
  - add_photo_lora: insert PhotoKAN low-rank adapters
  - compile_photokan_layers: AOT compile all KAN layers in a model
  - PhotoKANAttention: full KAN-based attention mechanism
"""
from .replacer import replace_mlp_with_photokan, compile_photokan_layers
from .adapters import add_photo_lora, PhotoLoRALinear
from .attention import PhotoKANAttention

__all__ = [
    "replace_mlp_with_photokan",
    "compile_photokan_layers",
    "add_photo_lora",
    "PhotoLoRALinear",
    "PhotoKANAttention",
]
