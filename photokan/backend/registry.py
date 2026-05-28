# photokan/backend/registry.py
"""Backend availability registry and helpers."""

from __future__ import annotations

import torch
from .qpal_backend import QPALBackend


def available_backends() -> dict[str, bool]:
    """
    Return a dict indicating which backends are currently available.

    Returns:
        {
          'cpu'  : True,
          'cuda' : bool,
          'qpal' : bool,
        }
    """
    return {
        "cpu":  True,
        "cuda": torch.cuda.is_available(),
        "qpal": QPALBackend.is_available(),
    }


def resolve_backend(requested: str) -> str:
    """
    Resolve 'auto' to a concrete backend string.

    Priority: qpal > cuda > cpu
    """
    if requested != "auto":
        return requested
    backends = available_backends()
    if backends["qpal"]:
        return "qpal"
    if backends["cuda"]:
        return "cuda"
    return "cpu"
