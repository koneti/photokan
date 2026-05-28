# photokan/backend/__init__.py
"""Hardware dispatch layer — vendor-agnostic."""

from .dispatch import apply_edge
from .sim_backend import NoiseModel, SimBackend

__all__ = [
    "NoiseModel",
    "SimBackend",
    "apply_edge",
]
