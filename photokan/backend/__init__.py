# photokan/backend/__init__.py
"""Hardware dispatch layer — vendor-agnostic."""

from .sim_backend import SimBackend, NoiseModel
from .dispatch import apply_edge
from ..backends import (
    available_backends,
    resolve_backend,
    get_backend,
    get_noise_config,
    all_vendor_names,
    PhotonicBackendError,
    PhotonicCompilerError,
    PhotonicHardwareError,
    PhotonicBackend,
)

__all__ = [
    "SimBackend",
    "NoiseModel",
    "apply_edge",
    "available_backends",
    "resolve_backend",
    "get_backend",
    "get_noise_config",
    "all_vendor_names",
    "PhotonicBackend",
    "PhotonicBackendError",
    "PhotonicCompilerError",
    "PhotonicHardwareError",
]
