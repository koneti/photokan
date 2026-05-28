# photokan/backends/registry.py
"""
Photonic backend registry — discovers available vendor backends
and dispatches to the right one.
"""

from __future__ import annotations

import importlib

import torch

from .base import PhotonicBackend
from .errors import PhotonicBackendError

# Vendor backends are registered here.
_VENDOR_BACKENDS: dict[str, type] = {}


def register_vendor(backend_cls: type[PhotonicBackend]) -> None:
    """Register a vendor backend class."""
    name = backend_cls.name()
    _VENDOR_BACKENDS[name] = backend_cls


def available_backends() -> dict[str, bool]:
    """
    Return availability status for all registered vendors plus cpu/cuda.

    Returns:
        {'cpu': True, 'cuda': bool, 'qant': bool, 'lightmatter': bool, ...}
    """
    result = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
    }
    for name, cls in _VENDOR_BACKENDS.items():
        try:
            result[name] = cls.is_available()
        except Exception:
            result[name] = False
    return result


def get_backend(name: str) -> type[PhotonicBackend]:
    """Return a vendor backend class by name."""
    if name not in _VENDOR_BACKENDS:
        raise PhotonicBackendError(
            f"Unknown backend '{name}'. Registered: {list(_VENDOR_BACKENDS.keys())}"
        )
    return _VENDOR_BACKENDS[name]


def resolve_backend(requested: str) -> str:
    """
    Resolve 'auto' to the first available photonic backend.

    Priority order: registered vendors (alphabetical) > cuda > cpu.
    """
    if requested != "auto":
        return requested

    # Check vendors in registration order
    for name in _VENDOR_BACKENDS:
        try:
            if _VENDOR_BACKENDS[name].is_available():
                return name
        except Exception:
            continue

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_noise_config(backend_name: str, profile: str) -> dict:
    """Get a noise profile from a specific vendor backend."""
    if backend_name not in _VENDOR_BACKENDS:
        raise PhotonicBackendError(
            f"Unknown backend '{backend_name}'. Registered: {list(_VENDOR_BACKENDS.keys())}"
        )
    profiles = _VENDOR_BACKENDS[backend_name].noise_profiles()
    if profile not in profiles:
        raise ValueError(
            f"Unknown profile '{profile}' for vendor '{backend_name}'. "
            f"Available: {list(profiles.keys())}"
        )
    return dict(profiles[profile], enabled=True)


def all_vendor_names() -> list[str]:
    """Return list of registered vendor names."""
    return list(_VENDOR_BACKENDS.keys())


# ---------------------------------------------------------------------------
# Auto-discover vendor backends
# ---------------------------------------------------------------------------


def _discover_vendors() -> None:
    """Import vendor modules to trigger their register_vendor() calls."""
    vendor_modules = [
        "photokan.backends.qant",
        "photokan.backends.lightmatter",
        "photokan.backends.salience",
    ]
    for mod_path in vendor_modules:
        try:
            importlib.import_module(mod_path)
        except ImportError:
            pass


_discover_vendors()
