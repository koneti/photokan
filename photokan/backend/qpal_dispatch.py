# photokan/backend/qpal_dispatch.py
"""
Legacy Q.PAL dispatch — redirects to the new vendor-agnostic dispatch.

Kept for backward compatibility. Use photokan.backend.dispatch instead.
"""

from .dispatch import apply_edge  # noqa: F401
