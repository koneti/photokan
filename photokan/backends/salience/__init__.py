# photokan/backends/salience/__init__.py
"""Salience Labs vendor backend — auto-registers on import."""

from .backend import SalienceBackend
from ..registry import register_vendor

register_vendor(SalienceBackend)
