# photokan/backends/salience/__init__.py
"""Salience Labs vendor backend — auto-registers on import."""

from ..registry import register_vendor
from .backend import SalienceBackend

register_vendor(SalienceBackend)
