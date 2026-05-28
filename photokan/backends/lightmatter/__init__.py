# photokan/backends/lightmatter/__init__.py
"""Lightmatter vendor backend — auto-registers on import."""

from .backend import LightmatterBackend
from ..registry import register_vendor

register_vendor(LightmatterBackend)
