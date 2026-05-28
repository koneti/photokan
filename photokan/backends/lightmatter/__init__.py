# photokan/backends/lightmatter/__init__.py
"""Lightmatter vendor backend — auto-registers on import."""

from ..registry import register_vendor
from .backend import LightmatterBackend

register_vendor(LightmatterBackend)
