# photokan/backends/qant/__init__.py
"""Q.ANT vendor backend — auto-registers on import."""

from .backend import QANTBackend
from ..registry import register_vendor

register_vendor(QANTBackend)
