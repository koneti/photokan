# photokan/backends/qant/__init__.py
"""Q.ANT vendor backend — auto-registers on import."""

from ..registry import register_vendor
from .backend import QANTBackend

register_vendor(QANTBackend)
