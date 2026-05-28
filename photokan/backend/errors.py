"""
photokan/backend/errors.py
Custom exception types for PhotoKAN.
"""


class PhotonicBackendError(RuntimeError):
    """Raised when a required backend is unavailable."""


class PhotonicCompilerError(RuntimeError):
    """Raised when AOT compilation fails."""


class PhotonicHardwareError(RuntimeError):
    """Raised for NPU hardware communication errors."""
