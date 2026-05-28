# photokan/backends/errors.py
"""Shared error types for photonic backends."""


class PhotonicBackendError(Exception):
    """Raised for backend registration or dispatch failures."""


class PhotonicCompilerError(PhotonicBackendError):
    """Raised when AOT compilation to a vendor backend fails."""


class PhotonicHardwareError(PhotonicBackendError):
    """Raised when a vendor hardware operation fails at runtime."""
