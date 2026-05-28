# photokan/__init__.py
"""
PhotoKAN — Photonic Kolmogorov-Arnold Networks
================================================

Vendor-agnostic photonic computing framework. Supports Q.ANT, Lightmatter,
Salience Labs, and any future photonic hardware via pluggable backends.
"""
from .layers import PhotoKANLayer, PhotoKAN, PhotoConvKAN
from .backend import (
    available_backends, resolve_backend, get_backend,
    all_vendor_names, get_noise_config,
    PhotonicBackendError, PhotonicCompilerError, PhotonicHardwareError,
    PhotonicBackend,
)
from .backend.errors import PhotonicCompilerError, PhotonicHardwareError
from .sim import PhotonicSimulator
from .compiler import PhotonicCompiler, PhotonicProgram
from .utils import (
    Profiler, plot_kan_graph, estimate_model_energy,
    gradcheck_activation, gradcheck_layer, export_onnx,
)

__version__ = "0.4.0"

__all__ = [
    "PhotoKAN", "PhotoKANLayer", "PhotoConvKAN",
    "PhotonicSimulator",
    "PhotonicCompiler", "PhotonicProgram",
    "Profiler", "plot_kan_graph",
    "estimate_model_energy",
    "gradcheck_activation", "gradcheck_layer",
    "export_onnx",
    "available_backends", "resolve_backend",
    "get_backend", "all_vendor_names", "get_noise_config",
    "PhotonicBackend",
    "PhotonicBackendError", "PhotonicCompilerError", "PhotonicHardwareError",
    "__version__",
]
