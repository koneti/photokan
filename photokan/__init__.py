# photokan/__init__.py
"""
PhotoKAN — Photonic Kolmogorov-Arnold Networks
================================================

Vendor-agnostic photonic computing framework. Supports Q.ANT, Lightmatter,
Salience Labs, and any future photonic hardware via pluggable backends.
"""

from .backend import NoiseModel, SimBackend, apply_edge
from .backends import (
    PhotonicBackend,
    PhotonicBackendError,
    PhotonicCompilerError,
    PhotonicHardwareError,
    all_vendor_names,
    available_backends,
    get_backend,
    get_noise_config,
    resolve_backend,
)
from .compiler import PhotonicCompiler, PhotonicProgram
from .layers import PhotoConvKAN, PhotoKAN, PhotoKANLayer
from .sim import PhotonicSimulator
from .utils import (
    Profiler,
    estimate_model_energy,
    export_onnx,
    gradcheck_activation,
    gradcheck_layer,
    plot_kan_graph,
)

__version__ = "0.4.3"

__all__ = [
    "PhotoConvKAN",
    "PhotoKAN",
    "PhotoKANLayer",
    "PhotonicBackend",
    "PhotonicBackendError",
    "PhotonicCompiler",
    "PhotonicCompilerError",
    "PhotonicHardwareError",
    "PhotonicProgram",
    "PhotonicSimulator",
    "Profiler",
    "__version__",
    "all_vendor_names",
    "available_backends",
    "estimate_model_energy",
    "export_onnx",
    "get_backend",
    "get_noise_config",
    "gradcheck_activation",
    "gradcheck_layer",
    "plot_kan_graph",
    "resolve_backend",
]
