# photokan/activations/__init__.py
"""KAN edge activation functions."""

from .base import EdgeActivation
from .sine_edge import SineEdgeActivation
from .fourier_edge import FourierEdgeActivation
from .spline_edge import SplineEdgeActivation
from .relu_edge import ReLUEdgeActivation

__all__ = [
    "EdgeActivation",
    "SineEdgeActivation",
    "FourierEdgeActivation",
    "SplineEdgeActivation",
    "ReLUEdgeActivation",
    "get_activation_class",
    "ACTIVATION_REGISTRY",
]

ACTIVATION_REGISTRY: dict[str, type[EdgeActivation]] = {
    "sine":    SineEdgeActivation,
    "fourier": FourierEdgeActivation,
    "spline":  SplineEdgeActivation,
    "relu":    ReLUEdgeActivation,
}


def get_activation_class(name: str) -> type[EdgeActivation]:
    """
    Return an activation class by name.

    Args:
        name: One of 'sine', 'fourier', 'spline', 'relu', or a class
              that is already a subclass of EdgeActivation.

    Raises:
        ValueError: If the name is not registered.
    """
    if isinstance(name, type) and issubclass(name, EdgeActivation):
        return name
    name = name.lower()
    if name not in ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Choose from: {list(ACTIVATION_REGISTRY.keys())}"
        )
    return ACTIVATION_REGISTRY[name]
