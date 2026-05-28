# photokan/activations/__init__.py
"""KAN edge activation functions."""

from .base import EdgeActivation
from .fourier_edge import FourierEdgeActivation
from .relu_edge import ReLUEdgeActivation
from .sine_edge import SineEdgeActivation
from .spline_edge import SplineEdgeActivation

__all__ = [
    "ACTIVATION_REGISTRY",
    "EdgeActivation",
    "FourierEdgeActivation",
    "ReLUEdgeActivation",
    "SineEdgeActivation",
    "SplineEdgeActivation",
    "get_activation_class",
]

ACTIVATION_REGISTRY: dict[str, type[EdgeActivation]] = {
    "sine": SineEdgeActivation,
    "fourier": FourierEdgeActivation,
    "spline": SplineEdgeActivation,
    "relu": ReLUEdgeActivation,
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
            f"Unknown activation '{name}'. Choose from: {list(ACTIVATION_REGISTRY.keys())}"
        )
    return ACTIVATION_REGISTRY[name]
