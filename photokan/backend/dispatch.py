# photokan/backend/dispatch.py
"""
Generic dispatch: routes KAN edge computations to the appropriate
photonic vendor backend or CPU simulation, with full PyTorch autograd.

For the CPU/simulation path, we use native PyTorch autograd directly.
Custom autograd.Function is reserved for vendor hardware paths.
"""

from __future__ import annotations

import torch

from ..backends.registry import get_backend, resolve_backend
from ..backends.errors import PhotonicBackendError
from .sim_backend import SimBackend


def _make_vendor_fn(vendor_backend, activation):
    """
    Return a torch.autograd.Function bound to a vendor backend via closure.
    Only used for the hardware-accelerated path.
    """

    class _BoundVendorFunction(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x: torch.Tensor, *param_tensors) -> torch.Tensor:
            ctx.save_for_backward(x, *param_tensors)
            op_type = type(activation).__name__
            return vendor_backend.execute(x, activation, op_type)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            saved = ctx.saved_tensors
            x = saved[0]
            op_type = type(activation).__name__
            grad_x, grad_params_dict = vendor_backend.compute_gradient(
                grad_output, x, activation, op_type
            )
            param_grads = tuple(grad_params_dict.values())
            return (grad_x, *param_grads)

    return _BoundVendorFunction


def apply_edge(
    x: torch.Tensor,
    activation,
    backend_mode: str = "auto",
    noise_config: dict | None = None,
) -> torch.Tensor:
    """
    Dispatch one KAN edge through the photonic backend with full autograd.

    Args:
        x            : [batch] input for this edge.
        activation   : EdgeActivation instance.
        backend_mode : 'auto', a vendor name ('qant', 'lightmatter', 'salience'),
                       'cuda', or 'cpu'.
        noise_config : Optional dict with noise overrides for CPU simulation.

    Returns:
        [batch] edge output phi(x).
    """
    resolved = resolve_backend(backend_mode) if backend_mode == "auto" else backend_mode

    # Check if resolved backend is a registered vendor with hardware present
    vendor_name = resolved
    try:
        vendor_cls = get_backend(vendor_name)
        if vendor_cls.is_available():
            fn_cls = _make_vendor_fn(vendor_cls, activation)
            param_tensors = tuple(activation.parameters())
            return fn_cls.apply(x, *param_tensors)
    except PhotonicBackendError:
        pass

    # Explicit vendor requested but unavailable
    if backend_mode not in ("auto", "cpu", "cuda"):
        try:
            get_backend(backend_mode)
            raise PhotonicBackendError(
                f"backend='{backend_mode}' requested but hardware is not available. "
                f"Use backend='auto' or backend='cpu'."
            )
        except PhotonicBackendError:
            if backend_mode == "qpal":
                raise PhotonicBackendError(
                    "backend='qpal' is deprecated. Use backend='qant' instead."
                )
            raise

    # CPU/simulation path — native PyTorch autograd handles everything
    return SimBackend.forward(x, activation, noise_config)
