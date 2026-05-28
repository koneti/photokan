# photokan/utils/gradient_check.py
"""
Gradient validation utilities for Phase 2.

Provides tools to verify that PhotoKAN gradients are numerically
correct — essential before trusting training on real hardware.
"""

from __future__ import annotations

from typing import Any

import torch


def gradcheck_activation(
    activation,
    x: torch.Tensor | None = None,
    eps: float = 1e-5,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> dict[str, Any]:
    """
    Run torch.autograd.gradcheck on a single EdgeActivation.

    Args:
        activation : EdgeActivation instance.
        x          : Input tensor (default: small random float64 tensor).
        eps        : Finite-difference step size.
        atol       : Absolute tolerance for gradient comparison.
        rtol       : Relative tolerance.

    Returns:
        dict with 'passed', 'activation', 'n_params', and optionally 'error'.
    """
    if x is None:
        x = torch.randn(4, dtype=torch.float64) * 0.5

    # Convert all params to float64 for gradcheck
    activation_f64 = _to_float64(activation)
    param_names = [n for n, _ in activation_f64.named_parameters()]
    x_f64 = x.detach().double().requires_grad_(True)
    params_f64 = [p.detach().double().requires_grad_(True) for p in activation_f64.parameters()]

    def fn(x_in, *params):
        # Use functional_call to pass params without mutating module state
        param_dict = dict(zip(param_names, params))
        return torch.func.functional_call(activation_f64, param_dict, (x_in,))

    try:
        passed = torch.autograd.gradcheck(
            fn,
            (x_f64, *params_f64),
            eps=eps,
            atol=atol,
            rtol=rtol,
            raise_exception=True,
        )
        return {
            "passed": True,
            "activation": type(activation).__name__,
            "n_params": sum(p.numel() for p in activation.parameters()),
        }
    except Exception as exc:
        return {
            "passed": False,
            "activation": type(activation).__name__,
            "n_params": sum(p.numel() for p in activation.parameters()),
            "error": str(exc),
        }


def gradcheck_layer(
    layer,
    x: torch.Tensor | None = None,
    eps: float = 1e-5,
    atol: float = 1e-3,
) -> dict[str, Any]:
    """
    Numerical gradient check for a full PhotoKANLayer.

    Uses a small model to keep the check tractable.

    Returns:
        dict with 'passed', 'max_err', 'layer_info'.
    """
    if x is None:
        x = torch.randn(3, layer.in_features, dtype=torch.float64) * 0.3

    layer = _to_float64(layer)
    x_in = x.detach().double().requires_grad_(True)

    # Forward
    with torch.enable_grad():
        y = layer(x_in)
    grad_outputs = torch.ones_like(y)

    # Analytical gradient
    grads_analytic = torch.autograd.grad(y, x_in, grad_outputs=grad_outputs)[0]

    # Numerical gradient (finite differences)
    grads_numeric = torch.zeros_like(x_in)
    for i in range(x_in.numel()):
        x_plus = x_in.detach().clone()
        x_minus = x_in.detach().clone()
        x_plus.view(-1)[i] += eps
        x_minus.view(-1)[i] -= eps

        y_plus = layer(x_plus.requires_grad_(False))
        y_minus = layer(x_minus.requires_grad_(False))
        grads_numeric.view(-1)[i] = ((y_plus - y_minus) * grad_outputs).sum() / (2 * eps)

    max_err = (grads_analytic - grads_numeric).abs().max().item()
    passed = max_err <= atol

    return {
        "passed": passed,
        "max_err": max_err,
        "atol": atol,
        "layer_info": layer.extra_repr(),
    }


def compare_backends(
    model,
    x: torch.Tensor,
    backends: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run inference on multiple backends and compare outputs.

    Args:
        model    : PhotoKAN model.
        x        : Input tensor.
        backends : List of backend names to compare (default ['cpu']).

    Returns:
        dict mapping backend name → output tensor, plus pairwise max_diff.
    """
    from photokan.backends import available_backends

    if backends is None:
        backends = ["cpu"]

    avail = available_backends()
    outputs = {}

    original_backend = model.backend_mode

    for backend in backends:
        if backend == "qpal" and not avail["qpal"]:
            outputs[backend] = None
            continue
        if backend == "cuda" and not avail["cuda"]:
            outputs[backend] = None
            continue

        # Temporarily override backend on all layers
        for layer in model.layers:
            layer.backend_mode = backend

        model.eval()
        with torch.no_grad():
            outputs[backend] = model(x)

    # Restore
    for layer in model.layers:
        layer.backend_mode = original_backend

    # Pairwise max diffs
    keys = [k for k, v in outputs.items() if v is not None]
    diffs = {}
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1 :]:
            diff = (outputs[k1] - outputs[k2]).abs().max().item()
            diffs[f"{k1}_vs_{k2}"] = diff

    return {"outputs": outputs, "max_diffs": diffs}


def _to_float64(module: torch.nn.Module) -> torch.nn.Module:
    """Convert all parameters and buffers to float64."""
    return module.double()
