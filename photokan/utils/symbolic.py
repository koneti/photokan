# photokan/utils/symbolic.py
"""Symbolic regression utilities for KAN edge functions."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def symbolic_regress_activation(
    activation,
    n_points: int = 100,
    x_range: tuple = (-2.0, 2.0),
) -> Any:
    """
    Fit a symbolic formula to a learned edge activation.

    Uses scipy curve_fit with a library of candidate functions,
    then tries sympy simplification.

    Returns:
        sympy.Expr if sympy is available, else a descriptive string.
    """
    x_np = np.linspace(*x_range, n_points)
    x_t = torch.tensor(x_np, dtype=torch.float32)
    with torch.no_grad():
        y_np = activation(x_t).numpy()

    try:
        import sympy as sp
        from scipy.optimize import curve_fit

        x_sym = sp.Symbol("x")
        best_expr = None
        best_residual = float("inf")

        # Candidate library
        candidates = [
            ("linear", lambda x, a, b: a * x + b, [1.0, 0.0]),
            ("quadratic", lambda x, a, b, c: a * x**2 + b * x + c, [1.0, 0.0, 0.0]),
            ("sine", lambda x, a, b, c: a * np.sin(b * x + c), [1.0, 1.0, 0.0]),
            ("exp", lambda x, a, b: a * np.exp(b * x), [1.0, 0.1]),
        ]

        for name, fn, p0 in candidates:
            try:
                popt, _ = curve_fit(fn, x_np, y_np, p0=p0, maxfev=2000)
                residual = np.mean((fn(x_np, *popt) - y_np) ** 2)
                if residual < best_residual:
                    best_residual = residual
                    best_expr = _build_sympy_expr(name, popt, x_sym)
            except Exception:
                continue

        if best_expr is not None:
            return sp.simplify(best_expr)
        return sp.Symbol("?")

    except ImportError:
        # Fallback: describe the activation params
        params = {k: v.detach().numpy().tolist() for k, v in activation.named_parameters()}
        return f"{type(activation).__name__}(params={params})"


def _build_sympy_expr(name: str, popt, x):
    import sympy as sp

    if name == "linear":
        a, b = popt
        return a * x + b
    elif name == "quadratic":
        a, b, c = popt
        return a * x**2 + b * x + c
    elif name == "sine":
        a, b, c = popt
        return a * sp.sin(b * x + c)
    elif name == "exp":
        a, b = popt
        return a * sp.exp(b * x)
    return x
