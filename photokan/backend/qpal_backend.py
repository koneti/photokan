# photokan/backend/qpal_backend.py
"""
Q.ANT NPU backend — wraps the Q.PAL Python SDK.

Phase 2 additions:
  - Structured QPALDeviceInfo dataclass
  - NPU memory management helpers
  - Op-type registry mapping activation classes → Q.PAL op strings
  - Richer error context on SDK failures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import torch

from .errors import PhotonicBackendError, PhotonicHardwareError

try:
    import qpal as _qpal          # type: ignore[import]
    _QPAL_AVAILABLE = True
except ImportError:
    _qpal = None
    _QPAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Q.PAL op-type registry
# Maps EdgeActivation class name → Q.PAL op_type string
# ---------------------------------------------------------------------------
_OP_REGISTRY: dict[str, str] = {
    "SineEdgeActivation":    "sine_waveguide",
    "FourierEdgeActivation": "fourier_waveguide",
    "SplineEdgeActivation":  "spline_lut",
    "ReLUEdgeActivation":    "relu_piecewise",
}


def _op_type(activation) -> str:
    cls_name = type(activation).__name__
    if cls_name not in _OP_REGISTRY:
        raise PhotonicBackendError(
            f"No Q.PAL op registered for '{cls_name}'. "
            f"Registered: {list(_OP_REGISTRY.keys())}"
        )
    return _OP_REGISTRY[cls_name]


# ---------------------------------------------------------------------------
# Device info
# ---------------------------------------------------------------------------

@dataclass
class QPALDeviceInfo:
    available: bool
    generation: str = "unknown"
    memory_mb: int = 0
    pcie_bandwidth_gbps: float = 0.0
    max_batch_size: int = 0
    supported_ops: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    def __str__(self) -> str:
        if not self.available:
            return "Q.ANT NPU: not available"
        return (
            f"Q.ANT NPU ({self.generation}) | "
            f"{self.memory_mb} MB | "
            f"PCIe {self.pcie_bandwidth_gbps:.1f} GB/s | "
            f"max_batch={self.max_batch_size}"
        )


# ---------------------------------------------------------------------------
# QPALBackend
# ---------------------------------------------------------------------------

class QPALBackend:
    """Static wrapper around the Q.PAL Python API."""

    @staticmethod
    def is_available() -> bool:
        if not _QPAL_AVAILABLE:
            return False
        try:
            return bool(_qpal.npu_available())
        except Exception:
            return False

    @staticmethod
    def get_device_info() -> QPALDeviceInfo:
        """Return structured NPU device information."""
        if not QPALBackend.is_available():
            return QPALDeviceInfo(available=False)
        try:
            raw = _qpal.device_info()
            return QPALDeviceInfo(
                available=True,
                generation=raw.get("generation", "unknown"),
                memory_mb=raw.get("memory_mb", 0),
                pcie_bandwidth_gbps=raw.get("pcie_bandwidth_gbps", 0.0),
                max_batch_size=raw.get("max_batch_size", 0),
                supported_ops=raw.get("supported_ops", []),
                raw=raw,
            )
        except Exception as exc:
            raise PhotonicHardwareError(
                f"Failed to query NPU device info: {exc}"
            ) from exc

    @staticmethod
    def nonlinear_forward(
        x: torch.Tensor,
        activation,
        op_type: str | None = None,
    ) -> torch.Tensor:
        """
        Execute a nonlinear edge op on the NPU via Q.PAL.

        Args:
            x         : Input tensor.
            activation: EdgeActivation instance (provides params).
            op_type   : Override Q.PAL op string; auto-resolved if None.

        Returns:
            Output tensor on CPU.
        """
        if not QPALBackend.is_available():
            raise PhotonicHardwareError(
                "nonlinear_forward called but NPU is not available."
            )
        op = op_type or _op_type(activation)
        params = {k: v.detach() for k, v in activation.named_parameters()}
        try:
            return _qpal.nonlinear_op(x, params, op)
        except Exception as exc:
            raise PhotonicHardwareError(
                f"Q.PAL nonlinear_op failed (op={op}): {exc}"
            ) from exc

    @staticmethod
    def adjoint_gradient(
        grad_output: torch.Tensor,
        x: torch.Tensor,
        activation,
        op_type: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute gradients on NPU via adjoint method.

        Returns:
            (grad_x, grad_params_dict)
        """
        if not QPALBackend.is_available():
            raise PhotonicHardwareError(
                "adjoint_gradient called but NPU is not available."
            )
        op = op_type or _op_type(activation)
        params = {k: v.detach() for k, v in activation.named_parameters()}
        try:
            return _qpal.gradient_op(grad_output, x, params, op)
        except Exception as exc:
            raise PhotonicHardwareError(
                f"Q.PAL gradient_op failed (op={op}): {exc}"
            ) from exc

    @staticmethod
    def estimate_flops(layer) -> dict[str, Any]:
        """
        Ask Q.PAL for op-count and energy estimates for a PhotoKANLayer.
        Falls back to software estimate when NPU is unavailable.
        """
        if not QPALBackend.is_available():
            return _software_flop_estimate(layer)
        try:
            return _qpal.estimate_flops(layer)
        except Exception:
            return _software_flop_estimate(layer)

    @staticmethod
    def validate_gradient(
        activation,
        x: torch.Tensor,
        atol: float = 1e-3,
        rtol: float = 1e-3,
    ) -> dict[str, Any]:
        """
        Compare adjoint NPU gradient vs exact autograd gradient.
        Used in Phase 2 integration tests.

        Returns:
            dict with 'passed', 'max_err', 'rel_err' keys.
        """
        if not QPALBackend.is_available():
            return {"passed": None, "reason": "NPU not available — skipped"}

        x = x.detach().requires_grad_(True)
        # Autograd path
        with torch.enable_grad():
            y_auto = activation(x)
        grad_auto = torch.autograd.grad(y_auto.sum(), x)[0]

        # Adjoint path
        y_npu = QPALBackend.nonlinear_forward(x.detach(), activation)
        grad_npu, _ = QPALBackend.adjoint_gradient(
            torch.ones_like(y_npu), x.detach(), activation
        )

        max_err = (grad_auto - grad_npu).abs().max().item()
        rel_err = (
            (grad_auto - grad_npu).abs()
            / (grad_auto.abs() + 1e-8)
        ).max().item()

        passed = (max_err <= atol) and (rel_err <= rtol)
        return {
            "passed": passed,
            "max_abs_err": max_err,
            "max_rel_err": rel_err,
            "atol": atol,
            "rtol": rtol,
        }


def _software_flop_estimate(layer) -> dict[str, Any]:
    """Fallback FLOP estimate when Q.PAL is unavailable."""
    n_edges = layer.in_features * layer.out_features
    act = layer.edge_activations[0]
    ops_per_edge = sum(p.numel() for p in act.parameters())
    total_ops = n_edges * ops_per_edge
    return {
        "n_edges": n_edges,
        "ops_per_edge": ops_per_edge,
        "total_ops": total_ops,
        "source": "software_estimate",
    }
