# photokan/backends/qant/backend.py
"""
Q.ANT backend — wraps the Q.PAL SDK.

Q.ANT builds photonic NPUs based on Thin-Film Lithium Niobate (TFLN):
- Electro-optic modulation for high-speed analog matrix operations
- Waveguide-based nonlinear activation functions
- Low-loss, high-bandwidth photonic interconnect

Technology: TFLN offers fast modulation (>40 GHz) with low chirp,
making it ideal for high-speed KAN edge operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..base import PhotonicBackend
from ..errors import PhotonicBackendError, PhotonicHardwareError

# Attempt to import Q.PAL SDK
try:
    import qpal as _qpal  # type: ignore[import]

    _QPAL_AVAILABLE = True
except ImportError:
    _qpal = None
    _QPAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Q.PAL op-type registry
# Maps EdgeActivation class name → Q.PAL op_type string
# ---------------------------------------------------------------------------
_OP_REGISTRY: dict[str, str] = {
    "SineEdgeActivation": "sine_waveguide",
    "FourierEdgeActivation": "fourier_waveguide",
    "SplineEdgeActivation": "spline_lut",
    "ReLUEdgeActivation": "relu_piecewise",
}


def _op_type(activation) -> str:
    cls_name = type(activation).__name__
    if cls_name not in _OP_REGISTRY:
        raise PhotonicBackendError(
            f"No Q.PAL op registered for '{cls_name}'. Registered: {list(_OP_REGISTRY.keys())}"
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
# QANTBackend
# ---------------------------------------------------------------------------


class QANTBackend(PhotonicBackend):
    @staticmethod
    def name() -> str:
        return "qant"

    @staticmethod
    def display_name() -> str:
        return "Q.ANT"

    @staticmethod
    def is_available() -> bool:
        if not _QPAL_AVAILABLE:
            return False
        try:
            return bool(_qpal.npu_available())
        except Exception:
            return False

    @staticmethod
    def device_info() -> dict:
        if not QANTBackend.is_available():
            return {"available": False, "vendor": "qant"}
        try:
            info = _qpal.npu_info()
            info["available"] = True
            info["vendor"] = "qant"
            return info
        except Exception as exc:
            return {"available": False, "vendor": "qant", "error": str(exc)}

    @staticmethod
    def get_device_info() -> QPALDeviceInfo:
        if not QANTBackend.is_available():
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
            raise PhotonicHardwareError(f"Failed to query NPU device info: {exc}") from exc

    @staticmethod
    def execute(x: torch.Tensor, activation, op_type: str) -> torch.Tensor:
        if not QANTBackend.is_available():
            raise RuntimeError("QANTBackend.execute called but hardware is not available.")
        params = {k: v.detach() for k, v in activation.named_parameters()}
        return _qpal.optical_forward(x, params, op_type)

    @staticmethod
    def compute_gradient(
        grad_output: torch.Tensor,
        x: torch.Tensor,
        activation,
        op_type: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not QANTBackend.is_available():
            raise RuntimeError("QANTBackend.compute_gradient called but hardware is not available.")
        params = {k: v.detach() for k, v in activation.named_parameters()}
        return _qpal.optical_gradient(grad_output, x, params, op_type)

    @staticmethod
    def nonlinear_forward(
        x: torch.Tensor,
        activation,
        op_type: str | None = None,
    ) -> torch.Tensor:
        if not QANTBackend.is_available():
            raise PhotonicHardwareError("nonlinear_forward called but NPU is not available.")
        op = op_type or _op_type(activation)
        params = {k: v.detach() for k, v in activation.named_parameters()}
        try:
            return _qpal.nonlinear_op(x, params, op)
        except Exception as exc:
            raise PhotonicHardwareError(f"Q.PAL nonlinear_op failed (op={op}): {exc}") from exc

    @staticmethod
    def adjoint_gradient(
        grad_output: torch.Tensor,
        x: torch.Tensor,
        activation,
        op_type: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not QANTBackend.is_available():
            raise PhotonicHardwareError("adjoint_gradient called but NPU is not available.")
        op = op_type or _op_type(activation)
        params = {k: v.detach() for k, v in activation.named_parameters()}
        try:
            return _qpal.gradient_op(grad_output, x, params, op)
        except Exception as exc:
            raise PhotonicHardwareError(f"Q.PAL gradient_op failed (op={op}): {exc}") from exc

    @staticmethod
    def validate_gradient(
        activation,
        x: torch.Tensor,
        atol: float = 1e-3,
        rtol: float = 1e-3,
    ) -> dict[str, Any]:
        if not QANTBackend.is_available():
            return {"passed": None, "reason": "NPU not available — skipped"}

        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            y_auto = activation(x)
        grad_auto = torch.autograd.grad(y_auto.sum(), x)[0]

        y_npu = QANTBackend.nonlinear_forward(x.detach(), activation)
        grad_npu, _ = QANTBackend.adjoint_gradient(torch.ones_like(y_npu), x.detach(), activation)

        max_err = (grad_auto - grad_npu).abs().max().item()
        rel_err = ((grad_auto - grad_npu).abs() / (grad_auto.abs() + 1e-8)).max().item()

        passed = (max_err <= atol) and (rel_err <= rtol)
        return {
            "passed": passed,
            "max_abs_err": max_err,
            "max_rel_err": rel_err,
            "atol": atol,
            "rtol": rtol,
        }

    @staticmethod
    def noise_profiles() -> dict[str, dict]:
        return {
            "npu1": {
                "snr_db": 14.0,
                "bit_depth": 6,
                "phase_noise_rad": 0.02,
                "technology": "tfln",
            },
            "npu2": {
                "snr_db": 16.0,
                "bit_depth": 8,
                "phase_noise_rad": 0.01,
                "technology": "tfln",
            },
            "ideal": {
                "snr_db": 60.0,
                "bit_depth": 16,
                "phase_noise_rad": 0.0,
                "technology": "tfln",
            },
        }

    @staticmethod
    def estimate_flops(layer) -> dict[str, Any]:
        if not QANTBackend.is_available():
            return _software_flop_estimate(layer)
        try:
            return _qpal.estimate_flops(layer)
        except Exception:
            return _software_flop_estimate(layer)


def _software_flop_estimate(layer) -> dict[str, Any]:
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
