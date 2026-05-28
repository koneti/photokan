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

import torch

from ..base import PhotonicBackend

# Attempt to import Q.PAL SDK
try:
    import qpal as _qpal  # type: ignore[import]
    _QPAL_AVAILABLE = True
except ImportError:
    _qpal = None
    _QPAL_AVAILABLE = False


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
    def execute(x: torch.Tensor, activation, op_type: str) -> torch.Tensor:
        if not QANTBackend.is_available():
            raise RuntimeError(
                "QANTBackend.execute called but hardware is not available."
            )
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
            raise RuntimeError(
                "QANTBackend.compute_gradient called but hardware is not available."
            )
        params = {k: v.detach() for k, v in activation.named_parameters()}
        return _qpal.optical_gradient(grad_output, x, params, op_type)

    @staticmethod
    def noise_profiles() -> dict[str, dict]:
        """
        Q.ANT TFLN NPU noise profiles.

        TFLN characteristics:
        - Low propagation loss (~0.1 dB/cm)
        - High electro-optic bandwidth (>40 GHz)
        - Moderate phase noise from RF drive electronics
        """
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
    def estimate_flops(layer) -> dict:
        if not QANTBackend.is_available():
            return {}
        try:
            return _qpal.estimate_ops(layer)
        except Exception:
            return {}
