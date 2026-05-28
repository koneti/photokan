# photokan/backends/salience/backend.py
"""
Salience Labs backend — wraps the Salience SDK.

Salience Labs builds photonic AI accelerators using III-V semiconductor
photonic integrated circuits (Indium Phosphide / InP-based):
- Micro-ring resonator weight banks for analog MAC operations
- Ultra-fast optical nonlinearities for edge activation functions
- Targeting ultra-low latency inference and training

Technology: III-V photonics (InP) offer higher optical gain and
faster modulation than silicon photonics, with better linearity
than TFLN. Trade-off: higher cost per chip, smaller scale currently.
"""

from __future__ import annotations

import torch

from ..base import PhotonicBackend

# Attempt to import Salience SDK
try:
    import salience as _sal  # type: ignore[import]

    _SAL_AVAILABLE = True
except ImportError:
    _sal = None
    _SAL_AVAILABLE = False


class SalienceBackend(PhotonicBackend):
    @staticmethod
    def name() -> str:
        return "salience"

    @staticmethod
    def display_name() -> str:
        return "Salience Labs"

    @staticmethod
    def is_available() -> bool:
        if not _SAL_AVAILABLE:
            return False
        try:
            return bool(_sal.accelerator_available())
        except Exception:
            return False

    @staticmethod
    def device_info() -> dict:
        if not SalienceBackend.is_available():
            return {"available": False, "vendor": "salience"}
        try:
            info = _sal.accelerator_info()
            info["available"] = True
            info["vendor"] = "salience"
            return info
        except Exception as exc:
            return {"available": False, "vendor": "salience", "error": str(exc)}

    @staticmethod
    def execute(x: torch.Tensor, activation, op_type: str) -> torch.Tensor:
        if not SalienceBackend.is_available():
            raise RuntimeError("SalienceBackend.execute called but hardware is not available.")
        params = {k: v.detach() for k, v in activation.named_parameters()}
        return _sal.optical_forward(x, params, op_type)

    @staticmethod
    def compute_gradient(
        grad_output: torch.Tensor,
        x: torch.Tensor,
        activation,
        op_type: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not SalienceBackend.is_available():
            raise RuntimeError(
                "SalienceBackend.compute_gradient called but hardware is not available."
            )
        params = {k: v.detach() for k, v in activation.named_parameters()}
        return _sal.optical_gradient(grad_output, x, params, op_type)

    @staticmethod
    def noise_profiles() -> dict[str, dict]:
        """
        Salience Labs III-V photonics noise profiles.

        III-V (InP) photonics characteristics:
        - Higher optical gain -> better SNR than silicon photonics
        - Micro-ring resonator thermal cross-talk
        - Lower phase noise than SOI due to better electro-optic response
        - Faster modulation bandwidth
        """
        return {
            "mr100": {
                "snr_db": 18.0,
                "bit_depth": 8,
                "phase_noise_rad": 0.008,
                "ring_thermal_crosstalk": 0.003,
                "technology": "iii_v_photonics",
            },
            "mr200": {
                "snr_db": 22.0,
                "bit_depth": 10,
                "phase_noise_rad": 0.004,
                "ring_thermal_crosstalk": 0.001,
                "technology": "iii_v_photonics",
            },
            "ideal": {
                "snr_db": 60.0,
                "bit_depth": 16,
                "phase_noise_rad": 0.0,
                "ring_thermal_crosstalk": 0.0,
                "technology": "iii_v_photonics",
            },
        }

    @staticmethod
    def estimate_flops(layer) -> dict:
        if not SalienceBackend.is_available():
            return {}
        try:
            return _sal.estimate_ops(layer)
        except Exception:
            return {}
