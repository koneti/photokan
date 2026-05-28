# photokan/backends/lightmatter/backend.py
"""
Lightmatter backend — wraps the Lightmatter SDK.

Lightmatter builds photonic AI accelerators using silicon photonics:
- Mach-Zehnder Interferometer (MZI) mesh for optical matrix multiplication
- Photonic activation functions
- High-throughput inference and training via optical interconnect

Technology: Silicon photonics enables dense integration with CMOS,
high manufacturing scalability, and mature fabrication processes.
Trade-off: higher propagation loss and thermal sensitivity vs TFLN.
"""

from __future__ import annotations

import torch

from ..base import PhotonicBackend

# Attempt to import Lightmatter SDK
try:
    import lightmatter as _lm  # type: ignore[import]

    _LM_AVAILABLE = True
except ImportError:
    _lm = None
    _LM_AVAILABLE = False


class LightmatterBackend(PhotonicBackend):
    @staticmethod
    def name() -> str:
        return "lightmatter"

    @staticmethod
    def display_name() -> str:
        return "Lightmatter"

    @staticmethod
    def is_available() -> bool:
        if not _LM_AVAILABLE:
            return False
        try:
            return bool(_lm.accelerator_available())
        except Exception:
            return False

    @staticmethod
    def device_info() -> dict:
        if not LightmatterBackend.is_available():
            return {"available": False, "vendor": "lightmatter"}
        try:
            info = _lm.accelerator_info()
            info["available"] = True
            info["vendor"] = "lightmatter"
            return info
        except Exception as exc:
            return {"available": False, "vendor": "lightmatter", "error": str(exc)}

    @staticmethod
    def execute(x: torch.Tensor, activation, op_type: str) -> torch.Tensor:
        if not LightmatterBackend.is_available():
            raise RuntimeError("LightmatterBackend.execute called but hardware is not available.")
        params = {k: v.detach() for k, v in activation.named_parameters()}
        return _lm.optical_forward(x, params, op_type)

    @staticmethod
    def compute_gradient(
        grad_output: torch.Tensor,
        x: torch.Tensor,
        activation,
        op_type: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not LightmatterBackend.is_available():
            raise RuntimeError(
                "LightmatterBackend.compute_gradient called but hardware is not available."
            )
        params = {k: v.detach() for k, v in activation.named_parameters()}
        return _lm.optical_gradient(grad_output, x, params, op_type)

    @staticmethod
    def noise_profiles() -> dict[str, dict]:
        """
        Lightmatter silicon photonics noise profiles.

        Silicon photonics characteristics:
        - Higher propagation loss (~2-3 dB/cm) than TFLN
        - Thermal crosstalk between adjacent MZIs
        - Good manufacturing scalability via CMOS-compatible processes
        """
        return {
            "envise1": {
                "snr_db": 12.0,
                "bit_depth": 5,
                "phase_noise_rad": 0.025,
                "thermal_drift": 0.005,
                "technology": "silicon_photonics",
            },
            "mars1": {
                "snr_db": 15.0,
                "bit_depth": 7,
                "phase_noise_rad": 0.015,
                "thermal_drift": 0.002,
                "technology": "silicon_photonics",
            },
            "ideal": {
                "snr_db": 60.0,
                "bit_depth": 16,
                "phase_noise_rad": 0.0,
                "thermal_drift": 0.0,
                "technology": "silicon_photonics",
            },
        }

    @staticmethod
    def estimate_flops(layer) -> dict:
        if not LightmatterBackend.is_available():
            return {}
        try:
            return _lm.estimate_ops(layer)
        except Exception:
            return {}
