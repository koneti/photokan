# photokan/backend/sim_backend.py
"""
CPU simulation backend — physically accurate model of Q.ANT NPU nonlinear ops.

Injects realistic photonic noise (SNR, bit-depth quantisation, phase noise)
so simulation results match hardware behaviour, enabling pre-deployment
accuracy sweeps without owning an NPU.
"""

from __future__ import annotations

import torch


class NoiseModel:
    """Container for photonic hardware impairment parameters."""

    def __init__(
        self,
        snr_db: float = 14.0,
        bit_depth: int = 6,
        phase_noise_rad: float = 0.01,
        enabled: bool = True,
    ):
        self.snr_db = snr_db
        self.bit_depth = bit_depth
        self.phase_noise_rad = phase_noise_rad
        self.enabled = enabled

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Inject noise into a tensor according to this model."""
        if not self.enabled:
            return x

        # 1. Bit-depth quantisation (ADC model) with straight-through estimator
        n_levels = 2**self.bit_depth
        x_range = x.max() - x.min() + 1e-8
        x_norm = (x - x.min()) / x_range * (n_levels - 1)
        x_quant = x_norm + (torch.round(x_norm) - x_norm).detach()
        x = x_quant / (n_levels - 1) * x_range + x.min()

        # 2. Additive Gaussian noise at specified SNR
        signal_power = x.pow(2).mean().clamp_min(1e-12)
        snr_linear = 10 ** (self.snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(x) * noise_power.sqrt()
        x = x + noise

        # 3. Phase noise (small multiplicative perturbation)
        phase_perturb = 1.0 + self.phase_noise_rad * torch.randn_like(x)
        x = x * phase_perturb

        return x


# Default noise profiles per hardware generation
HARDWARE_PROFILES: dict[str, dict] = {
    "npu1": dict(snr_db=14.0, bit_depth=6, phase_noise_rad=0.02),
    "npu2": dict(snr_db=16.0, bit_depth=8, phase_noise_rad=0.01),
    "ideal": dict(snr_db=60.0, bit_depth=16, phase_noise_rad=0.0),
}

# Default noise config used when no explicit config is provided
_DEFAULT_NOISE_CONFIG: dict = {
    "snr_db": 14.0,
    "bit_depth": 6,
    "phase_noise_rad": 0.01,
    "enabled": True,
}


class SimBackend:
    """
    Physically accurate CPU simulation of photonic nonlinear operations.

    All methods are class methods. Noise configuration is passed explicitly
    per call via noise_config dict, avoiding shared mutable state.
    """

    @classmethod
    def set_default_noise_config(cls, config: dict) -> None:
        """Update the default noise config used when no per-call config is given."""
        global _DEFAULT_NOISE_CONFIG
        _DEFAULT_NOISE_CONFIG = dict(_DEFAULT_NOISE_CONFIG, **config)

    @classmethod
    def set_hardware_profile(cls, profile: str) -> dict:
        """Return a noise config dict for a named hardware profile ('npu1', 'npu2', 'ideal')."""
        if profile not in HARDWARE_PROFILES:
            raise ValueError(
                f"Unknown profile '{profile}'. Choose from: {list(HARDWARE_PROFILES.keys())}"
            )
        return dict(HARDWARE_PROFILES[profile], enabled=True)

    @classmethod
    def forward(
        cls,
        x: torch.Tensor,
        activation,
        noise_config: dict | None = None,
    ) -> torch.Tensor:
        """
        Run activation forward pass with optional noise injection.

        Args:
            x          : Input tensor.
            activation : An EdgeActivation instance.
            noise_config: Optional dict with noise overrides for this call.

        Returns:
            Output tensor, same shape as x.
        """
        out = activation(x)

        config = noise_config if noise_config is not None else _DEFAULT_NOISE_CONFIG
        nm = NoiseModel(**config)

        return nm.apply(out)

    @classmethod
    def get_transfer_function(
        cls,
        activation,
        n_points: int = 256,
        x_range: tuple = (-2.0, 2.0),
        noise_config: dict | None = None,
    ):
        """Return (x, y_ideal, y_noisy) for visualisation."""
        x = torch.linspace(x_range[0], x_range[1], n_points)
        with torch.no_grad():
            y_ideal = activation(x)
            y_noisy = cls.forward(x, activation, noise_config)
        return x.numpy(), y_ideal.numpy(), y_noisy.numpy()
