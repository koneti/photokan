# photokan/sim/photonic_sim.py
"""
PhotonicSimulator — high-level interface for pre-deployment analysis.

Enables accuracy sweeps across SNR, noise profiling, and transfer
function visualisation without any photonic hardware.
"""

from __future__ import annotations

import torch

from ..backend.sim_backend import HARDWARE_PROFILES, SimBackend


class PhotonicSimulator:
    """
    Simulate Q.ANT NPU behaviour on CPU for pre-deployment testing.

    Usage::

        sim = PhotonicSimulator()
        sim.set_hardware_profile('npu2')
        results = sim.sweep_snr(model, x_test, y_test, snr_range=[8, 10, 12, 14, 16])
        sim.plot_snr_accuracy(results)
    """

    def __init__(self, profile: str = "npu1"):
        if profile not in HARDWARE_PROFILES and profile != "custom":
            raise ValueError(
                f"Unknown profile '{profile}'. "
                f"Available: {list(HARDWARE_PROFILES.keys()) + ['custom']}"
            )
        self._profile = profile
        self._noise_config: dict = (
            dict(HARDWARE_PROFILES[profile], enabled=True)
            if profile in HARDWARE_PROFILES
            else {"snr_db": 14.0, "bit_depth": 6, "phase_noise_rad": 0.01, "enabled": True}
        )

    def set_hardware_profile(self, profile: str) -> None:
        """Switch noise profile ('npu1', 'npu2', 'ideal', or 'custom')."""
        if profile not in HARDWARE_PROFILES and profile != "custom":
            raise ValueError(
                f"Unknown profile '{profile}'. "
                f"Available: {list(HARDWARE_PROFILES.keys()) + ['custom']}"
            )
        if profile in HARDWARE_PROFILES:
            self._noise_config = dict(HARDWARE_PROFILES[profile], enabled=True)
        self._profile = profile

    def set_noise_model(self, **kwargs) -> None:
        """Set a custom noise model. See NoiseModel parameters."""
        self._noise_config = dict(self._noise_config, **kwargs)
        self._profile = "custom"

    # ------------------------------------------------------------------
    # SNR sweep
    # ------------------------------------------------------------------

    def sweep_snr(
        self,
        model,
        x: torch.Tensor,
        y_true: torch.Tensor | None = None,
        snr_range: list[float] | None = None,
        metric: str = "mse",
    ) -> dict:
        """
        Evaluate model accuracy across a range of SNR levels.

        Temporarily overrides each layer's noise_config for each SNR level,
        then restores the originals.

        Args:
            model    : PhotoKAN or PhotoKANLayer.
            x        : Input test tensor.
            y_true   : Ground-truth labels (optional; returns raw outputs if None).
            snr_range: List of SNR values in dB (default [8,10,12,14,16,20]).
            metric   : 'mse' or 'mae'.

        Returns:
            Dict with 'snr_db' and 'scores' lists.
        """
        if snr_range is None:
            snr_range = [8.0, 10.0, 12.0, 14.0, 16.0, 20.0]

        # Collect layers to sweep — works for both PhotoKAN and PhotoKANLayer
        if hasattr(model, "layers"):
            layers = model.layers
        else:
            layers = [model]

        # Save original noise configs
        original_configs = [dict(layer.noise_config) for layer in layers]

        scores = []
        try:
            for snr in snr_range:
                # Override each layer's noise_config with the current SNR
                for layer in layers:
                    layer.noise_config = dict(self._noise_config, snr_db=snr, enabled=True)

                model.eval()
                with torch.no_grad():
                    pred = model(x)

                if y_true is not None:
                    if metric == "mse":
                        score = float(
                            torch.nn.functional.mse_loss(pred.squeeze(), y_true.squeeze())
                        )
                    elif metric == "mae":
                        score = float(torch.nn.functional.l1_loss(pred.squeeze(), y_true.squeeze()))
                    else:
                        raise ValueError(f"Unknown metric '{metric}'.")
                else:
                    score = float(pred.abs().mean())
                scores.append(score)
        finally:
            # Restore original noise configs
            for layer, original in zip(layers, original_configs):
                layer.noise_config = original

        return {
            "snr_db": snr_range,
            "scores": scores,
            "metric": metric,
        }

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def plot_transfer_function(
        self,
        activation,
        n_points: int = 256,
        x_range: tuple = (-2.0, 2.0),
    ):
        """
        Plot ideal vs noisy transfer function φ(x).

        Returns:
            matplotlib.Figure
        """
        import matplotlib.pyplot as plt

        x, y_ideal, y_noisy = SimBackend.get_transfer_function(
            activation, n_points, x_range, noise_config=self._noise_config
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, y_ideal, label="Ideal φ(x)", linewidth=2)
        ax.plot(
            x,
            y_noisy,
            label=f"Simulated (SNR={self._noise_config['snr_db']:.0f} dB)",
            alpha=0.7,
            linestyle="--",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("φ(x)")
        ax.set_title(f"Transfer function: {type(activation).__name__}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_snr_accuracy(self, sweep_results: dict):
        """
        Plot accuracy vs SNR curve from sweep_snr() output.

        Returns:
            matplotlib.Figure
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            sweep_results["snr_db"],
            sweep_results["scores"],
            marker="o",
            linewidth=2,
        )
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(sweep_results["metric"].upper())
        ax.set_title("Model accuracy vs photonic noise level")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig
