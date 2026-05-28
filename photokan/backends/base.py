# photokan/backends/base.py
"""
Abstract base class for photonic vendor backends.

All vendor backends (Q.ANT, Lightmatter, Salience Labs, etc.) must
implement this interface to integrate with the photokan dispatch layer.
"""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod


class PhotonicBackend(ABC):
    """Interface that every photonic vendor backend must implement."""

    @staticmethod
    @abstractmethod
    def name() -> str:
        """Machine-readable backend name (e.g. 'qant', 'lightmatter')."""

    @staticmethod
    @abstractmethod
    def display_name() -> str:
        """Human-readable vendor name (e.g. 'Q.ANT', 'Lightmatter')."""

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Return True if the vendor SDK and hardware are accessible."""

    @staticmethod
    @abstractmethod
    def device_info() -> dict:
        """Return a dict describing the connected device."""

    @staticmethod
    @abstractmethod
    def execute(x: torch.Tensor, activation, op_type: str) -> torch.Tensor:
        """Run a forward pass on the photonic hardware."""

    @staticmethod
    @abstractmethod
    def compute_gradient(
        grad_output: torch.Tensor,
        x: torch.Tensor,
        activation,
        op_type: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute gradients via the vendor's adjoint/backprop method."""

    @staticmethod
    @abstractmethod
    def noise_profiles() -> dict[str, dict]:
        """Return named noise profiles for this vendor's hardware."""

    @staticmethod
    def estimate_flops(layer) -> dict:
        """Optional: estimate FLOPs for a layer on this hardware."""
        return {}
