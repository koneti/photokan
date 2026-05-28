"""Tests for the PhotonicBackend abstract base class."""

import pytest
import torch

from photokan.backends.base import PhotonicBackend


class TestPhotonicBackendAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            PhotonicBackend()

    def test_minimal_subclass_can_be_instantiated(self):
        class DummyBackend(PhotonicBackend):
            @staticmethod
            def name() -> str:
                return "dummy"

            @staticmethod
            def display_name() -> str:
                return "Dummy Backend"

            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def device_info() -> dict:
                return {"device": "dummy"}

            @staticmethod
            def execute(x: torch.Tensor, activation, op_type: str) -> torch.Tensor:
                return x

            @staticmethod
            def compute_gradient(
                grad_output: torch.Tensor,
                x: torch.Tensor,
                activation,
                op_type: str,
            ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
                return x, {}

            @staticmethod
            def noise_profiles() -> dict[str, dict]:
                return {"default": {"snr_db": 14.0}}

        backend = DummyBackend()
        assert backend.name() == "dummy"
        assert backend.display_name() == "Dummy Backend"
        assert backend.is_available() is False
        assert isinstance(backend.device_info(), dict)
        assert isinstance(backend.noise_profiles(), dict)

    def test_estimate_flops_default_is_empty_dict(self):
        class MinimalBackend(PhotonicBackend):
            @staticmethod
            def name() -> str:
                return "minimal"

            @staticmethod
            def display_name() -> str:
                return "Minimal"

            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def device_info() -> dict:
                return {}

            @staticmethod
            def execute(x, activation, op_type):
                return x

            @staticmethod
            def compute_gradient(grad_output, x, activation, op_type):
                return x, {}

            @staticmethod
            def noise_profiles():
                return {}

        backend = MinimalBackend()
        assert backend.estimate_flops(None) == {}
