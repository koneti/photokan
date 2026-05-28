"""Tests for the Profiler context manager."""

import time

import pytest
import torch

from photokan.layers import PhotoKAN
from photokan.utils.profiler import Profiler


class TestProfiler:
    def test_context_manager_records_elapsed_ms(self):
        prof = Profiler()
        with prof:
            time.sleep(0.05)

        assert prof.elapsed_ms > 0
        assert prof.elapsed_ms >= 40  # at least ~50ms minus tolerance

    def test_summary_returns_dict_with_elapsed_ms(self):
        prof = Profiler()
        with prof:
            x = torch.randn(100, 100)
            _ = x @ x

        summary = prof.summary()
        assert isinstance(summary, dict)
        assert "elapsed_ms" in summary
        assert summary["elapsed_ms"] > 0

    def test_summary_includes_note(self):
        prof = Profiler()
        with prof:
            pass

        summary = prof.summary()
        assert "note" in summary

    def test_profiler_with_model_forward(self):
        torch.manual_seed(42)
        model = PhotoKAN(
            layer_sizes=[2, 4, 1],
            activation="sine",
            backend="cpu",
            noise_sim=False,
            n_basis=4,
        )
        x = torch.randn(16, 2)

        with Profiler() as prof:
            model(x)

        assert prof.elapsed_ms > 0

    def test_elapsed_ms_before_use_is_zero(self):
        prof = Profiler()
        assert prof.elapsed_ms == 0.0
