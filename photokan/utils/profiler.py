# photokan/utils/profiler.py
"""Simple profiler context manager for PhotoKAN forward passes."""

from __future__ import annotations

import time


class Profiler:
    """
    Context manager that times a PhotoKAN forward pass.

    Usage::

        with pk.Profiler() as prof:
            y = model(x)
        print(prof.summary())
    """

    def __init__(self):
        self._start: float = 0.0
        self._end: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self._end = time.perf_counter()
        self.elapsed_ms = (self._end - self._start) * 1000.0

    def summary(self) -> dict:
        return {
            "elapsed_ms": round(self.elapsed_ms, 3),
            "note": "CPU wall-clock time. For NPU op timing use QPALBackend.get_device_info().",
        }
