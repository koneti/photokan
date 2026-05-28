"""
Integration: train a model, compile it, run via LUT interpreter,
verify output fidelity end-to-end.
"""

import os
import tempfile

import pytest
import torch

from photokan.compiler import PhotonicCompiler, PhotonicProgram
from photokan.layers import PhotoKAN


def _make_and_train(activation="sine", n_basis=6, epochs=100):
    torch.manual_seed(1)
    model = PhotoKAN(
        [2, 6, 1],
        activation=activation,
        backend="cpu",
        noise_sim=False,
        n_basis=n_basis,
    )
    x = torch.rand(200, 2) * 3 - 1.5
    y = torch.sin(x[:, 0]) + x[:, 1] ** 2
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(epochs):
        loss = torch.nn.functional.mse_loss(model(x).squeeze(), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@pytest.mark.parametrize("activation", ["sine", "relu", "fourier"])
def test_compile_run_fidelity(activation):
    """Compiled LUT output should match model output within tolerance."""
    model = _make_and_train(activation=activation, epochs=50)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.npu")
        compiler = PhotonicCompiler(n_lut_points=256, max_lut_mse=1e-3)
        prog = compiler.compile(model, path, validate=False)

        x = torch.rand(64, 2) * 1.5 - 0.75  # within LUT range
        model.eval()
        with torch.no_grad():
            y_model = model(x)
        y_lut = prog.run(x, backend="cpu")

        mse = torch.nn.functional.mse_loss(y_lut, y_model).item()
        assert mse < 0.5, f"{activation}: LUT MSE {mse:.4f}"


def test_bundle_survives_disk_roundtrip():
    """Load from disk should produce same results as in-memory program."""
    model = _make_and_train(epochs=30)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.npu")
        prog_mem = PhotonicCompiler(n_lut_points=64, max_lut_mse=1e-2).compile(
            model, path, validate=False
        )
        prog_disk = PhotonicProgram.load(path)

        x = torch.rand(8, 2)
        y_mem = prog_mem.run(x, backend="cpu")
        y_disk = prog_disk.run(x, backend="cpu")

        assert torch.allclose(y_mem, y_disk, atol=1e-5)


def test_benchmark_returns_valid_stats():
    model = _make_and_train(epochs=10)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.npu")
        prog = PhotonicCompiler(n_lut_points=64, max_lut_mse=1e-2).compile(
            model, path, validate=False
        )
        stats = prog.benchmark(torch.rand(4, 2), n_runs=5)
        assert stats["mean_ms"] > 0
        assert stats["batch_size"] == 4
