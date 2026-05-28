"""Tests for PhotonicCompiler and PhotonicProgram."""
import json
import os
import struct
import tempfile
import pytest
import torch
from photokan.compiler import PhotonicCompiler, PhotonicProgram
from photokan.layers import PhotoKAN
from photokan.backend.errors import PhotonicCompilerError


def _trained_model():
    torch.manual_seed(0)
    model = PhotoKAN([2, 4, 1], activation="sine", backend="cpu",
                      noise_sim=False, n_basis=4)
    x = torch.rand(32, 2)
    y = torch.sin(x[:, 0]) + x[:, 1] ** 2
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(10):
        loss = torch.nn.functional.mse_loss(model(x).squeeze(), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


class TestPhotonicCompiler:

    def test_compile_creates_bundle_files(self):
        model = _trained_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.npu")
            compiler = PhotonicCompiler(n_lut_points=64, max_lut_mse=1e-2)
            compiler.compile(model, path, validate=False)
            assert os.path.exists(os.path.join(path, "op_graph.json"))
            assert os.path.exists(os.path.join(path, "weights.bin"))
            assert os.path.exists(os.path.join(path, "metadata.json"))
            assert os.path.exists(os.path.join(path, "manifest.txt"))

    def test_metadata_content(self):
        model = _trained_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.npu")
            PhotonicCompiler(n_lut_points=64, max_lut_mse=1e-2).compile(
                model, path, validate=False
            )
            with open(os.path.join(path, "metadata.json")) as f:
                meta = json.load(f)
            assert meta["layer_sizes"] == [2, 4, 1]
            assert meta["activation"] == "sine"
            assert meta["phase"] == "2"
            assert "n_edges" in meta
            assert "avg_lut_mse" in meta

    def test_weights_bin_magic(self):
        model = _trained_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.npu")
            PhotonicCompiler(n_lut_points=64, max_lut_mse=1e-2).compile(
                model, path, validate=False
            )
            with open(os.path.join(path, "weights.bin"), "rb") as f:
                magic = f.read(4)
            assert magic == b"PKAN"

    def test_estimate_ops(self):
        model = _trained_model()
        compiler = PhotonicCompiler()
        ops = compiler.estimate_ops(model)
        assert "n_params" in ops
        assert "estimated_energy_uj" in ops
        assert ops["estimated_energy_uj"] > 0

    def test_compile_returns_program(self):
        model = _trained_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.npu")
            prog = PhotonicCompiler(n_lut_points=64, max_lut_mse=1e-2).compile(
                model, path, validate=False
            )
            assert isinstance(prog, PhotonicProgram)


class TestPhotonicProgram:

    @pytest.fixture
    def bundle(self, tmp_path):
        model = _trained_model()
        path = str(tmp_path / "model.npu")
        PhotonicCompiler(n_lut_points=64, max_lut_mse=1e-2).compile(
            model, path, validate=False
        )
        return path

    def test_load_from_disk(self, bundle):
        prog = PhotonicProgram.load(bundle)
        assert isinstance(prog.metadata, dict)
        assert prog.metadata["layer_sizes"] == [2, 4, 1]

    def test_inspect(self, bundle):
        prog = PhotonicProgram.load(bundle)
        info = prog.inspect()
        assert "n_edges" in info
        assert "layer_sizes" in info

    def test_run_cpu_shape(self, bundle):
        prog = PhotonicProgram.load(bundle)
        x = torch.rand(8, 2)
        y = prog.run(x, backend="cpu")
        assert y.shape == (8, 1)

    def test_run_cpu_output_finite(self, bundle):
        prog = PhotonicProgram.load(bundle)
        x = torch.rand(16, 2)
        y = prog.run(x, backend="cpu")
        assert torch.isfinite(y).all()

    def test_run_vs_model_accuracy(self, bundle):
        """LUT interpreter output should be close to original model."""
        model = _trained_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.npu")
            PhotonicCompiler(n_lut_points=256, max_lut_mse=1e-4).compile(
                model, path, validate=False
            )
            prog = PhotonicProgram.load(path)
            x = torch.rand(32, 2) * 2 - 1   # within LUT range
            model.eval()
            with torch.no_grad():
                y_model = model(x)
            y_prog = prog.run(x, backend="cpu")
            mse = torch.nn.functional.mse_loss(y_prog, y_model).item()
            assert mse < 0.1, f"LUT interpreter MSE {mse:.4f} too high"

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            PhotonicProgram.load("/nonexistent/path")

    def test_benchmark_cpu(self, bundle):
        prog = PhotonicProgram.load(bundle)
        x = torch.rand(4, 2)
        result = prog.benchmark(x, n_runs=5, backend="cpu")
        assert "mean_ms" in result
        assert result["mean_ms"] > 0
        assert result["throughput_samples_per_sec"] > 0
