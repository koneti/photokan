"""Tests for ONNX export — requires photokan[onnx]."""

import os
import tempfile

import pytest

onnxscript = pytest.importorskip("onnxscript")

import torch  # noqa: E402

from photokan.layers import PhotoKAN  # noqa: E402
from photokan.utils.onnx_export import export_onnx  # noqa: E402


def _tiny_trained():
    torch.manual_seed(0)
    model = PhotoKAN([2, 4, 1], activation="relu", backend="cpu", noise_sim=False, n_basis=4)
    x = torch.rand(32, 2)
    y = x[:, 0] + x[:, 1]
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(5):
        loss = torch.nn.functional.mse_loss(model(x).squeeze(), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


class TestONNXExport:
    def test_creates_onnx_file(self):
        model = _tiny_trained()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_onnx(model, os.path.join(tmpdir, "model.onnx"))
            assert os.path.exists(path)
            assert path.endswith(".onnx")

    def test_adds_extension_if_missing(self):
        model = _tiny_trained()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_onnx(model, os.path.join(tmpdir, "model"))
            assert path.endswith(".onnx")

    def test_onnx_output_matches_pytorch(self):
        pytest.importorskip("onnxruntime")
        import numpy as np
        import onnxruntime as ort

        model = _tiny_trained()
        model.eval()
        x = torch.rand(4, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_onnx(model, os.path.join(tmpdir, "model.onnx"))
            sess = ort.InferenceSession(path)
            y_ort = sess.run(None, {"input": x.numpy()})[0]

        with torch.no_grad():
            y_pt = model(x).numpy()

        assert np.allclose(y_ort, y_pt, atol=1e-4), f"Max diff: {abs(y_ort - y_pt).max():.2e}"

    def test_no_input_shape_raises_for_plain_module(self):
        import torch.nn as nn

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Cannot infer input shape"):
                export_onnx(nn.Sequential(nn.ReLU()), os.path.join(tmpdir, "m.onnx"))
