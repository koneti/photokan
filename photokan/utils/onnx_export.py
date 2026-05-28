# photokan/utils/onnx_export.py
"""
ONNX export for PhotoKAN models — non-Q.PAL deployment path.

Traces the model through torch.onnx.export with a dummy input,
producing a standard ONNX graph that can run on ONNX Runtime,
TensorRT, CoreML, or any other ONNX-compatible backend.

Note: exported model runs the CPU/GPU forward path (no photonic ops).
Use PhotonicCompiler for Q.PAL hardware deployment.
"""

from __future__ import annotations

import os

import torch


def export_onnx(
    model,
    output_path: str,
    input_shape: tuple[int, ...] | None = None,
    opset_version: int = 17,
    dynamic_axes: dict | None = None,
) -> str:
    """
    Export a PhotoKAN model to ONNX format.

    Args:
        model        : PhotoKAN or PhotoKANLayer instance.
        output_path  : File path for the .onnx output.
        input_shape  : Shape of one input sample excluding batch dim,
                       e.g. (4,) for a 4-feature model.
                       Inferred from layer_sizes[0] when None.
        opset_version: ONNX opset (default 17).
        dynamic_axes : Dict of dynamic axis specs, e.g.
                       {'input': {0: 'batch'}, 'output': {0: 'batch'}}.
                       Defaults to batch-dynamic for input + output.

    Returns:
        Absolute path to the written .onnx file.

    Example::

        export_onnx(model, './model.onnx')
        # run with onnxruntime:
        # import onnxruntime as ort
        # sess = ort.InferenceSession('./model.onnx')
        # y = sess.run(None, {'input': x.numpy()})[0]
    """
    if not output_path.endswith(".onnx"):
        output_path = output_path + ".onnx"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Infer input size
    if input_shape is None:
        if hasattr(model, "layer_sizes"):
            in_f = model.layer_sizes[0]
        elif hasattr(model, "in_features"):
            in_f = model.in_features
        else:
            raise ValueError("Cannot infer input shape. Pass input_shape=(n_features,) explicitly.")
        input_shape = (in_f,)

    # Disable noise for export (deterministic graph)
    was_training = model.training
    model.eval()
    _disable_noise(model)

    dummy = torch.zeros(1, *input_shape)

    if dynamic_axes is None:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    _restore_noise(model)
    if was_training:
        model.train()

    return os.path.abspath(output_path)


def _disable_noise(model):
    """Temporarily disable noise simulation for ONNX tracing."""
    for module in model.modules():
        if hasattr(module, "noise_config") and module.noise_config:
            module._noise_config_backup = module.noise_config.copy()
            module.noise_config = dict(module.noise_config, enabled=False)


def _restore_noise(model):
    """Restore noise config after ONNX export."""
    for module in model.modules():
        if hasattr(module, "_noise_config_backup"):
            module.noise_config = module._noise_config_backup
            del module._noise_config_backup
