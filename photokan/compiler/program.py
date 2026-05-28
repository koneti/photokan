# photokan/compiler/program.py
"""
PhotonicCompiler (Phase 2) and PhotonicProgram.

Full AOT compilation pipeline:
  1. LUTCompiler  — sample & int8-quantise all edge activations
  2. QPALGraphBuilder — assign ops to NPU slots, produce op_graph.json
  3. Bundle writer — weights.bin (raw int8), metadata.json, manifest.txt

PhotonicProgram.run() executes the bundle on Q.PAL when available,
falling back to a pure-Python LUT interpreter for CPU validation.
"""

from __future__ import annotations

import json
import os
import struct
import time

import numpy as np
import torch

from ..backends.errors import PhotonicCompilerError, PhotonicHardwareError
from ..backends.qant.backend import QANTBackend
from .graph_builder import ExecGraph, QPALGraphBuilder
from .lut_compiler import LUTCompiler, LUTEntry

# ---------------------------------------------------------------------------
# Bundle format constants
# ---------------------------------------------------------------------------
BUNDLE_VERSION = "2.0"
WEIGHTS_MAGIC = b"PKAN"  # 4-byte file header for weights.bin


# ---------------------------------------------------------------------------
# PhotonicProgram
# ---------------------------------------------------------------------------


class PhotonicProgram:
    """
    A compiled photonic deployment bundle (.npu directory).

    Bundle layout::

        my_model.npu/
        ├── op_graph.json     ← execution graph (QPALGraphBuilder output)
        ├── weights.bin       ← packed int8 LUT values with header
        ├── metadata.json     ← model info, compile options, op stats
        └── manifest.txt      ← version, hashes, Q.PAL requirements

    Usage::

        program = PhotonicCompiler().compile(model, './my_model.npu')

        # CPU validation (LUT interpreter)
        y = program.run(x, backend='cpu')

        # NPU inference (requires Q.PAL)
        y = program.run(x, backend='qpal')
    """

    def __init__(
        self,
        path: str,
        metadata: dict,
        graph: ExecGraph | None = None,
        luts: list[list[LUTEntry]] | None = None,
    ):
        self.path = path
        self.metadata = metadata
        self._graph = graph  # kept in memory after compile
        self._luts = luts  # kept in memory after compile

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> PhotonicProgram:
        """Load a compiled bundle from disk."""
        meta_path = os.path.join(path, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Not a valid .npu bundle (missing metadata.json): {path}")
        with open(meta_path) as f:
            metadata = json.load(f)
        return cls(path, metadata)

    def inspect(self) -> dict:
        """Return bundle metadata and graph summary."""
        info = dict(self.metadata)
        graph_path = os.path.join(self.path, "op_graph.json")
        if os.path.exists(graph_path):
            with open(graph_path) as f:
                g = json.load(f)
            info["n_ops"] = len(g.get("nodes", []))
            info["n_slots_used"] = g.get("n_slots_used", 0)
        return info

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(
        self,
        x: torch.Tensor,
        backend: str = "auto",
    ) -> torch.Tensor:
        """
        Run inference using the compiled bundle.

        Args:
            x       : Input tensor [batch, in_features].
            backend : 'auto', 'qpal' (NPU), or 'cpu' (LUT interpreter).

        Returns:
            Output tensor [batch, out_features].
        """
        use_npu = backend in ("auto", "qpal") and QANTBackend.is_available()

        if backend == "qpal" and not use_npu:
            raise PhotonicHardwareError("backend='qpal' requested but Q.ANT NPU is not available.")

        if use_npu:
            return self._run_npu(x)
        else:
            return self._run_lut_cpu(x)

    def _run_npu(self, x: torch.Tensor) -> torch.Tensor:
        """Execute bundle on Q.ANT NPU via Q.PAL runtime."""
        try:
            import qpal as _qpal  # type: ignore[import]
        except ImportError as exc:
            raise PhotonicHardwareError("Q.PAL SDK not installed. Cannot run on NPU.") from exc
        graph_path = os.path.join(self.path, "op_graph.json")
        weights_path = os.path.join(self.path, "weights.bin")
        return _qpal.run_bundle(x, graph_path, weights_path)

    def _run_lut_cpu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pure-Python LUT interpreter — CPU validation path.

        Loads LUTs from weights.bin and evaluates the model by
        linear interpolation within each LUT segment.
        """
        graph_path = os.path.join(self.path, "op_graph.json")
        weights_path = os.path.join(self.path, "weights.bin")

        if not os.path.exists(graph_path):
            raise FileNotFoundError(
                "op_graph.json not found in bundle. Re-compile with PhotonicCompiler."
            )

        with open(graph_path) as f:
            graph = json.load(f)

        lut_store = _load_weights_bin(weights_path)

        layer_sizes = graph["layer_sizes"]
        batch = x.shape[0]
        current = x

        # Replay the graph layer by layer
        layers_by_idx: dict[int, list[dict]] = {}
        for node in graph["nodes"]:
            layers_by_idx.setdefault(node["layer_idx"], []).append(node)

        for l_idx in sorted(layers_by_idx.keys()):
            nodes = sorted(layers_by_idx[l_idx], key=lambda n: n["edge_idx"])
            in_f = layer_sizes[l_idx]
            out_f = layer_sizes[l_idx + 1]
            out = torch.zeros(batch, out_f, dtype=current.dtype)

            for node in nodes:
                lut_entry = lut_store[node["lut_ref"]]
                in_i = node["in_node"]
                out_j = node["out_node"]
                phi = _lut_interpolate(current[:, in_i], lut_entry)
                out[:, out_j] = out[:, out_j] + phi

            current = out

        return current

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def benchmark(
        self,
        x: torch.Tensor,
        n_runs: int = 20,
        backend: str = "cpu",
    ) -> dict:
        """
        Benchmark inference latency.

        Returns:
            dict with mean_ms, std_ms, throughput_samples_per_sec.
        """
        # Warmup
        for _ in range(3):
            self.run(x, backend=backend)

        times_ms = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.run(x, backend=backend)
            times_ms.append((time.perf_counter() - t0) * 1000)

        mean_ms = float(np.mean(times_ms))
        std_ms = float(np.std(times_ms))
        throughput = x.shape[0] / (mean_ms / 1000)

        return {
            "mean_ms": round(mean_ms, 3),
            "std_ms": round(std_ms, 3),
            "min_ms": round(float(np.min(times_ms)), 3),
            "max_ms": round(float(np.max(times_ms)), 3),
            "throughput_samples_per_sec": round(throughput, 1),
            "backend": backend,
            "n_runs": n_runs,
            "batch_size": x.shape[0],
        }


# ---------------------------------------------------------------------------
# PhotonicCompiler
# ---------------------------------------------------------------------------


class PhotonicCompiler:
    """
    AOT compiler: trained PhotoKAN → photonic deployment bundle.

    Args:
        n_lut_points : LUT resolution per edge activation (default 256).
        lut_x_range  : Input domain for LUT sampling (default (-2.0, 2.0)).
        max_lut_mse  : Maximum allowable LUT reconstruction MSE (default 1e-4).
        n_npu_slots  : NPU execution slots (default 64).
        quantization : Weight quantisation scheme ('int8' only in Phase 2).
    """

    def __init__(
        self,
        n_lut_points: int = 256,
        lut_x_range: tuple[float, float] = (-2.0, 2.0),
        max_lut_mse: float = 1e-4,
        n_npu_slots: int = 64,
        quantization: str = "int8",
    ):
        self.n_lut_points = n_lut_points
        self.lut_x_range = lut_x_range
        self.max_lut_mse = max_lut_mse
        self.n_npu_slots = n_npu_slots
        self.quantization = quantization

        self._lut_compiler = LUTCompiler(
            n_points=n_lut_points,
            x_range=lut_x_range,
            max_mse=max_lut_mse,
        )
        self._graph_builder = QPALGraphBuilder(n_slots=n_npu_slots)

    def compile(
        self,
        model,
        output_path: str,
        validate: bool = True,
        optimise_graph: bool = True,
    ) -> PhotonicProgram:
        """
        Compile a trained PhotoKAN to a .npu deployment bundle.

        Pipeline:
            1. LUT compilation (sample + int8 quantise all edge activations)
            2. Graph construction (assign ops to NPU slots)
            3. Optional graph optimisation (slot packing)
            4. Bundle write (op_graph.json, weights.bin, metadata.json, manifest.txt)

        Args:
            model          : Trained PhotoKAN instance.
            output_path    : Directory path for the .npu bundle.
            validate       : Raise if LUT MSE > max_lut_mse.
            optimise_graph : Apply slot-packing optimisation.

        Returns:
            PhotonicProgram pointing at the written bundle.
        """
        os.makedirs(output_path, exist_ok=True)

        # Step 1: LUT compilation
        luts = self._lut_compiler.compile_model(model, validate=validate)

        # Step 2: Graph construction
        graph = self._graph_builder.build(model, luts)

        # Step 3: Optional optimisation
        if optimise_graph:
            graph = self._graph_builder.optimise(graph)

        # Step 4: Write bundle
        self._write_graph(graph, output_path)
        self._write_weights(luts, graph, output_path)
        metadata = self._write_metadata(model, graph, luts, output_path)
        self._write_manifest(output_path, metadata)

        return PhotonicProgram(output_path, metadata, graph, luts)

    def estimate_ops(self, model) -> dict:
        """
        Estimate op counts and efficiency vs MLP baseline.

        Returns:
            Dict with n_params, n_edges, mlp_equivalent_params,
            param_ratio, estimated_energy_uj.
        """
        base = model.estimate_ops()

        # Energy estimate using published Q.ANT figures
        total_edges = sum(layer.in_features * layer.out_features for layer in model.layers)
        act = model.layers[0].edge_activations[0]
        ops_per_edge = sum(p.numel() for p in act.parameters())
        photonic_energy_uj = total_edges * ops_per_edge * 0.12e-6  # 0.12 pJ/op

        return {**base, "estimated_energy_uj": round(photonic_energy_uj, 8)}

    # ------------------------------------------------------------------
    # Private bundle writers
    # ------------------------------------------------------------------

    def _write_graph(self, graph: ExecGraph, path: str) -> None:
        with open(os.path.join(path, "op_graph.json"), "w") as f:
            f.write(graph.to_json())

    def _write_weights(
        self,
        luts: list[list[LUTEntry]],
        graph: ExecGraph,
        path: str,
    ) -> None:
        """
        Write weights.bin: packed int8 LUT values.

        Format:
            4 bytes  : magic "PKAN"
            4 bytes  : version uint32
            4 bytes  : n_entries uint32
            Per entry:
              16 bytes : key as null-padded ASCII (e.g. "L0_E3")
              4 bytes  : n_points uint32
              4 bytes  : scale float32
              4 bytes  : zero_point int32
              n_points : int8 values
        """
        entries: dict[str, LUTEntry] = {}
        for l_idx, layer_luts in enumerate(luts):
            for e_idx, lut in enumerate(layer_luts):
                entries[f"L{l_idx}_E{e_idx}"] = lut

        with open(os.path.join(path, "weights.bin"), "wb") as f:
            f.write(WEIGHTS_MAGIC)
            f.write(struct.pack("<I", 2))  # version
            f.write(struct.pack("<I", len(entries)))  # n_entries

            for key, lut in entries.items():
                key_bytes = key.encode("ascii").ljust(16, b"\x00")[:16]
                f.write(key_bytes)
                f.write(struct.pack("<I", len(lut.values_int8)))
                f.write(struct.pack("<f", lut.scale))
                f.write(struct.pack("<i", lut.zero_point))
                f.write(lut.values_int8.tobytes())

    def _write_metadata(
        self,
        model,
        graph: ExecGraph,
        luts: list[list[LUTEntry]],
        path: str,
    ) -> dict:
        avg_mse = float(np.mean([lut.mse_error for layer_luts in luts for lut in layer_luts]))
        metadata = {
            "version": BUNDLE_VERSION,
            "layer_sizes": model.layer_sizes,
            "activation": model.activation_name,
            "backend": model.backend_mode,
            "quantization": self.quantization,
            "n_lut_points": self.n_lut_points,
            "lut_x_range": list(self.lut_x_range),
            "n_edges": graph.metadata["n_edges"],
            "n_slots_used": graph.n_slots_used,
            "avg_lut_mse": avg_mse,
            "phase": "2",
            **model.estimate_ops(),
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        return metadata

    def _write_manifest(self, path: str, metadata: dict) -> None:
        lines = [
            "PhotoKAN Bundle Manifest",
            f"version: {BUNDLE_VERSION}",
            "qpal_min_version: 1.0.0",
            "npu_required: false",
            f"n_edges: {metadata['n_edges']}",
            f"activation: {metadata['activation']}",
            f"quantization: {metadata['quantization']}",
            f"avg_lut_mse: {metadata['avg_lut_mse']:.2e}",
        ]
        with open(os.path.join(path, "manifest.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# LUT interpreter helpers
# ---------------------------------------------------------------------------


def _load_weights_bin(path: str) -> dict[str, dict]:
    """Parse weights.bin into a dict of {key: {values, scale, zero_point}}."""
    entries = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != WEIGHTS_MAGIC:
            raise PhotonicCompilerError(
                f"Invalid weights.bin magic: {magic!r} (expected {WEIGHTS_MAGIC!r})"
            )
        _version = struct.unpack("<I", f.read(4))[0]
        n_entries = struct.unpack("<I", f.read(4))[0]

        for _ in range(n_entries):
            key_raw = f.read(16)
            key = key_raw.rstrip(b"\x00").decode("ascii")
            n_points = struct.unpack("<I", f.read(4))[0]
            scale = struct.unpack("<f", f.read(4))[0]
            zero_point = struct.unpack("<i", f.read(4))[0]
            values = np.frombuffer(f.read(n_points), dtype=np.int8).copy()
            entries[key] = {
                "values": values,
                "scale": scale,
                "zero_point": zero_point,
            }
    return entries


def _lut_interpolate(x: torch.Tensor, lut_entry: dict) -> torch.Tensor:
    """
    Evaluate a LUT at arbitrary x values via linear interpolation.

    Args:
        x         : [batch] input values.
        lut_entry : dict with values (int8 np.array), scale, zero_point.

    Returns:
        [batch] output values.
    """
    values = lut_entry["values"]
    scale = lut_entry["scale"]
    zero_point = lut_entry["zero_point"]
    n = len(values)

    # Dequantise LUT
    y_lut = (values.astype(np.float32) - zero_point) * scale
    y_t = torch.tensor(y_lut, dtype=torch.float32)

    # Map x to fractional LUT index (hardcoded x_range from bundle — Phase 2
    # stores this per-entry in metadata.json; here we use the graph's known range)
    # For robustness we clamp to [0, n-1]
    x_np = x.detach().float().numpy()
    # x_range stored in weights is not per-entry in this format yet;
    # the LUT is sampled over (-2, 2) by default
    x_min, x_max = -2.0, 2.0
    idx_f = (x_np - x_min) / (x_max - x_min) * (n - 1)
    idx_f = np.clip(idx_f, 0, n - 1)

    idx_lo = np.floor(idx_f).astype(int)
    idx_hi = np.minimum(idx_lo + 1, n - 1)
    frac = idx_f - idx_lo

    y_lo = y_t[idx_lo]
    y_hi = y_t[idx_hi]
    result = y_lo + torch.tensor(frac, dtype=torch.float32) * (y_hi - y_lo)
    return result
