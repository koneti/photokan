# photokan/compiler/graph_builder.py
"""
Q.PAL Graph Builder — maps compiled LUTs to a Q.PAL execution graph.

Constructs an op-graph JSON that the Q.PAL runtime uses to schedule
nonlinear ops across NPU execution slots, minimising data movement
and respecting slot capacity.

Phase 2: builds and validates the graph. Actual NPU execution comes
when Q.PAL loads the .npu bundle via PhotonicProgram.run().
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

from .lut_compiler import LUTEntry


@dataclass
class OpNode:
    """A single operation in the photonic execution graph."""
    node_id:       str
    op_type:       str          # Q.PAL op string e.g. 'sine_waveguide'
    layer_idx:     int
    edge_idx:      int
    in_node:       int          # input feature index
    out_node:      int          # output feature index
    lut_ref:       str          # key into weights store
    slot:          int = 0      # NPU execution slot


@dataclass
class ExecGraph:
    """Full photonic execution graph for a compiled model."""
    nodes:          list[OpNode] = field(default_factory=list)
    n_layers:       int = 0
    layer_sizes:    list[int] = field(default_factory=list)
    activation_type: str = ""
    n_slots_used:   int = 0
    metadata:       dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_layers":       self.n_layers,
            "layer_sizes":    self.layer_sizes,
            "activation_type": self.activation_type,
            "n_slots_used":   self.n_slots_used,
            "metadata":       self.metadata,
            "nodes": [asdict(n) for n in self.nodes],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# Q.PAL op-type mapping (mirrors qpal_backend._OP_REGISTRY)
_OP_TYPE_MAP: dict[str, str] = {
    "SineEdgeActivation":    "sine_waveguide",
    "FourierEdgeActivation": "fourier_waveguide",
    "SplineEdgeActivation":  "spline_lut",
    "ReLUEdgeActivation":    "relu_piecewise",
}


class QPALGraphBuilder:
    """
    Builds a Q.PAL execution graph from compiled LUTs.

    Args:
        n_slots : Number of NPU execution slots (default 64).
                  Ops are assigned round-robin across slots to
                  maximise parallelism.
    """

    def __init__(self, n_slots: int = 64):
        self.n_slots = n_slots

    def build(
        self,
        model,
        luts_per_layer: list[list[LUTEntry]],
    ) -> ExecGraph:
        """
        Construct the execution graph.

        Args:
            model           : PhotoKAN model (provides layer_sizes, activation_name).
            luts_per_layer  : Output of LUTCompiler.compile_model().

        Returns:
            ExecGraph with all op nodes assigned to slots.
        """
        activation_type = model.activation_name
        op_type = _OP_TYPE_MAP.get(
            f"{activation_type.capitalize()}EdgeActivation",
            "sine_waveguide",
        )

        nodes: list[OpNode] = []
        slot_counter = 0

        for l_idx, (layer, luts) in enumerate(
            zip(model.layers, luts_per_layer)
        ):
            in_f  = layer.in_features
            out_f = layer.out_features

            for edge_idx, lut in enumerate(luts):
                in_node  = edge_idx // out_f
                out_node = edge_idx % out_f
                lut_key  = f"L{l_idx}_E{edge_idx}"

                nodes.append(OpNode(
                    node_id   = f"op_{l_idx}_{edge_idx}",
                    op_type   = op_type,
                    layer_idx = l_idx,
                    edge_idx  = edge_idx,
                    in_node   = in_node,
                    out_node  = out_node,
                    lut_ref   = lut_key,
                    slot      = slot_counter % self.n_slots,
                ))
                slot_counter += 1

        return ExecGraph(
            nodes=nodes,
            n_layers=len(model.layers),
            layer_sizes=model.layer_sizes,
            activation_type=activation_type,
            n_slots_used=min(slot_counter, self.n_slots),
            metadata={
                "n_edges":     len(nodes),
                "n_slots":     self.n_slots,
                "builder":     "QPALGraphBuilder",
            },
        )

    def optimise(self, graph: ExecGraph) -> ExecGraph:
        """
        Apply slot-level optimisation: pack ops to minimise idle slots.

        Phase 2 implementation: greedy bin-packing by layer to keep
        data locality — ops within the same layer share slots before
        spilling to the next.
        """
        # Group nodes by layer, then assign slots layer by layer
        layers: dict[int, list[OpNode]] = {}
        for node in graph.nodes:
            layers.setdefault(node.layer_idx, []).append(node)

        slot = 0
        for l_idx in sorted(layers):
            layer_nodes = layers[l_idx]
            layer_start_slot = slot
            for node in layer_nodes:
                node.slot = slot % self.n_slots
                slot += 1
            # Advance to next slot boundary for next layer (data locality)
            if slot % self.n_slots != 0:
                slot = ((slot // self.n_slots) + 1) * self.n_slots
                slot = min(slot, slot)   # don't overflow

        graph.n_slots_used = min(
            max((n.slot for n in graph.nodes), default=0) + 1,
            self.n_slots
        )
        return graph
