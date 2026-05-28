"""Tests for QPALGraphBuilder."""
import pytest
from photokan.compiler import LUTCompiler, QPALGraphBuilder, ExecGraph
from photokan.layers import PhotoKAN


def _make_model():
    return PhotoKAN([2, 4, 1], activation="sine", backend="cpu",
                     noise_sim=False, n_basis=4)


def _compile_luts(model):
    return LUTCompiler(n_points=64, max_mse=1e-2).compile_model(
        model, validate=False
    )


class TestQPALGraphBuilder:

    def test_build_returns_exec_graph(self):
        model = _make_model()
        luts  = _compile_luts(model)
        graph = QPALGraphBuilder().build(model, luts)
        assert isinstance(graph, ExecGraph)

    def test_node_count(self):
        model = _make_model()
        luts  = _compile_luts(model)
        graph = QPALGraphBuilder().build(model, luts)
        # 2*4 + 4*1 = 12 edges
        assert len(graph.nodes) == 12

    def test_all_nodes_have_slot(self):
        model = _make_model()
        graph = QPALGraphBuilder(n_slots=8).build(model, _compile_luts(model))
        for node in graph.nodes:
            assert 0 <= node.slot < 8

    def test_layer_sizes_preserved(self):
        model = _make_model()
        graph = QPALGraphBuilder().build(model, _compile_luts(model))
        assert graph.layer_sizes == [2, 4, 1]

    def test_to_json_is_valid_json(self):
        import json
        model = _make_model()
        graph = QPALGraphBuilder().build(model, _compile_luts(model))
        parsed = json.loads(graph.to_json())
        assert "nodes" in parsed

    def test_optimise_preserves_node_count(self):
        model  = _make_model()
        builder = QPALGraphBuilder(n_slots=4)
        graph  = builder.build(model, _compile_luts(model))
        graph  = builder.optimise(graph)
        assert len(graph.nodes) == 12

    def test_lut_refs_unique_per_edge(self):
        model = _make_model()
        graph = QPALGraphBuilder().build(model, _compile_luts(model))
        refs  = [n.lut_ref for n in graph.nodes]
        assert len(refs) == len(set(refs))
