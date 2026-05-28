"""Tests for symbolic regression utilities."""

import pytest

import sympy as sp
import torch

from photokan.activations import SineEdgeActivation
from photokan.utils.symbolic import symbolic_regress_activation


class TestSymbolicRegressActivation:
    @pytest.fixture()
    def activation(self):
        torch.manual_seed(42)
        return SineEdgeActivation(n_basis=4)

    def test_returns_sympy_expression(self, activation):
        result = symbolic_regress_activation(activation)
        assert isinstance(result, sp.Basic)

    def test_result_depends_on_x(self, activation):
        result = symbolic_regress_activation(activation)
        # The expression should contain the symbol 'x'
        assert sp.Symbol("x") in result.free_symbols or isinstance(result, sp.Symbol)
