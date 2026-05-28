"""Tests for energy cost estimator."""

import pytest
import torch

from photokan.layers import PhotoKAN, PhotoKANLayer
from photokan.utils.energy import EnergyReport, estimate_layer_energy, estimate_model_energy


class TestEstimateLayerEnergy:
    @pytest.fixture()
    def layer(self):
        torch.manual_seed(42)
        return PhotoKANLayer(3, 2, activation="sine", backend="cpu", noise_sim=False, n_basis=4)

    def test_returns_energy_report(self, layer):
        report = estimate_layer_energy(layer)
        assert isinstance(report, EnergyReport)

    def test_report_has_correct_fields(self, layer):
        report = estimate_layer_energy(layer)
        assert report.n_edges == 6  # 3 * 2
        assert report.n_ops_per_edge > 0
        assert report.total_ops > 0
        assert report.cmos_energy_uj > 0
        assert report.photonic_energy_uj > 0
        assert report.efficiency_ratio > 0
        assert isinstance(report.activation_type, str)

    def test_batch_size_affects_total_ops(self, layer):
        report_b1 = estimate_layer_energy(layer, batch_size=1)
        report_b8 = estimate_layer_energy(layer, batch_size=8)
        assert report_b8.total_ops == report_b1.total_ops * 8

    def test_cmos_more_expensive_than_photonic(self, layer):
        report = estimate_layer_energy(layer)
        assert report.cmos_energy_uj > report.photonic_energy_uj

    def test_efficiency_ratio_is_correct(self, layer):
        report = estimate_layer_energy(layer)
        expected_ratio = report.cmos_energy_uj / report.photonic_energy_uj
        assert abs(report.efficiency_ratio - expected_ratio) < 1e-6

    def test_summary_returns_string(self, layer):
        report = estimate_layer_energy(layer)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "Energy Report" in summary


class TestEstimateModelEnergy:
    @pytest.fixture()
    def model(self):
        torch.manual_seed(42)
        return PhotoKAN(
            layer_sizes=[2, 4, 1],
            activation="sine",
            backend="cpu",
            noise_sim=False,
            n_basis=4,
        )

    def test_returns_list_of_reports(self, model):
        reports = estimate_model_energy(model)
        assert isinstance(reports, list)
        assert len(reports) == 2  # [2,4,1] has 2 layers

    def test_each_report_is_energy_report(self, model):
        reports = estimate_model_energy(model)
        for report in reports:
            assert isinstance(report, EnergyReport)
            assert report.n_edges > 0

    def test_reports_match_layer_dimensions(self, model):
        reports = estimate_model_energy(model)
        # First layer: 2 in * 4 out = 8 edges
        assert reports[0].n_edges == 8
        # Second layer: 4 in * 1 out = 4 edges
        assert reports[1].n_edges == 4
