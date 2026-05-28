"""Tests for PhotonicSimulator — pre-deployment analysis tool."""

import pytest
import torch

from photokan.layers import PhotoKAN, PhotoKANLayer
from photokan.sim.photonic_sim import PhotonicSimulator


class TestPhotonicSimulator:
    """Tests for the PhotonicSimulator class."""

    @pytest.fixture()
    def small_model(self):
        torch.manual_seed(42)
        return PhotoKAN(
            layer_sizes=[2, 4, 1],
            activation="sine",
            backend="cpu",
            noise_sim=False,
            n_basis=4,
        )

    @pytest.fixture()
    def test_data(self):
        torch.manual_seed(0)
        x = torch.randn(8, 2)
        y = torch.randn(8, 1)
        return x, y

    def test_constructor_with_valid_profile(self):
        sim = PhotonicSimulator(profile="npu1")
        assert sim._profile == "npu1"
        assert sim._noise_config["enabled"] is True

    def test_constructor_with_all_profiles(self):
        for profile in ("npu1", "npu2", "ideal", "custom"):
            sim = PhotonicSimulator(profile=profile)
            assert sim._profile == profile

    def test_constructor_with_invalid_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            PhotonicSimulator(profile="nonexistent")

    def test_set_hardware_profile_switches(self):
        sim = PhotonicSimulator(profile="npu1")
        sim.set_hardware_profile("npu2")
        assert sim._profile == "npu2"
        assert sim._noise_config["snr_db"] == 16.0

    def test_set_hardware_profile_invalid_raises(self):
        sim = PhotonicSimulator(profile="npu1")
        with pytest.raises(ValueError, match="Unknown profile"):
            sim.set_hardware_profile("invalid_profile")

    def test_set_noise_model_custom_params(self):
        sim = PhotonicSimulator(profile="npu1")
        sim.set_noise_model(snr_db=20.0, bit_depth=10)
        assert sim._profile == "custom"
        assert sim._noise_config["snr_db"] == 20.0
        assert sim._noise_config["bit_depth"] == 10

    def test_sweep_snr_returns_correct_structure(self, small_model, test_data):
        sim = PhotonicSimulator(profile="npu1")
        x, y = test_data
        snr_range = [8.0, 12.0, 16.0]
        results = sim.sweep_snr(small_model, x, y, snr_range=snr_range, metric="mse")

        assert "snr_db" in results
        assert "scores" in results
        assert "metric" in results
        assert results["snr_db"] == snr_range
        assert len(results["scores"]) == len(snr_range)
        assert results["metric"] == "mse"

    def test_sweep_snr_mae_metric(self, small_model, test_data):
        sim = PhotonicSimulator(profile="npu1")
        x, y = test_data
        results = sim.sweep_snr(small_model, x, y, snr_range=[10.0, 14.0], metric="mae")
        assert results["metric"] == "mae"
        assert len(results["scores"]) == 2

    def test_sweep_snr_without_labels(self, small_model, test_data):
        sim = PhotonicSimulator(profile="npu1")
        x, _ = test_data
        results = sim.sweep_snr(small_model, x, snr_range=[10.0])
        assert len(results["scores"]) == 1

    def test_sweep_snr_default_range(self, small_model, test_data):
        sim = PhotonicSimulator(profile="npu1")
        x, _ = test_data
        results = sim.sweep_snr(small_model, x)
        assert len(results["scores"]) == 6  # default [8,10,12,14,16,20]

    def test_sweep_snr_with_layer(self, test_data):
        torch.manual_seed(42)
        layer = PhotoKANLayer(2, 1, activation="sine", backend="cpu", noise_sim=True, n_basis=4)
        x, _ = test_data
        sim = PhotonicSimulator(profile="npu1")
        results = sim.sweep_snr(layer, x, snr_range=[10.0, 14.0])
        assert len(results["scores"]) == 2

    def test_plot_transfer_function_returns_figure(self):
        from photokan.activations import SineEdgeActivation

        torch.manual_seed(42)
        sim = PhotonicSimulator(profile="npu1")
        activation = SineEdgeActivation(n_basis=4)
        fig = sim.plot_transfer_function(activation)

        import matplotlib.figure

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_snr_accuracy_returns_figure(self, small_model, test_data):
        sim = PhotonicSimulator(profile="npu1")
        x, y = test_data
        results = sim.sweep_snr(small_model, x, y, snr_range=[8.0, 12.0, 16.0])
        fig = sim.plot_snr_accuracy(results)

        import matplotlib.figure

        assert isinstance(fig, matplotlib.figure.Figure)
