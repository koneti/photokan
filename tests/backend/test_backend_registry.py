"""Tests for vendor-agnostic backend registry."""
import pytest
from photokan.backends import (
    available_backends,
    resolve_backend,
    get_backend,
    all_vendor_names,
    get_noise_config,
    PhotonicBackendError,
)
from photokan.backends.base import PhotonicBackend


class TestBackendRegistry:

    def test_available_backends_returns_dict(self):
        backends = available_backends()
        assert isinstance(backends, dict)
        assert "cpu" in backends
        assert "cuda" in backends
        assert "qant" in backends
        assert "lightmatter" in backends
        assert "salience" in backends

    def test_cpu_always_available(self):
        assert available_backends()["cpu"] is True

    def test_resolve_auto_returns_valid_backend(self):
        result = resolve_backend("auto")
        assert result in ("cpu", "cuda", "qant", "lightmatter", "salience")

    def test_resolve_explicit_unchanged(self):
        assert resolve_backend("cpu") == "cpu"
        assert resolve_backend("cuda") == "cuda"
        assert resolve_backend("qant") == "qant"

    def test_all_vendor_names(self):
        names = all_vendor_names()
        assert "qant" in names
        assert "lightmatter" in names
        assert "salience" in names

    def test_get_backend_known_vendor(self):
        cls = get_backend("qant")
        assert issubclass(cls, PhotonicBackend)

    def test_get_backend_unknown_raises(self):
        with pytest.raises(PhotonicBackendError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_get_noise_config_valid(self):
        config = get_noise_config("qant", "npu1")
        assert config["enabled"] is True
        assert "snr_db" in config

    def test_get_noise_config_unknown_vendor(self):
        with pytest.raises(PhotonicBackendError, match="Unknown backend"):
            get_noise_config("nonexistent", "npu1")

    def test_get_noise_config_unknown_profile(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            get_noise_config("qant", "nonexistent")

    def test_lightmatter_noise_profiles(self):
        config = get_noise_config("lightmatter", "envise1")
        assert config["technology"] == "silicon_photonics"

    def test_salience_noise_profiles(self):
        config = get_noise_config("salience", "mr100")
        assert config["technology"] == "iii_v_photonics"
