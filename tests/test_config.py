"""Tests for config loading, device detection, and seed reproducibility."""

import torch
from omegaconf import DictConfig

from src.utils.config import load_config
from src.utils.device import get_device, set_seed


def test_load_default_config(default_config_path):
    """CONF-01: load_config returns DictConfig with correct default values."""
    cfg = load_config(config_path=default_config_path)
    assert isinstance(cfg, DictConfig)
    assert cfg.project.seed == 42
    assert cfg.model.name == "deepset/gbert-large"
    assert cfg.model.use_crf is False


def test_cli_override(default_config_path):
    """CONF-02: load_config with overrides changes the value."""
    cfg = load_config(config_path=default_config_path, overrides=["model.use_crf=true"])
    assert cfg.model.use_crf is True


def test_cli_override_nested(default_config_path):
    """CONF-02: load_config with nested overrides changes the value."""
    cfg = load_config(config_path=default_config_path, overrides=["training.batch_size=16"])
    assert cfg.training.batch_size == 16


def test_seed_reproducibility():
    """CONF-03: set_seed produces identical torch.randn outputs on repeated calls."""
    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_get_device_returns_valid():
    """get_device returns a torch.device with a valid type."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "cuda", "mps")
