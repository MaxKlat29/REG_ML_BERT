"""Configuration loading with YAML defaults and CLI overrides."""

import sys
from omegaconf import OmegaConf, DictConfig


def load_config(
    config_path: str = "config/default.yaml",
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load YAML config and apply CLI overrides.

    Always loads config/default.yaml as base. If config_path differs,
    merges it on top (allows partial override files like gpu.yaml).

    Args:
        config_path: Path to config file (default.yaml or an override file).
        overrides: List of "key=value" strings. If None, parses sys.argv[1:].

    Returns:
        Merged DictConfig with dot-notation access.
    """
    base = OmegaConf.load("config/default.yaml")
    if config_path != "config/default.yaml":
        overlay = OmegaConf.load(config_path)
        base = OmegaConf.merge(base, overlay)
    if overrides is not None:
        cli = OmegaConf.from_dotlist(overrides)
    else:
        args = [a.lstrip("-") if "=" in a else a for a in sys.argv[1:]]
        cli = OmegaConf.from_dotlist(args)
    return OmegaConf.merge(base, cli)
