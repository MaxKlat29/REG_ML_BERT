"""Configuration loading with YAML defaults and CLI overrides."""

import sys
from omegaconf import OmegaConf, DictConfig


def load_config(
    config_path: str = "config/default.yaml",
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load YAML config and apply CLI overrides.

    Args:
        config_path: Path to default.yaml
        overrides: List of "key=value" strings. If None, parses sys.argv[1:].

    Returns:
        Merged DictConfig with dot-notation access.
    """
    base = OmegaConf.load(config_path)
    if overrides is not None:
        cli = OmegaConf.from_dotlist(overrides)
    else:
        # Parse sys.argv[1:], strip leading "--" for user convenience
        args = [a.lstrip("-") if "=" in a else a for a in sys.argv[1:]]
        cli = OmegaConf.from_dotlist(args)
    return OmegaConf.merge(base, cli)
