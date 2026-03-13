"""Device detection and seed setup utilities."""

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU.

    Returns:
        torch.device ready for .to(device) calls.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Detected device: %s", device)
    return device


def set_seed(seed: int) -> None:
    """Set seeds for full reproducibility across PyTorch, NumPy, and Python random.

    Args:
        seed: Integer seed from config (e.g., cfg.project.seed).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
