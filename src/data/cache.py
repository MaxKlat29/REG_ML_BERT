"""JSONL disk cache for training samples.

Supports append-only writes and full-file reads.
Used for ensemble resampling and offline data inspection.
"""
from __future__ import annotations

import json
from pathlib import Path


def append_to_cache(sample: dict, cache_path: Path) -> None:
    """Append a single sample as a JSON line to the cache file.

    Creates the file (and parent directories) if it does not exist.
    Uses ensure_ascii=False to preserve Unicode characters (German umlauts, etc.).

    Args:
        sample: Dictionary to serialize.
        cache_path: Path to the JSONL cache file.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def load_cache(cache_path: Path) -> list[dict]:
    """Load all samples from a JSONL cache file.

    Args:
        cache_path: Path to the JSONL cache file.

    Returns:
        List of sample dicts. Returns empty list if file does not exist.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return []

    samples: list[dict] = []
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples
