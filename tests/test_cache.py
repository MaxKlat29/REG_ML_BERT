"""Tests for JSONL disk cache (append + load)."""
import json
from pathlib import Path

import pytest

from src.data.cache import append_to_cache, load_cache


def test_append_and_load_roundtrip(tmp_path):
    """Write 3 samples, load returns same 3 samples."""
    cache_file = tmp_path / "test_cache.jsonl"
    samples = [
        {"text": "Sample A", "labels": [0, 1, 2]},
        {"text": "Sample B", "labels": [0, 0, 0]},
        {"text": "Sample C", "labels": [1, 2, 2]},
    ]
    for s in samples:
        append_to_cache(s, cache_file)

    loaded = load_cache(cache_file)
    assert loaded == samples


def test_append_preserves_existing(tmp_path):
    """Write 2 samples, append 1 more, load returns all 3."""
    cache_file = tmp_path / "test_preserve.jsonl"
    first_two = [
        {"id": 1, "value": "alpha"},
        {"id": 2, "value": "beta"},
    ]
    for s in first_two:
        append_to_cache(s, cache_file)

    third = {"id": 3, "value": "gamma"}
    append_to_cache(third, cache_file)

    loaded = load_cache(cache_file)
    assert len(loaded) == 3
    assert loaded == first_two + [third]


def test_load_empty_file(tmp_path):
    """Load from non-existent file returns empty list."""
    non_existent = tmp_path / "does_not_exist.jsonl"
    result = load_cache(non_existent)
    assert result == []


def test_unicode_preservation(tmp_path):
    """German umlauts and special chars survive cache roundtrip."""
    cache_file = tmp_path / "unicode.jsonl"
    sample = {
        "text": "Gemäß § 25a KWG (Kreditwesengesetz) gilt: ü, ö, ä, ß",
        "domain": "Bankenaufsicht",
    }
    append_to_cache(sample, cache_file)

    loaded = load_cache(cache_file)
    assert len(loaded) == 1
    assert loaded[0]["text"] == sample["text"]
    assert "ä" in loaded[0]["text"]
    assert "ö" in loaded[0]["text"]
    assert "ü" in loaded[0]["text"]
    assert "ß" in loaded[0]["text"]
