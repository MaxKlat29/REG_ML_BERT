"""Tests for the gold test set generator.

All LLM calls are mocked — no live API required.

Coverage:
    test_gold_generation_produces_json        - Writes valid JSON file to gold_test_dir
    test_all_samples_have_needs_review        - Every entry has needs_review: true
    test_positive_negative_mix               - Correct positive/negative split
    test_gold_samples_have_required_fields   - Each sample has all required keys
    test_fixed_seed_reproducibility          - Same seed -> same structural output
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from omegaconf import OmegaConf

from scripts.generate_gold_test import generate_gold_set


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, negative_ratio: float = 0.4, seed: int = 1337) -> object:
    """Build a minimal OmegaConf config pointing gold_test_dir to tmp_path."""
    return OmegaConf.create({
        "data": {
            "negative_sample_ratio": negative_ratio,
            "gold_test_dir": str(tmp_path / "gold_test"),
            "llm_seed": seed,
            "llm_model": "google/gemini-flash-1.5",
        },
        "model": {
            "name": "deepset/gbert-large",
        },
    })


# Tagged text for positive samples (has <ref> tag)
_POSITIVE_TAGGED = "Gemäß <ref>§ 25a KWG</ref> gilt eine besondere Pflicht."
# Plain text for negative samples (no references)
_NEGATIVE_TAGGED = "Dies ist ein erläuternder Satz ohne Rechtsverweise."


def _make_side_effect():
    """Return an async side_effect that returns positive or negative text based on prompt."""
    async def _side_effect(client, model, messages, seed, **kw) -> str:
        prompt_content = messages[-1]["content"]
        if "KEINE" in prompt_content or "keine" in prompt_content:
            return _NEGATIVE_TAGGED
        return _POSITIVE_TAGGED
    return _side_effect


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGoldGenerationProducesJson:
    """Script writes a valid JSON file to configured gold_test_dir."""

    def test_gold_generation_produces_json(self, tmp_path):
        config = _make_config(tmp_path)
        output_path = tmp_path / "gold_test" / "gold_test_set.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with patch(
            "scripts.generate_gold_test.call_openrouter",
            new_callable=AsyncMock,
            side_effect=_make_side_effect(),
        ):
            samples = generate_gold_set(config, num_samples=5, output_path=output_path)

        assert output_path.exists(), "JSON file was not created"
        with open(output_path, encoding="utf-8") as fh:
            loaded = json.load(fh)
        assert isinstance(loaded, list)
        assert len(loaded) == len(samples)


class TestAllSamplesHaveNeedsReview:
    """Every entry in the gold JSON has needs_review: true."""

    def test_all_samples_have_needs_review(self, tmp_path):
        config = _make_config(tmp_path)
        output_path = tmp_path / "gold_test_set.json"

        with patch(
            "scripts.generate_gold_test.call_openrouter",
            new_callable=AsyncMock,
            side_effect=_make_side_effect(),
        ):
            samples = generate_gold_set(config, num_samples=10, output_path=output_path)

        assert len(samples) == 10
        for i, sample in enumerate(samples):
            assert "needs_review" in sample, f"Sample {i} missing 'needs_review'"
            assert sample["needs_review"] is True, f"Sample {i} has needs_review != True"


class TestPositiveNegativeMix:
    """Gold set has positive and negative examples with correct ratio."""

    def test_positive_negative_mix(self, tmp_path):
        config = _make_config(tmp_path, negative_ratio=0.4)
        output_path = tmp_path / "gold_test_set.json"

        with patch(
            "scripts.generate_gold_test.call_openrouter",
            new_callable=AsyncMock,
            side_effect=_make_side_effect(),
        ):
            samples = generate_gold_set(config, num_samples=10, output_path=output_path)

        positives = [s for s in samples if s["has_references"] is True]
        negatives = [s for s in samples if s["has_references"] is False]

        assert len(negatives) > 0, "No negative samples found"
        assert len(positives) > 0, "No positive samples found"

        # With ratio=0.4 and 10 samples: first 4 negative, next 6 positive
        expected_negatives = int(10 * 0.4)
        assert len(negatives) == expected_negatives, (
            f"Expected {expected_negatives} negatives, got {len(negatives)}"
        )


class TestGoldSamplesHaveRequiredFields:
    """Each sample has: text, spans, bio_labels, needs_review, domain, seed, has_references."""

    REQUIRED_KEYS = {"text", "spans", "bio_labels", "needs_review", "domain", "seed", "has_references"}

    def test_gold_samples_have_required_fields(self, tmp_path):
        config = _make_config(tmp_path)
        output_path = tmp_path / "gold_test_set.json"

        with patch(
            "scripts.generate_gold_test.call_openrouter",
            new_callable=AsyncMock,
            side_effect=_make_side_effect(),
        ):
            samples = generate_gold_set(config, num_samples=5, output_path=output_path)

        for i, sample in enumerate(samples):
            missing = self.REQUIRED_KEYS - set(sample.keys())
            assert not missing, f"Sample {i} missing fields: {missing}"

            # bio_labels is a dict with input_ids, attention_mask, labels
            bio = sample["bio_labels"]
            assert isinstance(bio, dict), f"Sample {i}: bio_labels should be dict"
            assert "labels" in bio, f"Sample {i}: bio_labels missing 'labels'"
            assert "input_ids" in bio, f"Sample {i}: bio_labels missing 'input_ids'"

            # spans is a list of [start, end] pairs (JSON serializes tuples as lists)
            assert isinstance(sample["spans"], list), f"Sample {i}: spans should be list"

            # domain is non-empty string
            assert isinstance(sample["domain"], str) and sample["domain"], (
                f"Sample {i}: domain is empty"
            )


class TestFixedSeedReproducibility:
    """Running generation twice with same seed produces same structural output."""

    def test_fixed_seed_reproducibility(self, tmp_path):
        config = _make_config(tmp_path, seed=42)

        with patch(
            "scripts.generate_gold_test.call_openrouter",
            new_callable=AsyncMock,
            side_effect=_make_side_effect(),
        ):
            path_a = tmp_path / "run_a.json"
            samples_a = generate_gold_set(config, num_samples=8, output_path=path_a)

        with patch(
            "scripts.generate_gold_test.call_openrouter",
            new_callable=AsyncMock,
            side_effect=_make_side_effect(),
        ):
            path_b = tmp_path / "run_b.json"
            samples_b = generate_gold_set(config, num_samples=8, output_path=path_b)

        # Same number of samples
        assert len(samples_a) == len(samples_b)

        # Same positive/negative structure
        refs_a = [s["has_references"] for s in samples_a]
        refs_b = [s["has_references"] for s in samples_b]
        assert refs_a == refs_b, "has_references sequence differs between runs"

        # Same domains selected
        domains_a = [s["domain"] for s in samples_a]
        domains_b = [s["domain"] for s in samples_b]
        assert domains_a == domains_b, "Domain sequence differs between runs"

        # Same seeds per sample
        seeds_a = [s["seed"] for s in samples_a]
        seeds_b = [s["seed"] for s in samples_b]
        assert seeds_a == seeds_b, "Seed sequence differs between runs"
