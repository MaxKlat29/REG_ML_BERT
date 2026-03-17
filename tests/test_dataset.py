"""Tests for LLMGeneratedDataset IterableDataset.

All LLM calls are mocked. A real tokenizer is used for BIO conversion.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.dataset import LLMGeneratedDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    from transformers import BertTokenizerFast
    tok = BertTokenizerFast.from_pretrained("deepset/gbert-large")
    assert tok.is_fast
    return tok


@pytest.fixture
def fake_config():
    """Minimal config object matching config/default.yaml structure."""
    data_cfg = SimpleNamespace(
        max_seq_length=512,
        samples_per_batch=2,
        negative_sample_ratio=0.4,
        cache_dir="data/cache",
        gold_test_dir="data/gold_test",
        llm_seed=1337,
        llm_model="qwen2.5:14b",
        ollama_endpoint="http://localhost:11434",
    )
    return SimpleNamespace(data=data_cfg)


# Tagged text that parse_ref_tags will turn into a span
TAGGED_TEXT = "Gemaess <ref>SS 25a KWG</ref> gilt das."


def _make_mock_llm_client(tagged_text: str = TAGGED_TEXT):
    """Return an AsyncMock that simulates call_ollama returning tagged_text."""
    mock = AsyncMock(return_value=tagged_text)
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_iterable_yields_samples(tokenizer, fake_config, tmp_path, monkeypatch):
    """Dataset yields dicts with keys input_ids, attention_mask, labels."""
    with (
        patch("src.data.dataset.call_ollama", new_callable=AsyncMock) as mock_call,
        patch("src.data.dataset.parse_ref_tags") as mock_parse,
        patch("src.data.dataset.build_generation_prompt", return_value="prompt"),
        patch("src.data.dataset.get_domain_for_seed", return_value="Bankenaufsicht"),
        patch("src.data.dataset.get_context_for_seed", return_value=("Regulatorischer Fachtext", "KWG")),
    ):
        mock_call.return_value = TAGGED_TEXT
        mock_parse.return_value = ("Gemaess SS 25a KWG gilt das.", [(7, 17)])

        dataset = LLMGeneratedDataset(fake_config, tokenizer, epoch=0)
        samples = list(dataset)

    assert len(samples) > 0
    sample = samples[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample


def test_labels_have_correct_values(tokenizer, fake_config, monkeypatch):
    """All label values are in {-100, 0, 1, 2}."""
    with (
        patch("src.data.dataset.call_ollama", new_callable=AsyncMock) as mock_call,
        patch("src.data.dataset.parse_ref_tags") as mock_parse,
        patch("src.data.dataset.build_generation_prompt", return_value="prompt"),
        patch("src.data.dataset.get_domain_for_seed", return_value="Bankenaufsicht"),
        patch("src.data.dataset.get_context_for_seed", return_value=("Regulatorischer Fachtext", "KWG")),
    ):
        mock_call.return_value = TAGGED_TEXT
        mock_parse.return_value = ("Gemaess SS 25a KWG gilt das.", [(7, 17)])

        dataset = LLMGeneratedDataset(fake_config, tokenizer, epoch=0)
        samples = list(dataset)

    valid_labels = {-100, 0, 1, 2}
    for sample in samples:
        for lbl in sample["labels"]:
            assert lbl in valid_labels, f"Unexpected label value: {lbl}"


def test_special_tokens_masked(tokenizer, fake_config, monkeypatch):
    """First real token ([CLS]) and last real token ([SEP]) have label -100."""
    with (
        patch("src.data.dataset.call_ollama", new_callable=AsyncMock) as mock_call,
        patch("src.data.dataset.parse_ref_tags") as mock_parse,
        patch("src.data.dataset.build_generation_prompt", return_value="prompt"),
        patch("src.data.dataset.get_domain_for_seed", return_value="Bankenaufsicht"),
        patch("src.data.dataset.get_context_for_seed", return_value=("Regulatorischer Fachtext", "KWG")),
    ):
        mock_call.return_value = TAGGED_TEXT
        mock_parse.return_value = ("Gemaess SS 25a KWG gilt das.", [(7, 17)])

        dataset = LLMGeneratedDataset(fake_config, tokenizer, epoch=0)
        samples = list(dataset)

    for sample in samples:
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        for i, tok_id in enumerate(input_ids):
            if tok_id == cls_id:
                assert labels[i] == -100, f"CLS at {i} should be -100, got {labels[i]}"
            if tok_id == sep_id:
                assert labels[i] == -100, f"SEP at {i} should be -100, got {labels[i]}"


def test_worker_sharding_different_seeds(tokenizer, fake_config, monkeypatch):
    """Two simulated workers produce different seeds (via worker_id offset)."""
    seeds_seen = []

    def capture_domain(seed):
        seeds_seen.append(seed)
        return "Bankenaufsicht"

    with (
        patch("src.data.dataset.call_ollama", new_callable=AsyncMock) as mock_call,
        patch("src.data.dataset.parse_ref_tags") as mock_parse,
        patch("src.data.dataset.build_generation_prompt", return_value="prompt"),
        patch("src.data.dataset.get_domain_for_seed", side_effect=capture_domain),
        patch("src.data.dataset.get_context_for_seed", return_value=("Regulatorischer Fachtext", "KWG")),
    ):
        mock_call.return_value = TAGGED_TEXT
        mock_parse.return_value = ("Gemaess SS 25a KWG gilt das.", [(7, 17)])

        worker0_info = SimpleNamespace(id=0, num_workers=2)
        worker1_info = SimpleNamespace(id=1, num_workers=2)

        seeds_seen.clear()
        with patch("src.data.dataset.get_worker_info", return_value=worker0_info):
            dataset0 = LLMGeneratedDataset(fake_config, tokenizer, epoch=0)
            list(dataset0)
        seeds_w0 = list(seeds_seen)

        seeds_seen.clear()
        with patch("src.data.dataset.get_worker_info", return_value=worker1_info):
            dataset1 = LLMGeneratedDataset(fake_config, tokenizer, epoch=0)
            list(dataset1)
        seeds_w1 = list(seeds_seen)

    assert seeds_w0 != seeds_w1, f"Worker 0 seeds {seeds_w0} should differ from worker 1 seeds {seeds_w1}"


def test_cache_mode_reads_from_disk(tokenizer, fake_config, tmp_path):
    """Dataset in cache_mode reads from JSONL instead of calling LLM."""
    from src.data.cache import append_to_cache
    from src.data.bio_converter import char_spans_to_bio

    cache_file = tmp_path / "test_dataset_cache.jsonl"

    text = "Gemaess SS 25a KWG gilt das."
    enc = char_spans_to_bio(text, [(7, 17)], tokenizer)
    append_to_cache(enc, cache_file)

    with patch("src.data.dataset.call_ollama") as mock_call:
        dataset = LLMGeneratedDataset(fake_config, tokenizer, epoch=0, cache_path=cache_file)
        samples = list(dataset)
        mock_call.assert_not_called()

    assert len(samples) == 1
    assert "input_ids" in samples[0]
    assert "labels" in samples[0]
