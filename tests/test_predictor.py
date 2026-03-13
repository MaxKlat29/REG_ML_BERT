"""Tests for Predictor class (INFR-01, INFR-02, INFR-03).

Covers:
  - predict() returns PredictedSpan list with correct char offsets and text
  - Confidence scores in [0, 1] for non-CRF; 1.0 for CRF
  - predict_batch() returns one list per input
  - Empty predictions for text with no references
  - find_latest_checkpoint() discovers most recent .pt file
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Helper to build a minimal config
# ---------------------------------------------------------------------------

def _make_config(use_crf: bool = False) -> SimpleNamespace:
    data = SimpleNamespace(max_seq_length=128)
    model = SimpleNamespace(
        name="bert-base-uncased",
        use_crf=use_crf,
        freeze_backbone=False,
        use_lora=False,
        lora_rank=16,
    )
    return SimpleNamespace(data=data, model=model)


# ---------------------------------------------------------------------------
# Helper: build mock tokenizer with known offset_mapping
# ---------------------------------------------------------------------------

def _make_mock_tokenizer(token_texts=None, offset_mapping=None):
    """Return a mock tokenizer whose __call__ returns a known encoding."""
    if offset_mapping is None:
        # [CLS](0,0), "Gemaess"(0,7), "§"(8,9), "25a"(10,13), [SEP](0,0), PAD...
        offset_mapping = [(0, 0), (0, 7), (8, 9), (10, 13), (0, 0)] + [(0, 0)] * 123

    seq_len = len(offset_mapping)
    input_ids = [101] + [102] * (seq_len - 2) + [103]
    attention_mask = [1] * 5 + [0] * (seq_len - 5)

    encoding = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "offset_mapping": offset_mapping,
    }

    # Build a MagicMock that supports dict-style access and returns the encoding values
    mock_enc = MagicMock()
    mock_enc.__getitem__ = lambda self_enc, k: encoding[k]
    mock_enc.get = lambda k, default=None: encoding.get(k, default)

    mock_tok = MagicMock()
    mock_tok.return_value = mock_enc

    return mock_tok, encoding


# ---------------------------------------------------------------------------
# Helper: build logits tensor — B-REF at token 2 ("§"), I-REF at token 3 ("25a")
# ---------------------------------------------------------------------------

def _make_logits(seq_len: int = 128, b_ref_idx: int = 2, i_ref_idx: int = 3) -> torch.Tensor:
    """Return logits (1, seq_len, 3) with high B-REF at b_ref_idx, I-REF at i_ref_idx."""
    # O=0, B-REF=1, I-REF=2
    logits = torch.zeros(1, seq_len, 3)
    # All O by default; make specific positions B-REF / I-REF
    logits[0, b_ref_idx, 0] = -10.0
    logits[0, b_ref_idx, 1] = 10.0   # high B-REF
    logits[0, b_ref_idx, 2] = -10.0
    logits[0, i_ref_idx, 0] = -10.0
    logits[0, i_ref_idx, 1] = -10.0
    logits[0, i_ref_idx, 2] = 10.0   # high I-REF
    return logits


# ---------------------------------------------------------------------------
# Fixture: patch heavy imports so no real model/tokenizer is loaded
# ---------------------------------------------------------------------------

def _build_predictor_with_mocks(use_crf: bool = False, logits=None):
    """Construct a Predictor with all heavy deps mocked out."""
    from src.model.predictor import Predictor

    config = _make_config(use_crf=use_crf)
    device = torch.device("cpu")
    seq_len = 128

    if logits is None:
        logits = _make_logits(seq_len)

    # Build mock model
    mock_model = MagicMock()
    mock_model._use_crf = use_crf

    if use_crf:
        # CRF returns list[list[int]]: position 2=B-REF=1, position 3=I-REF=2
        crf_labels = [0] * seq_len
        crf_labels[2] = 1
        crf_labels[3] = 2
        mock_model.return_value = [crf_labels]
    else:
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_model.return_value = mock_output

    # Build mock tokenizer
    mock_tok, encoding = _make_mock_tokenizer()

    # offset_mapping for the default: token 2 -> (8,9), token 3 -> (10,13)
    # text = "Gemaess § 25a"
    # span should be text[8:13] = "§ 25a"? No: decode_bio_to_char_spans expands per-token
    # Actually token 2 offset (8,9) and token 3 offset (10,13)
    # The span will be start=8 (from B-REF tok) end=13 (from last I-REF tok)

    with (
        patch("src.model.predictor.RegulatoryNERModel", return_value=mock_model),
        patch("src.model.predictor.load_checkpoint", return_value=0),
        patch("src.model.predictor.BertTokenizerFast.from_pretrained", return_value=mock_tok),
    ):
        predictor = Predictor(
            checkpoint_path=Path("/fake/ckpt.pt"),
            config=config,
            device=device,
        )

    # Replace the internal model and tokenizer with our mocks directly
    predictor._model = mock_model
    predictor._tokenizer = mock_tok
    predictor._device = device

    return predictor, mock_model, mock_tok, encoding


# ---------------------------------------------------------------------------
# TestPredict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_returns_list_of_predicted_spans(self):
        """predict() returns a list of PredictedSpan objects."""
        from src.model.predictor import Predictor, PredictedSpan

        predictor, mock_model, mock_tok, encoding = _build_predictor_with_mocks(use_crf=False)
        text = "Gemaess § 25a KWG gilt"

        spans = predictor.predict(text)

        assert isinstance(spans, list)
        assert len(spans) >= 1
        assert all(isinstance(s, PredictedSpan) for s in spans)

    def test_span_char_offsets_match_text(self):
        """PredictedSpan.text == original_text[start:end]."""
        predictor, _, _, _ = _build_predictor_with_mocks(use_crf=False)
        text = "Gemaess § 25a KWG gilt"

        spans = predictor.predict(text)

        for span in spans:
            assert span.text == text[span.start:span.end], (
                f"span.text={span.text!r} != text[{span.start}:{span.end}]={text[span.start:span.end]!r}"
            )

    def test_span_start_end_are_int(self):
        """start and end are integers."""
        predictor, _, _, _ = _build_predictor_with_mocks(use_crf=False)
        text = "Gemaess § 25a KWG gilt"

        spans = predictor.predict(text)

        for span in spans:
            assert isinstance(span.start, int)
            assert isinstance(span.end, int)
            assert span.end > span.start

    def test_predict_uses_no_grad(self):
        """Model is called inside torch.no_grad()."""
        predictor, mock_model, _, _ = _build_predictor_with_mocks(use_crf=False)
        text = "Gemaess § 25a"

        predictor.predict(text)

        mock_model.assert_called_once()


# ---------------------------------------------------------------------------
# TestConfidenceScores
# ---------------------------------------------------------------------------

class TestConfidenceScores:
    def test_confidence_in_0_1_non_crf(self):
        """Confidence scores are in [0, 1] for non-CRF models."""
        predictor, _, _, _ = _build_predictor_with_mocks(use_crf=False)
        text = "Gemaess § 25a KWG gilt"

        spans = predictor.predict(text)

        for span in spans:
            assert 0.0 <= span.confidence <= 1.0, (
                f"confidence={span.confidence} out of [0,1]"
            )

    def test_confidence_is_1_for_crf(self):
        """CRF model returns confidence=1.0 for all spans."""
        predictor, _, _, _ = _build_predictor_with_mocks(use_crf=True)
        text = "Gemaess § 25a KWG gilt"

        spans = predictor.predict(text)

        for span in spans:
            assert span.confidence == 1.0, (
                f"CRF span has confidence={span.confidence} (expected 1.0)"
            )


# ---------------------------------------------------------------------------
# TestBatchPredict
# ---------------------------------------------------------------------------

class TestBatchPredict:
    def test_batch_returns_list_of_lists(self):
        """predict_batch returns list[list[PredictedSpan]]."""
        from src.model.predictor import PredictedSpan

        predictor, _, _, _ = _build_predictor_with_mocks(use_crf=False)
        texts = ["Text with § 25a KWG", "Another § 8 HGB reference"]

        results = predictor.predict_batch(texts)

        assert isinstance(results, list)
        assert len(results) == len(texts)
        for item in results:
            assert isinstance(item, list)

    def test_batch_length_matches_input(self):
        """predict_batch returns one result per input text."""
        predictor, _, _, _ = _build_predictor_with_mocks(use_crf=False)
        texts = ["a", "b", "c", "d"]

        results = predictor.predict_batch(texts)

        assert len(results) == 4

    def test_batch_single_text(self):
        """predict_batch with one text works."""
        predictor, _, _, _ = _build_predictor_with_mocks(use_crf=False)

        results = predictor.predict_batch(["§ 25a KWG"])

        assert len(results) == 1


# ---------------------------------------------------------------------------
# TestNoReferences
# ---------------------------------------------------------------------------

class TestNoReferences:
    def test_all_O_returns_empty_list(self):
        """When model predicts all O, predict() returns empty list."""
        from src.model.predictor import Predictor

        config = _make_config(use_crf=False)
        device = torch.device("cpu")
        seq_len = 128

        # All-O logits: O class is highest at every position
        logits = torch.zeros(1, seq_len, 3)
        logits[0, :, 0] = 10.0  # O class wins everywhere

        mock_model = MagicMock()
        mock_model._use_crf = False
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_model.return_value = mock_output

        mock_tok, _ = _make_mock_tokenizer()

        with (
            patch("src.model.predictor.RegulatoryNERModel", return_value=mock_model),
            patch("src.model.predictor.load_checkpoint", return_value=0),
            patch("src.model.predictor.BertTokenizerFast.from_pretrained", return_value=mock_tok),
        ):
            predictor = Predictor(
                checkpoint_path=Path("/fake/ckpt.pt"),
                config=config,
                device=device,
            )

        predictor._model = mock_model
        predictor._tokenizer = mock_tok
        predictor._device = device

        spans = predictor.predict("Kein Verweis im Text")

        assert spans == [], f"Expected empty list, got {spans}"


# ---------------------------------------------------------------------------
# TestFindLatestCheckpoint
# ---------------------------------------------------------------------------

class TestFindLatestCheckpoint:
    def test_finds_most_recent_pt_file(self):
        """find_latest_checkpoint returns the most recently modified .pt file."""
        from src.model.predictor import Predictor
        from src.model.trainer import CHECKPOINT_BASE

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Create two .pt files, make the second "newer"
            old_ckpt = tmp_path / "epoch_0.pt"
            new_ckpt = tmp_path / "epoch_1.pt"
            old_ckpt.write_bytes(b"old")
            time.sleep(0.01)
            new_ckpt.write_bytes(b"new")

            with patch("src.model.predictor.CHECKPOINT_BASE", tmp_path):
                result = Predictor.find_latest_checkpoint(base_dir=tmp_path)

            assert result == new_ckpt

    def test_raises_if_no_checkpoints(self):
        """find_latest_checkpoint raises FileNotFoundError when no .pt files exist."""
        from src.model.predictor import Predictor

        with tempfile.TemporaryDirectory() as tmpdir:
            empty_path = Path(tmpdir)
            with pytest.raises(FileNotFoundError):
                Predictor.find_latest_checkpoint(base_dir=empty_path)

    def test_finds_pt_in_subdirectory(self):
        """find_latest_checkpoint searches recursively for .pt files."""
        from src.model.predictor import Predictor

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            subdir = tmp_path / "run_01"
            subdir.mkdir()
            ckpt = subdir / "epoch_2.pt"
            ckpt.write_bytes(b"ckpt")

            result = Predictor.find_latest_checkpoint(base_dir=tmp_path)

            assert result == ckpt
