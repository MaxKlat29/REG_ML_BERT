"""Tests for the extended evaluation subsystem.

Covers:
    - TestIoUScoring: span_iou() character-offset IoU
    - TestSpanTypeClassification: classify_span_type() reference classification
    - TestBIODecode: decode_bio_to_char_spans() BIO-to-char reconstruction
    - TestPartialMatchMetrics: compute_partial_match_metrics() IoU-based P/R/F1
    - TestTypedExtraction: RegexBaseline.extract_typed() per-type triples
    - TestMLEvaluation: Evaluator.evaluate_model() with mock model
    - TestFPFNDump: Evaluator.dump_errors() JSON output
    - TestComparisonReport: Evaluator.format_comparison_report() formatting
    - TestPerTypeBreakdown: per-reference-type metrics in evaluate_model()
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import torch

from src.evaluation.metrics import (
    classify_span_type,
    compute_partial_match_metrics,
    decode_bio_to_char_spans,
    span_iou,
)
from src.evaluation.regex_baseline import RegexBaseline


# ---------------------------------------------------------------------------
# TestIoUScoring
# ---------------------------------------------------------------------------


class TestIoUScoring:
    """Tests for span_iou() character-offset intersection over union."""

    def test_overlapping_half(self):
        """span_iou((0,10), (5,15)) == 0.5 — intersection=5, union=15."""
        assert span_iou((0, 10), (5, 15)) == pytest.approx(5 / 15, abs=1e-6)

    def test_exact_match(self):
        """span_iou((0,10), (0,10)) == 1.0."""
        assert span_iou((0, 10), (0, 10)) == pytest.approx(1.0)

    def test_disjoint(self):
        """span_iou((0,5), (10,15)) == 0.0 — no overlap."""
        assert span_iou((0, 5), (10, 15)) == pytest.approx(0.0)

    def test_degenerate_zero_length(self):
        """span_iou((0,0), (0,0)) == 1.0 — degenerate equal zero-length spans."""
        assert span_iou((0, 0), (0, 0)) == pytest.approx(1.0)

    def test_contained(self):
        """span_iou((2,8), (0,10)) > 0.5 — pred fully contained in gold."""
        result = span_iou((2, 8), (0, 10))
        assert result == pytest.approx(6 / 10, abs=1e-6)

    def test_adjacent_no_overlap(self):
        """Adjacent spans (0,5) and (5,10) have zero overlap."""
        assert span_iou((0, 5), (5, 10)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestSpanTypeClassification
# ---------------------------------------------------------------------------


class TestSpanTypeClassification:
    """Tests for classify_span_type() reference type detection."""

    def test_paragraph(self):
        """§ 25a KWG is classified as PARAGRAPH."""
        assert classify_span_type("§ 25a KWG") == "PARAGRAPH"

    def test_paragraphen_plural(self):
        """§§ 25a-25b KWG is classified as PARAGRAPH."""
        assert classify_span_type("§§ 25a-25b KWG") == "PARAGRAPH"

    def test_artikel(self):
        """Art. 5 DSGVO is classified as ARTIKEL."""
        assert classify_span_type("Art. 5 DSGVO") == "ARTIKEL"

    def test_artikel_full(self):
        """Artikel 5 DSGVO is classified as ARTIKEL."""
        assert classify_span_type("Artikel 5 DSGVO") == "ARTIKEL"

    def test_anhang(self):
        """Anhang IV CRR is classified as ANHANG."""
        assert classify_span_type("Anhang IV CRR") == "ANHANG"

    def test_absatz(self):
        """Abs. 3 is classified as ABSATZ."""
        assert classify_span_type("Abs. 3") == "ABSATZ"

    def test_nummer(self):
        """Nr. 5 is classified as NUMMER."""
        assert classify_span_type("Nr. 5") == "NUMMER"

    def test_satz(self):
        """Satz 2 is classified as SATZ."""
        assert classify_span_type("Satz 2") == "SATZ"

    def test_teilziffer(self):
        """Tz. 12 is classified as TEILZIFFER."""
        assert classify_span_type("Tz. 12") == "TEILZIFFER"

    def test_literal(self):
        """lit. a is classified as LITERAL."""
        assert classify_span_type("lit. a") == "LITERAL"

    def test_verordnung(self):
        """EU-Verordnung 575/2013 is classified as VERORDNUNG."""
        assert classify_span_type("EU-Verordnung 575/2013") == "VERORDNUNG"

    def test_fallback_ref(self):
        """Unknown span text falls back to REF."""
        assert classify_span_type("unknown text xyz") == "REF"


# ---------------------------------------------------------------------------
# TestBIODecode
# ---------------------------------------------------------------------------


class TestBIODecode:
    """Tests for decode_bio_to_char_spans()."""

    def test_basic_single_span(self):
        """B I I O reconstructs one span."""
        # token_labels: B=1, I=2, I=2, O=0
        # offsets: (0,1), (1,4), (4,8), (9,13)
        result = decode_bio_to_char_spans(
            token_labels=[1, 2, 2, 0],
            offset_mapping=[(0, 1), (1, 4), (4, 8), (9, 13)],
        )
        assert result == [(0, 8)]

    def test_skips_special_tokens(self):
        """(0,0) offset tokens (special) are skipped."""
        # CLS at (0,0), real token B at (0,5), SEP at (0,0)
        result = decode_bio_to_char_spans(
            token_labels=[-100, 1, 0, -100],
            offset_mapping=[(0, 0), (0, 5), (6, 10), (0, 0)],
        )
        assert result == [(0, 5)]

    def test_two_spans(self):
        """Two separate B-I spans produce two tuples."""
        result = decode_bio_to_char_spans(
            token_labels=[1, 2, 0, 1, 2, 2],
            offset_mapping=[(0, 3), (3, 6), (7, 8), (9, 12), (12, 15), (15, 18)],
        )
        assert result == [(0, 6), (9, 18)]

    def test_empty_labels(self):
        """All-O labels produce empty spans list."""
        result = decode_bio_to_char_spans(
            token_labels=[0, 0, 0],
            offset_mapping=[(0, 3), (4, 7), (8, 11)],
        )
        assert result == []

    def test_single_b_token(self):
        """Single B token (no I) produces one-token span."""
        result = decode_bio_to_char_spans(
            token_labels=[1],
            offset_mapping=[(5, 10)],
        )
        assert result == [(5, 10)]


# ---------------------------------------------------------------------------
# TestPartialMatchMetrics
# ---------------------------------------------------------------------------


class TestPartialMatchMetrics:
    """Tests for compute_partial_match_metrics() IoU-based P/R/F1."""

    def test_perfect_match(self):
        """Identical gold and pred spans → P=R=F1=1.0."""
        gold = [(0, 10), (20, 30)]
        pred = [(0, 10), (20, 30)]
        metrics = compute_partial_match_metrics(gold, pred)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_no_match(self):
        """Completely disjoint spans → P=R=F1=0.0."""
        gold = [(0, 5)]
        pred = [(10, 15)]
        metrics = compute_partial_match_metrics(gold, pred)
        assert metrics["precision"] == pytest.approx(0.0)
        assert metrics["recall"] == pytest.approx(0.0)
        assert metrics["f1"] == pytest.approx(0.0)

    def test_partial_overlap_above_threshold(self):
        """Overlap IoU > 0.5 counts as TP."""
        gold = [(0, 10)]
        pred = [(2, 10)]  # IoU = 8/10 = 0.8 > 0.5
        metrics = compute_partial_match_metrics(gold, pred, iou_threshold=0.5)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)

    def test_partial_overlap_below_threshold(self):
        """Overlap IoU < 0.5 does not count as TP."""
        gold = [(0, 10)]
        pred = [(8, 18)]  # intersection=2, union=18, IoU=1/9 < 0.5
        metrics = compute_partial_match_metrics(gold, pred, iou_threshold=0.5)
        assert metrics["precision"] == pytest.approx(0.0)
        assert metrics["recall"] == pytest.approx(0.0)

    def test_empty_pred(self):
        """No predictions → precision=1.0 (undefined, returns 1.0 by convention), recall=0.0."""
        gold = [(0, 10)]
        pred = []
        metrics = compute_partial_match_metrics(gold, pred)
        assert metrics["recall"] == pytest.approx(0.0)

    def test_empty_gold(self):
        """No gold spans → recall=1.0 (undefined), precision=0.0."""
        gold = []
        pred = [(0, 10)]
        metrics = compute_partial_match_metrics(gold, pred)
        assert metrics["precision"] == pytest.approx(0.0)

    def test_keys_present(self):
        """Result dict has precision, recall, f1 keys."""
        metrics = compute_partial_match_metrics([(0, 5)], [(0, 5)])
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics


# ---------------------------------------------------------------------------
# TestTypedExtraction
# ---------------------------------------------------------------------------


class TestTypedExtraction:
    """Tests for RegexBaseline.extract_typed()."""

    def test_paragraph_and_artikel(self):
        """Mixed text returns PARAGRAPH and ARTIKEL typed triples."""
        baseline = RegexBaseline()
        text = "Gemaess § 25a KWG und Art. 5 DSGVO gilt"
        results = baseline.extract_typed(text)
        types = [r[2] for r in results]
        assert "PARAGRAPH" in types
        assert "ARTIKEL" in types

    def test_returns_sorted_by_start(self):
        """Results are sorted by start position."""
        baseline = RegexBaseline()
        text = "Art. 5 DSGVO und § 25a KWG"
        results = baseline.extract_typed(text)
        starts = [r[0] for r in results]
        assert starts == sorted(starts)

    def test_triple_structure(self):
        """Each element is a (int, int, str) triple."""
        baseline = RegexBaseline()
        text = "§ 25a KWG"
        results = baseline.extract_typed(text)
        assert len(results) >= 1
        for triple in results:
            assert len(triple) == 3
            start, end, typ = triple
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert isinstance(typ, str)

    def test_no_overlap_dedup(self):
        """extract_typed deduplicates overlapping spans (keeps longest)."""
        baseline = RegexBaseline()
        # "§ 25a KWG" should produce only one span, not two overlapping ones
        text = "§ 25a KWG"
        results = baseline.extract_typed(text)
        # Check no two spans overlap
        for i, (s1, e1, _) in enumerate(results):
            for j, (s2, e2, _) in enumerate(results):
                if i != j:
                    intersection = max(0, min(e1, e2) - max(s1, s2))
                    assert intersection == 0, f"Overlapping spans: ({s1},{e1}) and ({s2},{e2})"

    def test_typed_patterns_class_attribute(self):
        """RegexBaseline.TYPED_PATTERNS is a dict of compiled patterns."""
        import regex as re_pkg
        assert hasattr(RegexBaseline, "TYPED_PATTERNS")
        assert isinstance(RegexBaseline.TYPED_PATTERNS, dict)
        assert len(RegexBaseline.TYPED_PATTERNS) > 0
        for key, pat in RegexBaseline.TYPED_PATTERNS.items():
            assert isinstance(key, str)
            # regex pattern objects have .pattern attribute
            assert hasattr(pat, "finditer")


# ---------------------------------------------------------------------------
# TestMLEvaluation
# ---------------------------------------------------------------------------


class TestMLEvaluation:
    """Tests for Evaluator.evaluate_model() with mock model/tokenizer."""

    def _make_gold_sample(self, text="§ 25a KWG gilt", span_start=0, span_end=9):
        """Create a minimal gold sample dict."""
        seq_len = 10
        labels = [-100] + [1, 2, 2, 2, 0, 0, 0, 0] + [-100]  # CLS, tokens, SEP
        labels = labels[:seq_len]
        while len(labels) < seq_len:
            labels.append(-100)
        return {
            "text": text,
            "spans": [(span_start, span_end)],
            "bio_labels": {
                "input_ids": [101] + [1000] * (seq_len - 2) + [102],
                "attention_mask": [1] * seq_len,
                "labels": labels,
            },
            "domain": "KWG",
        }

    def _make_mock_tokenizer(self):
        """Mock tokenizer returning fixed encoding."""
        seq_len = 10
        mock_enc = {
            "input_ids": torch.tensor([[101] + [1000] * 8 + [102]]),
            "attention_mask": torch.tensor([[1] * seq_len]),
            "offset_mapping": [(0, 0), (0, 1), (1, 4), (4, 5), (5, 6), (7, 8), (8, 9), (9, 10), (10, 11), (0, 0)],
        }

        class MockEncoding:
            def __getitem__(self, key):
                return mock_enc[key]

            def to(self, device):
                return self

        mock_tok = MagicMock()
        mock_tok.return_value = MockEncoding()
        return mock_tok

    def _make_mock_model_non_crf(self, seq_len=10, num_labels=3):
        """Mock non-CRF model returning logits."""
        import torch.nn as nn

        mock_model = MagicMock()
        mock_model._use_crf = False
        # Logits: shape (1, seq_len, num_labels) — all predict O except positions 1,2,3 → B,I,I
        logits = torch.zeros(1, seq_len, num_labels)
        logits[0, 1, 1] = 10.0  # B-REF
        logits[0, 2, 2] = 10.0  # I-REF
        logits[0, 3, 2] = 10.0  # I-REF

        output = MagicMock()
        output.logits = logits
        mock_model.return_value = output
        mock_model.eval = MagicMock(return_value=None)
        mock_model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        return mock_model

    def _make_mock_model_crf(self, seq_len=10):
        """Mock CRF model returning list[list[int]]."""
        mock_model = MagicMock()
        mock_model._use_crf = True
        mock_model.return_value = [[0, 1, 2, 2, 0, 0, 0, 0, 0, 0]]
        mock_model.eval = MagicMock(return_value=None)
        mock_model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        return mock_model

    def test_evaluate_model_returns_required_keys(self):
        """evaluate_model returns dict with precision, recall, f1, per_type, partial_match."""
        from omegaconf import OmegaConf
        from src.evaluation.evaluator import Evaluator

        cfg = OmegaConf.create({"model": {"max_length": 10}})
        evaluator = Evaluator(cfg)
        sample = self._make_gold_sample()
        mock_tok = self._make_mock_tokenizer()
        mock_model = self._make_mock_model_non_crf()
        device = torch.device("cpu")

        result = evaluator.evaluate_model(mock_model, mock_tok, [sample], device)

        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "per_type" in result
        assert "partial_match" in result

    def test_evaluate_model_non_crf_path(self):
        """Non-CRF model (logits) path executes without error."""
        from omegaconf import OmegaConf
        from src.evaluation.evaluator import Evaluator

        cfg = OmegaConf.create({"model": {"max_length": 10}})
        evaluator = Evaluator(cfg)
        sample = self._make_gold_sample()
        mock_tok = self._make_mock_tokenizer()
        mock_model = self._make_mock_model_non_crf()
        device = torch.device("cpu")

        result = evaluator.evaluate_model(mock_model, mock_tok, [sample], device)
        assert isinstance(result["f1"], float)

    def test_evaluate_model_crf_path(self):
        """CRF model (list[list[int]]) path executes without error."""
        from omegaconf import OmegaConf
        from src.evaluation.evaluator import Evaluator

        cfg = OmegaConf.create({"model": {"max_length": 10}})
        evaluator = Evaluator(cfg)
        sample = self._make_gold_sample()
        mock_tok = self._make_mock_tokenizer()
        mock_model = self._make_mock_model_crf()
        device = torch.device("cpu")

        result = evaluator.evaluate_model(mock_model, mock_tok, [sample], device)
        assert isinstance(result["f1"], float)

    def test_partial_match_keys(self):
        """partial_match sub-dict has precision, recall, f1."""
        from omegaconf import OmegaConf
        from src.evaluation.evaluator import Evaluator

        cfg = OmegaConf.create({"model": {"max_length": 10}})
        evaluator = Evaluator(cfg)
        sample = self._make_gold_sample()
        mock_tok = self._make_mock_tokenizer()
        mock_model = self._make_mock_model_non_crf()
        device = torch.device("cpu")

        result = evaluator.evaluate_model(mock_model, mock_tok, [sample], device)
        pm = result["partial_match"]
        assert "precision" in pm
        assert "recall" in pm
        assert "f1" in pm


# ---------------------------------------------------------------------------
# TestFPFNDump
# ---------------------------------------------------------------------------


class TestFPFNDump:
    """Tests for Evaluator.dump_errors() JSON output."""

    def _make_samples_and_preds(self):
        samples = [
            {"text": "§ 25a KWG gilt", "spans": [(0, 9)], "domain": "KWG"},
            {"text": "Art. 5 DSGVO", "spans": [(0, 12)], "domain": "DSGVO"},
        ]
        pred_spans = [
            [(0, 9)],    # correct
            [(0, 5)],    # partial — not in gold exactly
        ]
        return samples, pred_spans

    def test_dump_errors_creates_file(self):
        """dump_errors writes a JSON file at given path."""
        from omegaconf import OmegaConf
        from src.evaluation.evaluator import Evaluator

        cfg = OmegaConf.create({})
        evaluator = Evaluator(cfg)
        samples, preds = self._make_samples_and_preds()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "errors.json"
            evaluator.dump_errors(samples, preds, out_path)
            assert out_path.exists()

    def test_dump_errors_valid_json(self):
        """dump_errors output is valid JSON list."""
        from omegaconf import OmegaConf
        from src.evaluation.evaluator import Evaluator

        cfg = OmegaConf.create({})
        evaluator = Evaluator(cfg)
        samples, preds = self._make_samples_and_preds()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "errors.json"
            evaluator.dump_errors(samples, preds, out_path)
            data = json.loads(out_path.read_text())
            assert isinstance(data, list)
            assert len(data) == 2

    def test_dump_errors_required_fields(self):
        """Each error record has required fields."""
        from omegaconf import OmegaConf
        from src.evaluation.evaluator import Evaluator

        cfg = OmegaConf.create({})
        evaluator = Evaluator(cfg)
        samples, preds = self._make_samples_and_preds()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "errors.json"
            evaluator.dump_errors(samples, preds, out_path)
            data = json.loads(out_path.read_text())
            required = {"sample_idx", "text", "gold_spans", "pred_spans", "false_positives", "false_negatives", "domain"}
            for record in data:
                assert required.issubset(set(record.keys())), f"Missing keys: {required - set(record.keys())}"

    def test_dump_errors_false_negatives_correct(self):
        """False negatives = gold spans not in pred spans."""
        from omegaconf import OmegaConf
        from src.evaluation.evaluator import Evaluator

        cfg = OmegaConf.create({})
        evaluator = Evaluator(cfg)
        # gold has (0,9), pred has (5,15) — gold not in pred → FN
        samples = [{"text": "§ 25a KWG gilt", "spans": [(0, 9)], "domain": "KWG"}]
        preds = [[(5, 15)]]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "errors.json"
            evaluator.dump_errors(samples, preds, out_path)
            data = json.loads(out_path.read_text())
            record = data[0]
            # (0,9) is not in pred spans → should be FN
            assert [0, 9] in record["false_negatives"]


# ---------------------------------------------------------------------------
# TestComparisonReport
# ---------------------------------------------------------------------------


class TestComparisonReport:
    """Tests for Evaluator.format_comparison_report()."""

    def _make_comparison(self, ml_recall=0.85, baseline_recall=0.70):
        return {
            "ml": {"precision": 0.90, "recall": ml_recall, "f1": 0.87},
            "baseline": {"precision": 0.80, "recall": baseline_recall, "f1": 0.75},
            "delta": {
                "precision": 0.10,
                "recall": ml_recall - baseline_recall,
                "f1": 0.12,
            },
        }

    def test_format_comparison_report_is_string(self):
        """format_comparison_report returns a string."""
        from omegaconf import OmegaConf
        from src.evaluation.evaluator import Evaluator

        cfg = OmegaConf.create({})
        evaluator = Evaluator(cfg)
        comparison = self._make_comparison()
        report = evaluator.format_comparison_report(comparison)
        assert isinstance(report, str)

    def test_format_comparison_contains_metrics(self):
        """Report contains Precision, Recall, F1 labels."""
        from omegaconf import OmegaConf
        from src.evaluation.evaluator import Evaluator

        cfg = OmegaConf.create({})
        evaluator = Evaluator(cfg)
        comparison = self._make_comparison()
        report = evaluator.format_comparison_report(comparison)
        assert "Precision" in report or "precision" in report.lower()
        assert "Recall" in report or "recall" in report.lower()
        assert "F1" in report or "f1" in report.lower()

    def test_format_comparison_contains_verdict(self):
        """Report contains a Verdict line."""
        from omegaconf import OmegaConf
        from src.evaluation.evaluator import Evaluator

        cfg = OmegaConf.create({})
        evaluator = Evaluator(cfg)
        comparison = self._make_comparison()
        report = evaluator.format_comparison_report(comparison)
        assert "Verdict" in report or "verdict" in report.lower()

    def test_format_comparison_contains_delta(self):
        """Report contains a Delta column or section."""
        from omegaconf import OmegaConf
        from src.evaluation.evaluator import Evaluator

        cfg = OmegaConf.create({})
        evaluator = Evaluator(cfg)
        comparison = self._make_comparison()
        report = evaluator.format_comparison_report(comparison)
        assert "Delta" in report or "delta" in report.lower() or "+" in report or "-" in report

    def test_evaluate_comparison_returns_required_keys(self):
        """evaluate_comparison returns ml, baseline, delta keys."""
        from omegaconf import OmegaConf
        from src.evaluation.evaluator import Evaluator

        cfg = OmegaConf.create({"model": {"max_length": 10}})
        evaluator = Evaluator(cfg)

        # minimal gold sample
        sample = {
            "text": "§ 25a KWG gilt",
            "spans": [(0, 9)],
            "bio_labels": {
                "input_ids": [101] + [1000] * 8 + [102],
                "attention_mask": [1] * 10,
                "labels": [-100, 1, 2, 2, 2, 0, 0, 0, 0, -100],
            },
            "domain": "KWG",
        }

        # Mock model: non-CRF
        logits = torch.zeros(1, 10, 3)
        logits[0, 1, 1] = 10.0  # B-REF
        logits[0, 2, 2] = 10.0  # I-REF
        output = MagicMock()
        output.logits = logits
        mock_model = MagicMock()
        mock_model._use_crf = False
        mock_model.return_value = output
        mock_model.eval = MagicMock(return_value=None)

        mock_enc = {
            "input_ids": torch.tensor([[101] + [1000] * 8 + [102]]),
            "attention_mask": torch.tensor([[1] * 10]),
            "offset_mapping": [(0, 0), (0, 1), (1, 4), (4, 5), (5, 6), (7, 8), (8, 9), (9, 10), (10, 11), (0, 0)],
        }

        class MockEncoding:
            def __getitem__(self, key):
                return mock_enc[key]

            def to(self, device):
                return self

        mock_tok = MagicMock()
        mock_tok.return_value = MockEncoding()

        device = torch.device("cpu")
        result = evaluator.evaluate_comparison(mock_model, mock_tok, [sample], device)

        assert "ml" in result
        assert "baseline" in result
        assert "delta" in result


# ---------------------------------------------------------------------------
# TestPerTypeBreakdown
# ---------------------------------------------------------------------------


class TestPerTypeBreakdown:
    """Tests for per-reference-type breakdown in evaluate_model()."""

    def test_per_type_is_dict(self):
        """per_type result is a dict."""
        from omegaconf import OmegaConf
        from src.evaluation.evaluator import Evaluator

        cfg = OmegaConf.create({"model": {"max_length": 10}})
        evaluator = Evaluator(cfg)
        seq_len = 10
        sample = {
            "text": "§ 25a KWG gilt",
            "spans": [(0, 9)],
            "bio_labels": {
                "input_ids": [101] + [1000] * (seq_len - 2) + [102],
                "attention_mask": [1] * seq_len,
                "labels": [-100, 1, 2, 2, 2, 0, 0, 0, 0, -100],
            },
            "domain": "KWG",
        }

        logits = torch.zeros(1, seq_len, 3)
        logits[0, 1, 1] = 10.0
        logits[0, 2, 2] = 10.0
        output = MagicMock()
        output.logits = logits
        mock_model = MagicMock()
        mock_model._use_crf = False
        mock_model.return_value = output
        mock_model.eval = MagicMock(return_value=None)

        mock_enc = {
            "input_ids": torch.tensor([[101] + [1000] * 8 + [102]]),
            "attention_mask": torch.tensor([[1] * seq_len]),
            "offset_mapping": [(0, 0), (0, 1), (1, 4), (4, 5), (5, 6), (7, 8), (8, 9), (9, 10), (10, 11), (0, 0)],
        }

        class MockEncoding:
            def __getitem__(self, key):
                return mock_enc[key]

            def to(self, device):
                return self

        mock_tok = MagicMock()
        mock_tok.return_value = MockEncoding()

        result = evaluator.evaluate_model(mock_model, mock_tok, [sample], torch.device("cpu"))
        assert isinstance(result["per_type"], dict)


# ---------------------------------------------------------------------------
# Import check (pytest needs `pytest` for approx)
# ---------------------------------------------------------------------------
import pytest  # noqa: E402  (placed after class definitions to match test structure)
