"""Evaluator that runs the regex baseline and computes entity-level metrics.

Supports:
  - evaluate_baseline(): regex baseline P/R/F1
  - evaluate_model(): ML model inference with entity-level + per-type + IoU metrics
  - evaluate_comparison(): side-by-side ML vs regex comparison with deltas
  - dump_errors(): JSON FP/FN error analysis file
  - format_comparison_report(): human-readable comparison table
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from seqeval.metrics import classification_report as seqeval_report

from src.data.bio_converter import LABEL_B_REF, LABEL_I_REF, LABEL_IGNORE, LABEL_O
from src.evaluation.metrics import (
    classify_span_type,
    compute_entity_metrics,
    compute_partial_match_metrics,
    decode_bio_to_char_spans,
    spans_to_bio,
)
from src.evaluation.regex_baseline import RegexBaseline

logger = logging.getLogger(__name__)

_GOLD_TEST_DEFAULT = Path("data/gold_test/gold_test_set.json")


class Evaluator:
    """Runs baseline and ML model extraction and computes entity-level metrics."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.baseline = RegexBaseline()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_gold_set(self, path: str | Path | None = None) -> list[dict]:
        """Load gold_test_set.json.

        Args:
            path: Path to the JSON file. Defaults to data/gold_test/gold_test_set.json.

        Returns:
            List of gold sample dicts with spans converted to tuples.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(path) if path is not None else _GOLD_TEST_DEFAULT
        if not file_path.exists():
            raise FileNotFoundError(
                f"Gold test set not found at '{file_path}'. "
                "Run scripts/generate_gold_test.py first."
            )
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        samples = []
        for item in raw:
            item = dict(item)
            # Convert JSON arrays [start, end] to tuples (start, end)
            item["spans"] = [tuple(s) for s in item.get("spans", [])]
            samples.append(item)
        return samples

    def evaluate_baseline(self, samples: list[dict]) -> dict:
        """Evaluate the regex baseline against gold-annotated samples.

        Args:
            samples: List of dicts with keys:
                - "text": str — the input sentence
                - "spans": list[tuple[int, int]] — gold character-offset spans

        Returns:
            Dict with precision, recall, f1, report keys.
        """
        all_true: list[list[str]] = []
        all_pred: list[list[str]] = []

        for sample in samples:
            text = sample["text"]
            gold_spans = sample["spans"]
            pred_spans = self.baseline.extract(text)

            _, gold_labels = spans_to_bio(text, gold_spans)
            _, pred_labels = spans_to_bio(text, pred_spans)

            all_true.append(gold_labels)
            all_pred.append(pred_labels)

        return compute_entity_metrics(all_true, all_pred)

    def evaluate_model(
        self,
        model,
        tokenizer,
        samples: list[dict],
        device: torch.device,
    ) -> dict:
        """Run ML model inference on gold samples and compute entity-level metrics.

        Handles both CRF models (returning list[list[int]]) and non-CRF models
        (returning TokenClassifierOutput with .logits).

        Args:
            model: Trained RegulatoryNERModel (or any model with .forward()).
            tokenizer: Fast tokenizer that supports return_offsets_mapping.
            samples: Gold samples (each has "text", "spans", "bio_labels" keys).
            device: torch.device for inference.

        Returns:
            Dict with:
                precision, recall, f1 (float): seqeval entity-level metrics
                per_type (dict): per-reference-type P/R/F1 from seqeval
                partial_match (dict): IoU-based P/R/F1
                report (str): seqeval text classification report
        """
        max_length = int(self.cfg.get("model", {}).get("max_length", 512))

        model.eval()

        all_gold_seq: list[list[str]] = []
        all_pred_seq: list[list[str]] = []
        all_typed_gold: list[list[str]] = []
        all_typed_pred: list[list[str]] = []
        all_gold_char_spans: list[tuple[int, int]] = []
        all_pred_char_spans: list[tuple[int, int]] = []

        with torch.no_grad():
            for sample in samples:
                text = sample["text"]
                gold_char_spans: list[tuple[int, int]] = [
                    tuple(s) for s in sample.get("spans", [])
                ]
                bio_labels_dict = sample.get("bio_labels", {})
                gold_int_labels: list[int] = bio_labels_dict.get("labels", [])

                # Tokenize text to get offset_mapping
                enc = tokenizer(
                    text,
                    return_offsets_mapping=True,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                )

                # Build tensor inputs from gold bio_labels (which may differ in
                # padding from re-tokenizing) — use them for inference
                input_ids = torch.tensor(
                    [bio_labels_dict["input_ids"]], dtype=torch.long
                ).to(device)
                attention_mask = torch.tensor(
                    [bio_labels_dict["attention_mask"]], dtype=torch.long
                ).to(device)

                # Run model forward
                output = model(input_ids, attention_mask)

                # Decode predictions
                use_crf = getattr(model, "_use_crf", False)
                if use_crf:
                    # CRF: returns list[list[int]] (Viterbi decode)
                    pred_int_labels: list[int] = output[0]
                else:
                    # Non-CRF: returns TokenClassifierOutput with .logits
                    logits = output.logits  # shape (1, seq_len, num_labels)
                    pred_int_labels = logits.argmax(dim=-1)[0].tolist()

                # Align labels: filter positions where gold == -100 (specials/pad)
                gold_bio: list[str] = []
                pred_bio: list[str] = []
                gold_typed: list[str] = []
                pred_typed: list[str] = []

                # Build span-to-type mapping for typed BIO
                # For each gold span, classify its type
                span_type_map: dict[tuple[int, int], str] = {}
                for span in gold_char_spans:
                    span_text = text[span[0]:span[1]]
                    span_type_map[tuple(span)] = classify_span_type(span_text)

                offset_mapping: list[tuple[int, int]] = list(enc["offset_mapping"])

                for idx, gold_int in enumerate(gold_int_labels):
                    if gold_int == LABEL_IGNORE:
                        continue

                    pred_int = pred_int_labels[idx] if idx < len(pred_int_labels) else LABEL_O

                    # Generic BIO string labels
                    gold_bio.append(_int_to_bio(gold_int))
                    pred_bio.append(_int_to_bio(pred_int))

                    # Typed BIO: replace REF suffix with span type for gold
                    if gold_int == LABEL_B_REF:
                        g_type = _get_token_span_type(
                            idx, offset_mapping, text, gold_char_spans, span_type_map
                        )
                        gold_typed.append(f"B-{g_type}")
                        pred_typed.append(
                            f"B-{g_type}" if pred_int == LABEL_B_REF else _int_to_bio(pred_int)
                        )
                    elif gold_int == LABEL_I_REF:
                        g_type = _get_token_span_type(
                            idx, offset_mapping, text, gold_char_spans, span_type_map
                        )
                        gold_typed.append(f"I-{g_type}")
                        pred_typed.append(
                            f"I-{g_type}" if pred_int == LABEL_I_REF else _int_to_bio(pred_int)
                        )
                    else:
                        gold_typed.append("O")
                        pred_typed.append(_int_to_bio(pred_int))

                if gold_bio:
                    all_gold_seq.append(gold_bio)
                    all_pred_seq.append(pred_bio)
                if gold_typed:
                    all_typed_gold.append(gold_typed)
                    all_typed_pred.append(pred_typed)

                # IoU: decode pred int labels to char spans via offset_mapping
                pred_char_spans = decode_bio_to_char_spans(
                    pred_int_labels,
                    list(enc["offset_mapping"]),
                )
                all_gold_char_spans.extend(gold_char_spans)
                all_pred_char_spans.extend(pred_char_spans)

        # Compute metrics
        if all_gold_seq:
            base_metrics = compute_entity_metrics(all_gold_seq, all_pred_seq)
        else:
            base_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "report": ""}

        # Per-type breakdown via seqeval on typed labels
        if all_typed_gold:
            try:
                typed_report = seqeval_report(
                    all_typed_gold, all_typed_pred, output_dict=True
                )
                # Remove macro avg / weighted avg / micro avg from per_type
                per_type = {
                    k: v
                    for k, v in typed_report.items()
                    if k not in ("macro avg", "weighted avg", "micro avg")
                    and isinstance(v, dict)
                }
            except Exception:
                per_type = {}
        else:
            per_type = {}

        # IoU partial match
        partial_match = compute_partial_match_metrics(
            all_gold_char_spans,
            all_pred_char_spans,
            iou_threshold=0.5,
        )

        return {
            "precision": base_metrics["precision"],
            "recall": base_metrics["recall"],
            "f1": base_metrics["f1"],
            "report": base_metrics.get("report", ""),
            "per_type": per_type,
            "partial_match": partial_match,
        }

    def evaluate_comparison(
        self,
        model,
        tokenizer,
        samples: list[dict],
        device: torch.device,
    ) -> dict:
        """Run side-by-side ML model vs regex baseline evaluation.

        Args:
            model: Trained ML model.
            tokenizer: Fast tokenizer.
            samples: Gold samples.
            device: Torch device.

        Returns:
            Dict with keys:
                ml (dict): ML model metrics (precision, recall, f1, ...)
                baseline (dict): Regex baseline metrics
                delta (dict): ML minus baseline for precision, recall, f1
        """
        ml_metrics = self.evaluate_model(model, tokenizer, samples, device)
        baseline_metrics = self.evaluate_baseline(samples)

        delta = {
            "precision": ml_metrics["precision"] - baseline_metrics["precision"],
            "recall": ml_metrics["recall"] - baseline_metrics["recall"],
            "f1": ml_metrics["f1"] - baseline_metrics["f1"],
        }

        return {
            "ml": ml_metrics,
            "baseline": baseline_metrics,
            "delta": delta,
        }

    def dump_errors(
        self,
        samples: list[dict],
        pred_spans_per_sample: list[list[tuple[int, int]]],
        output_path: str | Path,
    ) -> None:
        """Write per-sample FP/FN analysis to a JSON file.

        Args:
            samples: Gold samples (each has "text", "spans", "domain" keys).
            pred_spans_per_sample: Predicted char spans per sample.
            output_path: Path to output JSON file.
        """
        records = []
        for idx, (sample, pred_spans) in enumerate(
            zip(samples, pred_spans_per_sample)
        ):
            gold_spans: list[tuple[int, int]] = [
                tuple(s) for s in sample.get("spans", [])
            ]
            pred_set = set(map(tuple, pred_spans))
            gold_set = set(map(tuple, gold_spans))

            false_positives = [list(s) for s in sorted(pred_set - gold_set)]
            false_negatives = [list(s) for s in sorted(gold_set - pred_set)]

            records.append(
                {
                    "sample_idx": idx,
                    "text": sample.get("text", ""),
                    "gold_spans": [list(s) for s in sorted(gold_set)],
                    "pred_spans": [list(s) for s in sorted(pred_set)],
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                    "domain": sample.get("domain", ""),
                }
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def format_comparison_report(self, comparison: dict) -> str:
        """Format side-by-side ML vs regex comparison as a human-readable table.

        Args:
            comparison: Dict with ml, baseline, delta keys (from evaluate_comparison).

        Returns:
            Formatted string with table and Verdict line.
        """
        ml = comparison["ml"]
        baseline = comparison["baseline"]
        delta = comparison["delta"]

        def fmt(v: float) -> str:
            return f"{v:.4f}"

        def fmt_delta(v: float) -> str:
            sign = "+" if v >= 0 else ""
            return f"{sign}{v:.4f}"

        width = 62
        lines = [
            "=" * width,
            "  ML Model vs Regex Baseline — Comparison Report",
            "=" * width,
            f"  {'Metric':<12} {'ML Model':>12} {'Regex Baseline':>16} {'Delta':>10}",
            "-" * width,
            f"  {'Precision':<12} {fmt(ml['precision']):>12} {fmt(baseline['precision']):>16} {fmt_delta(delta['precision']):>10}",
            f"  {'Recall':<12} {fmt(ml['recall']):>12} {fmt(baseline['recall']):>16} {fmt_delta(delta['recall']):>10}",
            f"  {'F1-Score':<12} {fmt(ml['f1']):>12} {fmt(baseline['f1']):>16} {fmt_delta(delta['f1']):>10}",
            "-" * width,
        ]

        # Verdict: focus on recall (PoC goal is recall over precision)
        recall_delta = delta["recall"]
        if recall_delta > 0.01:
            verdict = f"Verdict: ML model IMPROVES recall by {recall_delta:+.4f} over regex baseline."
        elif recall_delta < -0.01:
            verdict = f"Verdict: ML model REDUCES recall by {recall_delta:.4f} vs regex baseline."
        else:
            verdict = f"Verdict: ML model recall is ON PAR with regex baseline (delta={recall_delta:+.4f})."

        lines.append(f"  {verdict}")
        lines.append("=" * width)

        return "\n".join(lines)

    def format_report(self, metrics: dict) -> str:
        """Pretty-print Precision/Recall/F1 table.

        Args:
            metrics: Dict returned by evaluate_baseline().

        Returns:
            Formatted string with evaluation results.
        """
        lines = [
            "=" * 50,
            "  Regex Baseline Evaluation Report",
            "=" * 50,
            "",
            f"  Precision:  {metrics['precision']:.4f}",
            f"  Recall:     {metrics['recall']:.4f}",
            f"  F1-Score:   {metrics['f1']:.4f}",
            "",
            "  Detailed Report:",
            "-" * 50,
            metrics["report"],
            "=" * 50,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _int_to_bio(label: int) -> str:
    """Convert integer BIO label to seqeval string format."""
    if label == LABEL_B_REF:
        return "B-REF"
    if label == LABEL_I_REF:
        return "I-REF"
    return "O"


def _get_token_span_type(
    token_idx: int,
    offset_mapping: list[tuple[int, int]],
    text: str,
    gold_char_spans: list[tuple[int, int]],
    span_type_map: dict[tuple[int, int], str],
) -> str:
    """Find the reference type for a token at given index.

    Looks up which gold span the token falls inside, then returns that
    span's pre-computed type. Falls back to "REF".
    """
    if token_idx >= len(offset_mapping):
        return "REF"
    tok_start, tok_end = offset_mapping[token_idx]
    if tok_start == 0 and tok_end == 0:
        return "REF"

    for span in gold_char_spans:
        s_start, s_end = span
        # Token overlaps with span
        if tok_start < s_end and tok_end > s_start:
            return span_type_map.get(tuple(span), "REF")
    return "REF"
