---
phase: 04-evaluation-inference
plan: "01"
subsystem: evaluation
tags: [evaluation, metrics, ner, seqeval, iou, regex-baseline, ml-inference]
dependency_graph:
  requires:
    - src/model/ner_model.py
    - src/data/bio_converter.py
    - src/evaluation/metrics.py (pre-existing spans_to_bio, compute_entity_metrics)
    - src/evaluation/regex_baseline.py (pre-existing RegexBaseline.extract)
  provides:
    - span_iou()
    - classify_span_type()
    - decode_bio_to_char_spans()
    - compute_partial_match_metrics()
    - RegexBaseline.TYPED_PATTERNS
    - RegexBaseline.extract_typed()
    - Evaluator.evaluate_model()
    - Evaluator.evaluate_comparison()
    - Evaluator.dump_errors()
    - Evaluator.format_comparison_report()
    - Evaluator.load_gold_set()
  affects:
    - Any script or notebook calling Evaluator for PoC verdict
tech_stack:
  added:
    - seqeval classification_report with output_dict=True for per-type breakdown
    - regex PyPI for typed pattern matching in metrics.py
  patterns:
    - TDD (RED → GREEN) for both tasks
    - Greedy 1:1 IoU matching for partial span metrics
    - CRF/non-CRF dual-path dispatch in evaluate_model()
key_files:
  created:
    - tests/test_evaluator.py
  modified:
    - src/evaluation/metrics.py
    - src/evaluation/regex_baseline.py
    - src/evaluation/evaluator.py
decisions:
  - "classify_span_type uses regex.search (not \b word boundary) because dots are non-word chars and \b fails after Art."
  - "compute_partial_match_metrics: empty pred + non-empty gold → precision=1.0 (undefined, generous convention); empty gold + non-empty pred → precision=0.0"
  - "TYPED_PATTERNS prioritizes TEILZIFFER before PARAGRAPH to avoid Tz. being subsumed by §-based patterns"
  - "Per-type BIO uses typed span text at B position to classify; I tokens inherit same type as opening B"
metrics:
  duration: "8 min"
  completed_date: "2026-03-13"
  tasks_completed: 2
  files_modified: 4
  tests_added: 49
  tests_total: 132
---

# Phase 04 Plan 01: Evaluation Subsystem — ML Inference & Comparison Summary

**One-liner:** IoU partial matching + per-reference-type seqeval breakdown + CRF/non-CRF model evaluator with FP/FN JSON dump and ML-vs-regex comparison table.

## What Was Built

### Task 1: Utility Functions (metrics.py + regex_baseline.py)

Four new pure functions added to `src/evaluation/metrics.py`:

1. **`span_iou(pred, gold) -> float`** — Character-offset IoU with zero-length degenerate handling (returns 1.0 when both spans are zero-length at same position, union=0).

2. **`classify_span_type(span_text, full_text="") -> str`** — Pattern-based classifier returning one of 9 types (PARAGRAPH, ARTIKEL, ABSATZ, NUMMER, LITERAL, SATZ, TEILZIFFER, ANHANG, VERORDNUNG) or REF fallback. Uses `regex.search` — not `\b` word boundary — because dots are non-word chars and `\b` fails after "Art.".

3. **`decode_bio_to_char_spans(token_labels, offset_mapping) -> list[tuple]`** — Reconstructs character spans from integer BIO labels and tokenizer offset_mapping. Skips (0,0) special tokens. Generalizes `validate_bio_roundtrip` from bio_converter.py but returns spans instead of comparing.

4. **`compute_partial_match_metrics(gold, pred, iou_threshold=0.5) -> dict`** — Greedy 1:1 IoU matching: for each pred span, find best-IoU unmatched gold span, count TP if IoU > threshold. Returns P/R/F1 dict.

Two additions to `src/evaluation/regex_baseline.py`:

5. **`RegexBaseline.TYPED_PATTERNS`** — Class-level dict of 5 named compiled patterns (TEILZIFFER, PARAGRAPH, ARTIKEL, ANHANG, VERORDNUNG). TEILZIFFER is first to avoid its numeric suffix being eaten by § patterns.

6. **`RegexBaseline.extract_typed(text) -> list[tuple[int,int,str]]`** — Iterates TYPED_PATTERNS, collects all (start, end, type) candidates, sorts by start/length, deduplicates overlapping spans (keeps longest).

### Task 2: Evaluator Extension (evaluator.py)

Five new methods on the `Evaluator` class in `src/evaluation/evaluator.py`:

1. **`load_gold_set(path=None)`** — Loads gold_test_set.json (default: `data/gold_test/gold_test_set.json`), converts JSON arrays to tuples, raises `FileNotFoundError` with clear message if missing.

2. **`evaluate_model(model, tokenizer, samples, device)`** — Full ML inference pipeline:
   - Tokenizes each sample with `return_offsets_mapping=True`
   - Dispatches on `model._use_crf`: non-CRF → `output.logits.argmax(dim=-1)`; CRF → `output[0]` (Viterbi list)
   - Filters `LABEL_IGNORE` positions when building seqeval sequences
   - Builds typed BIO sequences (B-PARAGRAPH, I-ARTIKEL, etc.) using `classify_span_type` on gold spans
   - Runs seqeval for per-type breakdown via `classification_report(output_dict=True)`
   - Runs `decode_bio_to_char_spans` + `compute_partial_match_metrics` for IoU metrics
   - Returns: `{precision, recall, f1, report, per_type, partial_match}`

3. **`evaluate_comparison(model, tokenizer, samples, device)`** — Runs both `evaluate_model` and `evaluate_baseline` on same samples, returns `{ml, baseline, delta}`.

4. **`dump_errors(samples, pred_spans_per_sample, output_path)`** — Writes JSON list of `{sample_idx, text, gold_spans, pred_spans, false_positives, false_negatives, domain}` records.

5. **`format_comparison_report(comparison)`** — Produces 62-char-wide table with Metric/ML Model/Regex Baseline/Delta columns, plus a Verdict line interpreting recall delta (the PoC's primary signal).

## Test Coverage

49 new tests across 8 test classes in `tests/test_evaluator.py`:

| Class | Tests | What it validates |
|-------|-------|-------------------|
| TestIoUScoring | 6 | span_iou edge cases |
| TestSpanTypeClassification | 12 | All 9 types + fallback |
| TestBIODecode | 5 | Special token skipping, multi-span, edge cases |
| TestPartialMatchMetrics | 7 | P/R/F1 with IoU threshold |
| TestTypedExtraction | 5 | Dedup, sorting, TYPED_PATTERNS attribute |
| TestMLEvaluation | 4 | CRF/non-CRF paths, required keys |
| TestFPFNDump | 4 | JSON structure, FP/FN correctness |
| TestComparisonReport | 5 | Table format, Verdict, delta, evaluate_comparison keys |
| TestPerTypeBreakdown | 1 | per_type is dict |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `\b` word boundary fails after dots in classify_span_type**

- **Found during:** Task 1 GREEN phase — `classify_span_type("Art. 5 DSGVO")` returned "REF"
- **Issue:** Pattern `(?:Art\.|Artikel)\b` — `\b` requires adjacent word character, but "." is not a word char so the boundary never matches
- **Fix:** Removed `\b` from all type patterns in `_TYPE_PATTERNS`; `regex.search` without boundary is sufficient since we match on span text, not full document
- **Files modified:** `src/evaluation/metrics.py`
- **Commit:** included in `0b969b3`

## Self-Check: PASSED

All files verified present and commits confirmed in git log:
- `src/evaluation/metrics.py` — FOUND
- `src/evaluation/regex_baseline.py` — FOUND
- `src/evaluation/evaluator.py` — FOUND
- `tests/test_evaluator.py` — FOUND
- `.planning/phases/04-evaluation-inference/04-01-SUMMARY.md` — FOUND
- Commit `0b969b3` (Task 1) — FOUND
- Commit `01240e5` (Task 2) — FOUND
