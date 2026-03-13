# Phase 4: Evaluation + Inference - Validation Strategy

**Phase:** 4
**Slug:** evaluation-inference
**Created:** 2026-03-13

## Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest |
| Quick run | `pytest tests/test_evaluator.py tests/test_predictor.py -x` |
| Full suite | `pytest tests/ -x` |

## Requirements → Test Map

| Req ID | Behavior | Test Type | Command | Status |
|--------|----------|-----------|---------|--------|
| EVAL-01 | ML model eval returns entity-level P/R/F1 | unit | `pytest tests/test_evaluator.py::TestMLEvaluation -x` | Wave 0 |
| EVAL-04 | Per-type breakdown by ref type | unit | `pytest tests/test_evaluator.py::TestPerTypeBreakdown -x` | Wave 0 |
| EVAL-05 | FP/FN dump file written | unit | `pytest tests/test_evaluator.py::TestFPFNDump -x` | Wave 0 |
| EVAL-06 | IoU partial match scoring | unit | `pytest tests/test_evaluator.py::TestIoUScoring -x` | Wave 0 |
| INFR-01 | Predictor returns char-offset spans | unit | `pytest tests/test_predictor.py::TestPredict -x` | Wave 0 |
| INFR-02 | Confidence scores in [0, 1] | unit | `pytest tests/test_predictor.py::TestConfidenceScores -x` | Wave 0 |
| INFR-03 | Batch prediction returns list-of-lists | unit | `pytest tests/test_predictor.py::TestBatchPredict -x` | Wave 0 |
| DOCS-03 | Google-style docstrings on public API | unit | `pytest tests/test_docs.py -x` | Wave 0 |

## Sampling Rate

- **Per task:** `pytest tests/test_evaluator.py tests/test_predictor.py -x`
- **Per wave:** `pytest tests/ -x`
- **Phase gate:** Full suite green (83+ tests) before verify-work

## Wave 0 Gaps

- [ ] `tests/test_evaluator.py` — EVAL-01, EVAL-04, EVAL-05, EVAL-06
- [ ] `tests/test_predictor.py` — INFR-01, INFR-02, INFR-03
- [ ] Extend `tests/test_docs.py` — DOCS-03
