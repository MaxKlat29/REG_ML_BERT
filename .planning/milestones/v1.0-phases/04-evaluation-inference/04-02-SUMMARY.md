---
phase: 04-evaluation-inference
plan: "02"
subsystem: inference
tags: [predictor, cli, docstrings, inference, batch-predict, confidence-scores, tdd]
dependency_graph:
  requires:
    - src/model/ner_model.py
    - src/model/trainer.py (load_checkpoint, CHECKPOINT_BASE)
    - src/evaluation/metrics.py (decode_bio_to_char_spans)
    - src/evaluation/evaluator.py (evaluate_comparison, dump_errors, format_comparison_report)
    - src/data/bio_converter.py (LABEL_B_REF, LABEL_I_REF)
  provides:
    - Predictor class with predict() and predict_batch()
    - PredictedSpan dataclass (start, end, text, confidence)
    - Predictor.find_latest_checkpoint() classmethod
    - run.py evaluate subcommand (full implementation)
    - run.py predict subcommand (full implementation)
    - test_google_style_docstrings() docstring coverage guard
  affects:
    - Any notebook or script running end-to-end inference
    - CI — test_google_style_docstrings now prevents docstring regression

tech-stack:
  added:
    - torch.no_grad() context manager for inference efficiency
    - torch.softmax for confidence score computation
  patterns:
    - CRF vs non-CRF dual-path dispatch in predict() (mirrors evaluate_model pattern)
    - find_latest_checkpoint() uses rglob + stat().st_mtime for checkpoint discovery
    - predict_batch() simple loop over predict() — sufficient for PoC scale
    - AST-based docstring validator in test_docs.py for DOCS-03 regression prevention
    - _has_non_self_params / _has_non_none_return helpers make docstring check precise

key-files:
  created:
    - src/model/predictor.py
    - tests/test_predictor.py
  modified:
    - run.py
    - src/model/ner_model.py
    - tests/test_docs.py

key-decisions:
  - "predict() computes confidence as mean softmax prob at predicted label position for tokens in span — requires token-index scan over offset_mapping"
  - "CRF confidence always 1.0 — Viterbi decode does not produce marginals"
  - "predict_batch() is a simple loop over predict() — batch tokenization deferred as out of scope for PoC"
  - "test_google_style_docstrings uses AST parsing (not import/inspect) to avoid model loading side effects"
  - "Nested functions (closures inside methods) excluded from docstring checks — only module-level and class-level nodes checked"

patterns-established:
  - "CRF/non-CRF dual-path pattern: use getattr(model, '_use_crf', False) for safe dispatch"
  - "Docstring validation via ast.parse: cheaper and side-effect-free vs import-and-inspect"

requirements-completed: [INFR-01, INFR-02, INFR-03, DOCS-03]

duration: 4min
completed: 2026-03-13
---

# Phase 04 Plan 02: Predictor, CLI Subcommands, and Docstring Coverage Summary

**Predictor class with softmax-confidence char-offset spans, evaluate/predict CLI subcommands wired to Evaluator, and AST-based docstring regression guard covering all 14 src/ modules.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-13T20:26:57Z
- **Completed:** 2026-03-13T20:31:00Z
- **Tasks:** 2
- **Files modified:** 5 (2 created, 3 modified)

## Accomplishments

- Predictor class loads checkpoint, tokenizes text, dispatches CRF vs non-CRF, decodes BIO to char spans, computes softmax confidence per span
- predict_batch() handles arbitrary text lists; find_latest_checkpoint() discovers most recently modified .pt file via rglob
- run.py evaluate subcommand: loads model, runs evaluate_comparison(), prints table, dumps FP/FN errors.json
- run.py predict subcommand: handles --text (single) and --file (one-per-line batch) with formatted span output
- test_google_style_docstrings() uses ast.parse to check every public class/method in src/ for docstring + Args: + Returns: sections

## Task Commits

1. **Task 1: Predictor class (TDD RED→GREEN)** - `ced89ae` (feat)
2. **Task 2: CLI subcommands + docstrings** - `57fb8a2` (feat)

## Files Created/Modified

- `/Users/Admin/REG_ML/src/model/predictor.py` — PredictedSpan dataclass + Predictor class (predict, predict_batch, find_latest_checkpoint)
- `/Users/Admin/REG_ML/tests/test_predictor.py` — 13 tests: TestPredict, TestConfidenceScores, TestBatchPredict, TestNoReferences, TestFindLatestCheckpoint
- `/Users/Admin/REG_ML/run.py` — _run_evaluate() and _run_predict() replacing stubs; full argparse wiring
- `/Users/Admin/REG_ML/src/model/ner_model.py` — Added Args/Returns sections to forward(), use_crf property, get_bert_parameters(), get_head_parameters()
- `/Users/Admin/REG_ML/tests/test_docs.py` — test_google_style_docstrings() added for DOCS-03 compliance

## Decisions Made

- **CRF confidence = 1.0:** Viterbi decode doesn't produce token-level marginals; hard-coding 1.0 is the correct convention (documented in PredictedSpan docstring).
- **AST parsing for docstring tests:** Using `ast.parse` avoids importing torch/transformers/etc. during test collection — side-effect-free and fast.
- **Nested function exclusion:** Closures like `fmt()` inside `format_comparison_report()` are not public API; only top-level and class-level nodes checked.
- **predict_batch as loop:** Sequential calls are sufficient for PoC scale; batched tokenization can be added as a future optimization.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Mock tokenizer `__getitem__` on plain dict raises AttributeError**

- **Found during:** Task 1 GREEN phase — test run after writing predictor.py
- **Issue:** `_make_mock_tokenizer()` set `mock_tok.return_value = encoding` (plain dict), then tried `mock_tok.return_value.__getitem__ = ...` which raised `AttributeError: 'dict' object attribute '__getitem__' is read-only`
- **Fix:** Changed to assign a `MagicMock` for the encoding with a custom `__getitem__` lambda, keeping plain dict only as the data source
- **Files modified:** `tests/test_predictor.py`
- **Committed in:** `ced89ae` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — test helper bug)
**Impact on plan:** Bug was in test scaffolding, not in production code. All 13 predictor tests pass after fix.

## Issues Encountered

- 3 methods in ner_model.py had single-line docstrings missing `Returns:` sections — auto-detected by test_google_style_docstrings during Task 2 development and fixed inline.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Complete pipeline is now end-to-end: `python run.py train` → `python run.py evaluate` → `python run.py predict`
- Phase 4 (Evaluation + Inference) fully complete — both plans 04-01 and 04-02 done
- 146 tests total, all green
- To run a real evaluation: `python run.py evaluate --checkpoint checkpoints/<run_id>/epoch_N.pt`

---
*Phase: 04-evaluation-inference*
*Completed: 2026-03-13*
