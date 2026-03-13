---
phase: 02-data-pipeline
plan: 03
subsystem: data
tags: [gold-test, llm, bio-labels, evaluation, json, asyncio, tdd]

requires:
  - phase: 02-01
    provides: call_openrouter, parse_ref_tags, build_generation_prompt, get_domain_for_seed, DOMAIN_LIST
  - phase: 02-02
    provides: char_spans_to_bio, get_tokenizer, BIO label constants

provides:
  - scripts/generate_gold_test.py — CLI gold test set generator with fixed-seed determinism
  - generate_gold_set() importable function for use in tests and evaluation pipelines
  - data/gold_test/gold_test_set.json — frozen evaluation dataset (generated at runtime)
  - tests/test_gold_builder.py — 5 tests covering all gold set structural requirements

affects:
  - Phase 3 (model training) — gold set must be frozen before training starts
  - Phase 4 (evaluation) — evaluator reads data/gold_test/gold_test_set.json as ground truth

tech-stack:
  added: []
  patterns:
    - "Patch mock at call site (scripts.generate_gold_test.call_openrouter) not source module when function is imported via 'from'"
    - "Fixed deterministic split: first int(N*ratio) samples are negative, rest positive — no RNG needed"
    - "AsyncMock with async def side_effect for clean coroutine mocking in Python 3.14"

key-files:
  created:
    - scripts/__init__.py
    - scripts/generate_gold_test.py
    - tests/test_gold_builder.py
  modified: []

key-decisions:
  - "Mock call_openrouter at 'scripts.generate_gold_test.call_openrouter' (call site) not 'src.data.llm_client.call_openrouter' (source) because it is imported via 'from ... import'"
  - "Positive/negative split is index-based not RNG-based: first int(N*ratio) indices are negative, ensuring exact reproducibility without seeding random"
  - "Model name taken from config.model.name for tokenizer selection — makes gold set generation use the same tokenizer as training"
  - "asyncio.coroutine removed in Python 3.14 — use async def side_effect functions with AsyncMock instead"

patterns-established:
  - "Gold set generation pattern: fixed seed + index-based split + asyncio.gather for parallel LLM calls"
  - "TDD for async generators: mock at import namespace, use async def side_effect, call generate_gold_set synchronously from test"

requirements-completed: [GOLD-01, GOLD-02, GOLD-03]

duration: 3min
completed: 2026-03-13
---

# Phase 2 Plan 3: Gold Test Set Generator Summary

**Async gold test set generator with fixed-seed determinism, pos/neg split, BIO labels, and needs_review flag — 5 TDD tests green, 55/55 total suite passing**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-13T17:23:05Z
- **Completed:** 2026-03-13T17:25:48Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 3 created, 1 (scripts/__init__.py as package marker)

## Accomplishments

- Gold test set generator CLI script with deterministic seed-based positive/negative split and parallel async LLM calls via asyncio.gather
- All 5 required gold set structural tests pass with mocked LLM — no live API key needed during tests
- Full regression suite (55 tests) remains green with no existing test failures

## Task Commits

1. **Task 1: Gold test set builder tests (RED) and implementation (GREEN)** - `aff6444` (feat)

## Files Created/Modified

- `scripts/__init__.py` — makes scripts/ an importable Python package
- `scripts/generate_gold_test.py` — build_gold_prompt, generate_gold_set, save_gold_set, main; 175 lines
- `tests/test_gold_builder.py` — 5 TDD tests: JSON output, needs_review flag, pos/neg mix, required fields, seed reproducibility; 170 lines

## Decisions Made

- Mocking at call site (`scripts.generate_gold_test.call_openrouter`) is required when the function is imported via `from ... import` — patching the source module has no effect after the name is bound
- Fixed deterministic split uses index comparison (`sample_index >= num_negative`) rather than RNG, ensuring exact structural reproducibility across runs
- `asyncio.coroutine` was removed in Python 3.14 — all mock async side effects use `async def` functions instead
- Tokenizer sourced from `config.model.name` so gold set uses the same tokenizer as training

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Mock path corrected from source module to call site**
- **Found during:** Task 1 (GREEN phase — tests failing despite mock applied)
- **Issue:** Tests patched `src.data.llm_client.call_openrouter` but the function is imported by name in generate_gold_test.py, so patching the source has no effect — the real function was called and httpx made a live network request
- **Fix:** Changed all patch targets to `scripts.generate_gold_test.call_openrouter` (the bound name in the module under test)
- **Files modified:** tests/test_gold_builder.py
- **Verification:** All 5 tests pass with mocked LLM; no network calls made
- **Committed in:** aff6444

**2. [Rule 1 - Bug] Replaced asyncio.coroutine with async def in mock side effects**
- **Found during:** Task 1 (GREEN phase — Python 3.14 removed asyncio.coroutine)
- **Issue:** Original test drafts used `asyncio.coroutine(lambda: ...)()` — removed in Python 3.14, raises AttributeError
- **Fix:** Replaced with `async def` factory functions returning proper coroutines
- **Files modified:** tests/test_gold_builder.py
- **Verification:** All 5 tests pass cleanly with no deprecation errors
- **Committed in:** aff6444

---

**Total deviations:** 2 auto-fixed (both Rule 1 — bugs in test mock setup)
**Impact on plan:** Both fixes were in the test code, not production code. No scope creep. All planned functionality delivered as specified.

## Issues Encountered

- Python 3.14 removed `asyncio.coroutine` entirely — required updating mock patterns to use `async def` side effects. The production code was unaffected; only test helpers needed updating.

## User Setup Required

None for tests. To generate an actual gold set:

1. Set `OPENROUTER_API_KEY=sk-or-...` in your environment (or .env file)
2. Run from project root: `PYTHONPATH=. python scripts/generate_gold_test.py`
3. Outputs to `data/gold_test/gold_test_set.json` (50 samples, ~40% negative)
4. Review all samples before using for evaluation (all have `needs_review: true`)

## Next Phase Readiness

- Gold set generator is complete and frozen — Phase 2 data pipeline is fully done
- Phase 3 (Model + Training) can begin; gold set must be generated and reviewed before any model evaluation
- The evaluator in `scripts/evaluate.py` currently uses hardcoded demo samples — Phase 4 will wire it to `data/gold_test/gold_test_set.json`

## Self-Check: PASSED

All files verified present on disk. Commit aff6444 exists in git log.

---
*Phase: 02-data-pipeline*
*Completed: 2026-03-13*
