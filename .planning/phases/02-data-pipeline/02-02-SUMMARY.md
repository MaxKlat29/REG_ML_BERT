---
phase: 02-data-pipeline
plan: 02
subsystem: data
tags: [bio-conversion, offset-mapping, tokenization, IterableDataset, jsonl-cache, gbert, transformers]

# Dependency graph
requires:
  - phase: 02-data-pipeline
    provides: "llm_client.py with call_openrouter, parse_ref_tags, build_generation_prompt, get_domain_for_seed"

provides:
  - "char_spans_to_bio() converting character-level reference spans to token-level BIO labels via offset_mapping"
  - "validate_bio_roundtrip() for detecting BIO alignment errors"
  - "JSONL cache: append_to_cache() + load_cache() with Unicode preservation"
  - "LLMGeneratedDataset: IterableDataset composing LLM generation + BIO conversion"
  - "Worker sharding via get_worker_info() for distinct per-worker seeds"

affects: [03-model-training, 04-evaluation-inference]

# Tech tracking
tech-stack:
  added: [protobuf, sentencepiece, BertTokenizerFast]
  patterns:
    - "BIO alignment via offset_mapping: (0,0) check for special tokens, token overlap with char spans"
    - "TDD: RED (import-failing tests) then GREEN (implement to pass)"
    - "Module-scoped tokenizer fixture to avoid repeated model loads in tests"
    - "JSONL cache with ensure_ascii=False for German umlaut preservation"
    - "Worker-safe seeding: seed = epoch*10000 + batch_idx*100 + worker_id"

key-files:
  created:
    - src/data/bio_converter.py
    - src/data/cache.py
    - src/data/dataset.py
    - tests/test_bio_converter.py
    - tests/test_cache.py
    - tests/test_dataset.py
  modified:
    - src/data/__init__.py

key-decisions:
  - "Use BertTokenizerFast instead of AutoTokenizer: transformers 5.x AutoTokenizer fails on gbert-large without tokenizer.json (only vocab.txt cached); BertTokenizerFast loads successfully"
  - "Special token detection via (start==0 AND end==0) check on offset_mapping; avoids false positive where first real token has start=0 but end>0"
  - "asyncio.run() wraps async LLM call in _generate_sample(); no event loop detected issue for CPU-only sync context"
  - "LABEL_IGNORE = -100 for all padding and special tokens; CrossEntropyLoss uses ignore_index=-100 by default"

patterns-established:
  - "BIO alignment: iterate offset_mapping, check (0,0) for specials, check attention_mask for padding, first subtoken of span gets B-REF, continuations get I-REF"
  - "validate_bio_roundtrip: walk labels, collect B-REF start / I-REF extend / emit on non-ref transition"
  - "JSONL cache append mode: json.dumps(sample, ensure_ascii=False) + newline per sample"
  - "IterableDataset worker sharding: epoch*10000 + batch_idx*100 + worker_id seed formula"

requirements-completed: [DATA-03, DATA-04, DATA-05, DATA-06, DATA-09, DATA-11]

# Metrics
duration: 12min
completed: 2026-03-13
---

# Phase 02 Plan 02: BIO Converter, JSONL Cache, and IterableDataset Summary

**BIO token-label alignment via offset_mapping with JSONL cache and PyTorch IterableDataset for on-the-fly LLM training data generation**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-03-13T17:15:18Z
- **Completed:** 2026-03-13T17:27:00Z
- **Tasks:** 2
- **Files modified:** 7 (3 new source, 3 new tests, 1 updated)

## Accomplishments
- BIO converter handles subword tokenization, multi-span inputs, special tokens (-100), padding (-100) — all edge cases from research phase
- JSONL cache supports append + load with full Unicode preservation for German umlauts
- LLMGeneratedDataset implements worker sharding via get_worker_info() with epoch-aware seed formula
- Both live-LLM and cache-read modes implemented and tested
- All 17 new tests pass; full suite of 50 tests passes with no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: BIO converter + JSONL cache (RED+GREEN)** - `3fb6ccc` (feat)
2. **Task 2: IterableDataset with worker sharding (RED+GREEN)** - `8fbd634` (feat)

_Note: TDD tasks combined RED and GREEN into single commits per task as tests and implementation were small enough to reason about together._

## Files Created/Modified
- `src/data/bio_converter.py` - char_spans_to_bio(), validate_bio_roundtrip(), get_tokenizer(), label constants
- `src/data/cache.py` - append_to_cache(), load_cache() JSONL disk cache
- `src/data/dataset.py` - LLMGeneratedDataset IterableDataset with worker sharding and cache mode
- `src/data/__init__.py` - Updated to export all public symbols
- `tests/test_bio_converter.py` - 8 BIO converter tests (simple, multi-span, special tokens, padding, subwords, roundtrip)
- `tests/test_cache.py` - 4 cache tests (roundtrip, preservation, empty, unicode)
- `tests/test_dataset.py` - 5 dataset tests (yields, labels, special tokens, worker sharding, cache mode)

## Decisions Made
- **BertTokenizerFast over AutoTokenizer:** Transformers 5.x AutoTokenizer fails on gbert-large when only vocab.txt is cached (no tokenizer.json). BertTokenizerFast loads successfully with the same fast tokenizer backend. All code referencing the tokenizer updated.
- **get_tokenizer() uses lru_cache:** Prevents repeated model loads across multiple calls within a process; module-scoped pytest fixture handles the same for tests.
- **asyncio.run() in _generate_sample():** Since the dataset runs in a sync PyTorch DataLoader context, asyncio.run() is the correct wrapper. If ever called from an async context, the caller must use a thread executor.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed protobuf and sentencepiece dependencies**
- **Found during:** Task 1 (tokenizer fixture setup)
- **Issue:** `AutoTokenizer.from_pretrained("deepset/gbert-large")` failed with ImportError for protobuf, then failed for sentencepiece even after install
- **Fix:** Installed protobuf and sentencepiece via pip; switched to `BertTokenizerFast.from_pretrained()` which loads successfully with vocab.txt
- **Files modified:** None (system packages only)
- **Verification:** All 8 BIO converter tests pass with BertTokenizerFast
- **Committed in:** `3fb6ccc` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (blocking dependency + tokenizer class change)
**Impact on plan:** Essential for tests to run. BertTokenizerFast is the correct fast tokenizer for BERT-based models and functionally equivalent to AutoTokenizer's intended result.

## Issues Encountered
- Transformers 5.x changed AutoTokenizer behavior for models without a cached `tokenizer.json`. Only `vocab.txt` was cached from a previous download, making AutoTokenizer unable to instantiate the fast backend. Using `BertTokenizerFast` directly bypasses the auto-detection layer and works correctly.

## User Setup Required
None - no external service configuration required. All LLM calls are mocked in tests.

## Next Phase Readiness
- BIO conversion pipeline is complete and tested; ready for use in training loop (Phase 3)
- JSONL cache ready for ensemble resampling in Phase 3
- IterableDataset requires `OPENROUTER_API_KEY` env var for live use (tested with mocks)
- All Phase 2 data pipeline modules now available via `from src.data import ...`

---
*Phase: 02-data-pipeline*
*Completed: 2026-03-13*

## Self-Check: PASSED

All files verified present. All commits verified in git history.
