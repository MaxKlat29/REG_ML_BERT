---
phase: 02-data-pipeline
plan: 01
subsystem: data
tags: [llm, openrouter, httpx, tenacity, async, retry, ref-tags, german-regulatory, tdd]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: config system (load_config, default.yaml data.llm_model, data.llm_seed keys)
provides:
  - Async OpenRouter HTTP client with exponential-backoff retry for 429/408/502/503
  - parse_ref_tags: clean-text extraction + validated character-offset spans from <ref> tags
  - build_generation_prompt: German regulatory text prompt builder (positive/negative modes)
  - get_domain_for_seed: deterministic domain rotation over 13 German regulatory domains
  - RetryableAPIError custom exception
affects: [02-02, 02-03, 03-model-training, 04-evaluation-inference]

# Tech tracking
tech-stack:
  added: [httpx>=0.27.0, tenacity>=8.2.0, transformers>=4.40.0, pytest-asyncio>=0.23.0]
  patterns:
    - "Tenacity @retry on async functions with wait attribute patched in tests for instant retries"
    - "httpx.Response.raise_for_status() only called for status >= 400 (not on 2xx success)"
    - "TDD: RED (ImportError) -> GREEN (implementation) with auto-fix mid-GREEN"

key-files:
  created:
    - src/data/__init__.py
    - src/data/llm_client.py
    - tests/test_llm_client.py
  modified:
    - requirements.txt
    - pytest.ini

key-decisions:
  - "Use stdlib re (not regex PyPI) for ref-tag parsing — simple non-nested case only"
  - "call_openrouter.retry.wait patched to wait_none() in retry tests to avoid slow waits"
  - "raise_for_status() guarded by status >= 400 check — httpx.Response without request object raises RuntimeError on raise_for_status() even for 200"
  - "DOMAIN_LIST has 13 entries (BGB, KWG, MaRisk, DORA, DSGVO, CRR, HGB, WpHG, VAG, ZAG, GwG, SAG, KAGB)"
  - "asyncio_mode = auto added to pytest.ini so all async test functions are picked up without @pytest.mark.asyncio"

patterns-established:
  - "Retry pattern: @retry on async function, patch .retry.wait = wait_none() in tests"
  - "Ref-tag parsing: track clean_offset as tags are removed, build clean text and spans in single pass"
  - "Domain rotation: seed % len(DOMAIN_LIST) — deterministic, no randomness"

requirements-completed: [DATA-01, DATA-02, DATA-07, DATA-08, DATA-10]

# Metrics
duration: 3min
completed: 2026-03-13
---

# Phase 2 Plan 1: LLM Client Summary

**Async OpenRouter client using httpx + tenacity with 5-attempt exponential backoff, ref-tag parser with validated character-offset spans, and deterministic domain rotation over 13 German regulatory domains**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-13T17:14:45Z
- **Completed:** 2026-03-13T17:17:47Z
- **Tasks:** 2 (TDD: RED + GREEN)
- **Files modified:** 5

## Accomplishments
- Async `call_openrouter` function with tenacity retry for 429/408/502/503 status codes
- `parse_ref_tags` single-pass parser producing clean text + validated (start, end) spans
- `build_generation_prompt` returning German prompts in positive (with `<ref>` markers) and negative (KEINE Verweise) modes
- 13-domain DOMAIN_LIST with `get_domain_for_seed` for deterministic per-seed rotation
- 12 pytest-asyncio tests all green with mocked httpx (no live API calls)

## Task Commits

1. **Task 1: Write failing tests for LLM client (RED)** - `d3a6194` (test)
2. **Task 2: Implement LLM client to pass all tests (GREEN)** - `50271f8` (feat)

_Note: TDD tasks have two commits (test RED → feat GREEN)_

## Files Created/Modified
- `src/data/__init__.py` - Data package init
- `src/data/llm_client.py` - Async OpenRouter client with retry, ref-tag parser, prompt builder
- `tests/test_llm_client.py` - 12 unit tests (mocked httpx, retry, ref parsing, domain rotation)
- `requirements.txt` - Added httpx, tenacity, transformers, pytest-asyncio
- `pytest.ini` - Added asyncio_mode = auto

## Decisions Made
- Used stdlib `re` (not regex PyPI package) for ref-tag parsing; ref tags are simple non-nested and don't need the full regex feature set
- `raise_for_status()` is only invoked when `response.status_code >= 400` — calling it on a 200 response that was constructed without a request object raises RuntimeError in httpx
- Tenacity retry wait is patched via `call_openrouter.retry.wait = wait_none()` in tests to avoid actual sleep delays while still exercising retry logic
- `asyncio_mode = auto` added to pytest.ini so all coroutine test functions are collected as async tests without explicit decorators

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] guard raise_for_status() for non-2xx only**
- **Found during:** Task 2 (GREEN implementation)
- **Issue:** `httpx.Response(200, json=...)` constructed without a request object raises `RuntimeError: Cannot call raise_for_status as the request instance has not been set` — even on a successful 200 response
- **Fix:** Added `if response.status_code >= 400: response.raise_for_status()` guard so 2xx responses are handled by direct JSON parsing, not raise_for_status
- **Files modified:** `src/data/llm_client.py`
- **Verification:** `test_call_openrouter_success` passes
- **Committed in:** `50271f8` (Task 2 commit)

**2. [Rule 1 - Bug] fix tenacity wait patching in retry tests**
- **Found during:** Task 2 (GREEN implementation)
- **Issue:** Tests used `patch("tenacity.wait.wait_exponential_jitter.__call__", return_value=0)` which doesn't affect the already-bound wait instance on the decorator; retry tests still waited 1-60 seconds or failed with wrong assertion
- **Fix:** Updated tests to directly assign `call_openrouter.retry.wait = wait_none()` before each retry test, restoring original wait in `finally` block
- **Files modified:** `tests/test_llm_client.py`
- **Verification:** All retry tests pass in ~0.1s total
- **Committed in:** `50271f8` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - Bug)
**Impact on plan:** Both fixes required for tests to pass. No scope creep — no new features added.

## Issues Encountered
- `test_bio_converter.py` has 8 pre-existing errors (requires `deepset/gbert-large` download + sentencepiece). These were present before this plan and are out of scope.

## Next Phase Readiness
- `call_openrouter`, `parse_ref_tags`, `build_generation_prompt`, `get_domain_for_seed` all exported and tested
- Ready for Plan 02-02 (BIO converter + JSONL cache) and Plan 02-03 (IterableDataset + gold test builder)
- OPENROUTER_API_KEY env var must be set for live usage (tests mock HTTP, no key needed)

## Self-Check: PASSED

- src/data/__init__.py: FOUND
- src/data/llm_client.py: FOUND
- tests/test_llm_client.py: FOUND
- .planning/phases/02-data-pipeline/02-01-SUMMARY.md: FOUND
- Commit d3a6194 (test RED): FOUND
- Commit 50271f8 (feat GREEN): FOUND

---
*Phase: 02-data-pipeline*
*Completed: 2026-03-13*
