---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: in_progress
stopped_at: Completed 02-02-PLAN.md — BIO converter, JSONL cache, IterableDataset
last_updated: "2026-03-13T17:30:00Z"
last_activity: 2026-03-13 — Completed 02-02 (bio_converter, cache, LLMGeneratedDataset, 17 tests)
progress:
  total_phases: 4
  completed_phases: 1
  total_plans: 4
  completed_plans: 4
  percent: 44
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-13)

**Core value:** Reliably find every legal reference in German regulatory text (recall over precision)
**Current focus:** Phase 2 — Data Pipeline

## Current Position

Phase: 2 of 4 (Data Pipeline) -- In Progress
Plan: 2 of 3 in current phase (02-01, 02-02 complete)
Status: In Progress
Last activity: 2026-03-13 — Completed 02-02 (BIO converter, JSONL cache, IterableDataset, 17 tests)

Progress: [████░░░░░░] 44%

## Performance Metrics

**Velocity:**
- Total plans completed: 2
- Average duration: 3 min
- Total execution time: 0.10 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Foundation | 2/2 | 6 min | 3 min |
| 2. Data Pipeline | 2/3 | 15 min | 7.5 min |
| 3. Model + Training | 0/2 | - | - |
| 4. Evaluation + Inference | 0/2 | - | - |

**Recent Trend:**
- Last 5 plans: 01-01 (2 min), 01-02 (4 min), 02-01 (3 min)
- Trend: steady

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Roadmap: CRF and ensemble are Phase 3 config-toggle features, not a separate phase (coarse granularity)
- Roadmap: Gold test set builder placed in Phase 2 — must be frozen before any model training begins
- Roadmap: DOCS-03 (docstrings) deferred to Phase 4 as final polish alongside CLI integration
- 01-01: Used OmegaConf.from_dotlist() for explicit overrides, from_dotlist with stripped "--" for CLI
- 01-01: Config immutable after load_config() returns
- 01-02: Used regex PyPI package (not stdlib re) for German legal reference patterns
- 01-02: Law abbreviation matching requires 2+ uppercase letters to reduce false positives
- 01-02: BIO labels use B-REF/I-REF/O format per seqeval IOB2 requirement
- 02-01: Used stdlib re (not regex PyPI) for ref-tag parsing — simple non-nested case
- 02-01: raise_for_status() guarded by status >= 400 — httpx.Response without request raises RuntimeError even on 200
- 02-01: Tenacity retry wait patched via call_openrouter.retry.wait = wait_none() in tests
- 02-01: DOMAIN_LIST has 13 entries; domain rotation via seed % len(DOMAIN_LIST)
- 02-02: BertTokenizerFast instead of AutoTokenizer — transformers 5.x AutoTokenizer fails on gbert-large without tokenizer.json
- 02-02: Special token detection via (start==0 AND end==0) on offset_mapping; first real token can have start=0 but end>0
- 02-02: Worker seed formula: epoch*10000 + batch_idx*100 + worker_id for distinct samples per worker

### Pending Todos

None yet.

### Blockers/Concerns

- Research flag: Phase 2 BIO alignment edge cases addressed and tested. Remaining: OpenRouter async rate limits (handled by retry in 02-01), worker seeding (addressed in 02-02).

## Session Continuity

Last session: 2026-03-13
Stopped at: Completed 02-02-PLAN.md — BIO converter + JSONL cache + IterableDataset complete, next is 02-03
Resume file: None
