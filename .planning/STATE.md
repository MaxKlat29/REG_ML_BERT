# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-13)

**Core value:** Reliably find every legal reference in German regulatory text (recall over precision)
**Current focus:** Phase 1 — Foundation

## Current Position

Phase: 1 of 4 (Foundation) -- COMPLETE
Plan: 2 of 2 in current phase
Status: Phase Complete
Last activity: 2026-03-13 — Completed 01-02 (regex baseline, evaluation metrics, README, .env.example)

Progress: [██░░░░░░░░] 22%

## Performance Metrics

**Velocity:**
- Total plans completed: 2
- Average duration: 3 min
- Total execution time: 0.10 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Foundation | 2/2 | 6 min | 3 min |
| 2. Data Pipeline | 0/3 | - | - |
| 3. Model + Training | 0/2 | - | - |
| 4. Evaluation + Inference | 0/2 | - | - |

**Recent Trend:**
- Last 5 plans: 01-01 (2 min), 01-02 (4 min)
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

### Pending Todos

None yet.

### Blockers/Concerns

- Research flag: Phase 2 needs care around BIO alignment edge cases for German compound words; OpenRouter async rate limits; IterableDataset worker seeding to prevent duplicate samples

## Session Continuity

Last session: 2026-03-13
Stopped at: Completed 01-02-PLAN.md — Phase 1 complete, ready for Phase 2
Resume file: None
