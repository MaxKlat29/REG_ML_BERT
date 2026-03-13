---
phase: 2
slug: data-pipeline
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-13
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x + pytest-asyncio |
| **Config file** | pytest.ini (exists from Phase 1) |
| **Quick run command** | `pytest tests/ -x -q --timeout=10` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~8 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x -q --timeout=10`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 8 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 2-01-01 | 01 | 1 | DATA-01 | unit (mocked) | `pytest tests/test_llm_client.py::test_call_openrouter -x` | ❌ W0 | ⬜ pending |
| 2-01-02 | 01 | 1 | DATA-02 | unit | `pytest tests/test_llm_client.py::test_ref_tag_parsing -x` | ❌ W0 | ⬜ pending |
| 2-01-03 | 01 | 1 | DATA-07 | unit | `pytest tests/test_llm_client.py::test_seed_determinism -x` | ❌ W0 | ⬜ pending |
| 2-01-04 | 01 | 1 | DATA-08 | unit (mocked) | `pytest tests/test_llm_client.py::test_retry_on_rate_limit -x` | ❌ W0 | ⬜ pending |
| 2-01-05 | 01 | 1 | DATA-10 | unit | `pytest tests/test_llm_client.py::test_domain_rotation -x` | ❌ W0 | ⬜ pending |
| 2-02-01 | 02 | 1 | DATA-03 | unit | `pytest tests/test_bio_converter.py::test_char_to_bio -x` | ❌ W0 | ⬜ pending |
| 2-02-02 | 02 | 1 | DATA-04 | unit | `pytest tests/test_bio_converter.py::test_subword_labeling -x` | ❌ W0 | ⬜ pending |
| 2-02-03 | 02 | 1 | DATA-05 | unit | `pytest tests/test_bio_converter.py::test_special_token_masking -x` | ❌ W0 | ⬜ pending |
| 2-02-04 | 02 | 1 | DATA-09 | unit | `pytest tests/test_bio_converter.py::test_offset_validation -x` | ❌ W0 | ⬜ pending |
| 2-02-05 | 02 | 1 | DATA-06 | unit (mocked) | `pytest tests/test_dataset.py::test_iterable_yields -x` | ❌ W0 | ⬜ pending |
| 2-02-06 | 02 | 1 | DATA-11 | unit | `pytest tests/test_cache.py -x` | ❌ W0 | ⬜ pending |
| 2-03-01 | 03 | 2 | GOLD-01 | integration | `pytest tests/test_gold_builder.py::test_gold_generation -x` | ❌ W0 | ⬜ pending |
| 2-03-02 | 03 | 2 | GOLD-02 | unit | `pytest tests/test_gold_builder.py::test_needs_review_flag -x` | ❌ W0 | ⬜ pending |
| 2-03-03 | 03 | 2 | GOLD-03 | unit | `pytest tests/test_gold_builder.py::test_positive_negative_mix -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_llm_client.py` — covers DATA-01, DATA-02, DATA-07, DATA-08, DATA-10
- [ ] `tests/test_bio_converter.py` — covers DATA-03, DATA-04, DATA-05, DATA-09
- [ ] `tests/test_dataset.py` — covers DATA-06
- [ ] `tests/test_cache.py` — covers DATA-11
- [ ] `tests/test_gold_builder.py` — covers GOLD-01, GOLD-02, GOLD-03
- [ ] `pytest-asyncio` added to requirements.txt
- [ ] `pytest-timeout` added to requirements.txt (optional)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| OpenRouter API actually returns valid data | DATA-01 | Requires live API key + network | Run `python -c "from src.data.llm_client import ...; ..."` with .env loaded |
| LLM-generated text quality is realistic | DATA-02 | Subjective quality assessment | Inspect 10 generated samples visually |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 8s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
