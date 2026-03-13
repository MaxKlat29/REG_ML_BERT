---
phase: 3
slug: model-training
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-13
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x + pytest-asyncio |
| **Config file** | pytest.ini (exists from Phase 1) |
| **Quick run command** | `pytest tests/test_ner_model.py tests/test_trainer.py -x -q --timeout=15` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~12 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_ner_model.py tests/test_trainer.py -x -q --timeout=15`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 3-01-01 | 01 | 1 | MODL-01 | unit | `pytest tests/test_ner_model.py::test_forward_shape -x` | ❌ W0 | ⬜ pending |
| 3-01-02 | 01 | 1 | MODL-02 | unit | `pytest tests/test_ner_model.py::test_crf_toggle -x` | ❌ W0 | ⬜ pending |
| 3-01-03 | 01 | 1 | MODL-07 | unit | `pytest tests/test_ner_model.py::test_freeze_toggle -x` | ❌ W0 | ⬜ pending |
| 3-01-04 | 01 | 1 | MODL-07 | unit | `pytest tests/test_ner_model.py::test_lora_toggle -x` | ❌ W0 | ⬜ pending |
| 3-01-05 | 01 | 1 | MODL-08 | unit | `pytest tests/test_ner_model.py::test_cpu_forward -x` | ❌ W0 | ⬜ pending |
| 3-02-01 | 02 | 1 | MODL-03 | unit | `pytest tests/test_trainer.py::test_differential_lr -x` | ❌ W0 | ⬜ pending |
| 3-02-02 | 02 | 1 | MODL-04 | unit | `pytest tests/test_trainer.py::test_mixed_precision_resolution -x` | ❌ W0 | ⬜ pending |
| 3-02-03 | 02 | 1 | MODL-05 | unit | `pytest tests/test_trainer.py::test_gradient_clipping -x` | ❌ W0 | ⬜ pending |
| 3-02-04 | 02 | 1 | MODL-06 | unit | `pytest tests/test_trainer.py::test_lr_schedule_shape -x` | ❌ W0 | ⬜ pending |
| 3-02-05 | 02 | 1 | ENSM-01 | unit | `pytest tests/test_trainer.py::test_ensemble_n_checkpoints -x` | ❌ W0 | ⬜ pending |
| 3-02-06 | 02 | 1 | ENSM-02 | unit | `pytest tests/test_trainer.py::test_ensemble_cache_handoff -x` | ❌ W0 | ⬜ pending |
| 3-02-07 | 02 | 1 | ENSM-03 | unit | `pytest tests/test_trainer.py::test_ensemble_majority_vote -x` | ❌ W0 | ⬜ pending |
| 3-02-08 | 02 | 1 | ENSM-04 | unit | `pytest tests/test_trainer.py::test_ensemble_gradient_boost_stub -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_ner_model.py` — covers MODL-01, MODL-02, MODL-07, MODL-08
- [ ] `tests/test_trainer.py` — covers MODL-03, MODL-04, MODL-05, MODL-06, ENSM-01, ENSM-02, ENSM-03, ENSM-04
- [ ] `src/model/__init__.py` — package init
- [ ] `run.py` — CLI entry point stub
- [ ] `requirements.txt` update — add `accelerate`, `pytorch-crf`, `peft`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Training completes on MPS without OOM | MODL-04/08 | Requires Apple Silicon hardware | Run `python run.py train` on M1 with batch_size=4 |
| Training completes on CUDA with fp16 | MODL-04/08 | Requires NVIDIA GPU | Run `python run.py train` on RTX with mixed_precision=fp16 |
| Checkpoint loads and resumes correctly | MODL-01 | Requires completed training run | Interrupt training, resume from checkpoint, verify loss continuity |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
