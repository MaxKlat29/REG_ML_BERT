---
phase: 1
slug: foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-13
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pytest.ini (Wave 0 creates) |
| **Quick run command** | `pytest tests/ -x -q` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x -q`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | CONF-01 | unit | `pytest tests/test_config.py::test_load_default_config -x` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | CONF-02 | unit | `pytest tests/test_config.py::test_cli_override -x` | ❌ W0 | ⬜ pending |
| 1-01-03 | 01 | 1 | CONF-03 | unit | `pytest tests/test_config.py::test_seed_reproducibility -x` | ❌ W0 | ⬜ pending |
| 1-01-04 | 01 | 1 | CONF-04 | manual | manual — run once per environment | N/A | ⬜ pending |
| 1-02-01 | 02 | 1 | EVAL-02 | unit | `pytest tests/test_regex_baseline.py::test_all_reference_types -x` | ❌ W0 | ⬜ pending |
| 1-02-02 | 02 | 1 | EVAL-03 | unit | `pytest tests/test_metrics.py::test_entity_metrics -x` | ❌ W0 | ⬜ pending |
| 1-02-03 | 02 | 1 | DOCS-01 | unit | `pytest tests/test_docs.py::test_readme_exists -x` | ❌ W0 | ⬜ pending |
| 1-02-04 | 02 | 1 | DOCS-02 | unit | `pytest tests/test_docs.py::test_env_example -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/__init__.py` — package init
- [ ] `tests/conftest.py` — shared fixtures (sample sentences with known references)
- [ ] `tests/test_config.py` — covers CONF-01, CONF-02, CONF-03
- [ ] `tests/test_regex_baseline.py` — covers EVAL-02 (all 10 reference types)
- [ ] `tests/test_metrics.py` — covers EVAL-03 (known BIO → expected P/R/F1)
- [ ] `tests/test_docs.py` — covers DOCS-01, DOCS-02 (file existence)
- [ ] `pytest.ini` — pytest config with testpaths
- [ ] `pytest` added to requirements.txt

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| pip install resolves without conflicts | CONF-04 | Environment-dependent | Run `pip install -r requirements.txt` in fresh venv |
| Device detection matches hardware | CONF-04 | Hardware-dependent | Run `python -c "from src.utils.config import get_device; print(get_device())"` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
