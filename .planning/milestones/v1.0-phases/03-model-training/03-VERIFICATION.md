---
phase: 03-model-training
verified: 2026-03-13T00:00:00Z
status: human_needed
score: 11/12 must-haves verified
human_verification:
  - test: "Run 'python run.py train' with OPENROUTER_API_KEY set; allow at least one epoch to complete"
    expected: "Checkpoint file appears at checkpoints/<run_id>/epoch_0.pt; training log shows two distinct LRs (backbone ~2e-5, head ~1e-4) and warmup+decay schedule"
    why_human: "Actual checkpoint-on-disk requires a live training run with LLM data generation (API key needed). Cannot verify programmatically without credentials."
---

# Phase 3: Model + Training — Verification Report

**Phase Goal:** A trained gbert-large token classifier checkpoint exists on disk, produced by a training loop that uses differential learning rates, warmup, and mixed precision, with CRF and ensemble available as config toggles
**Verified:** 2026-03-13
**Status:** human_needed (all automated checks pass; one runtime item requires human verification)
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | RegulatoryNERModel produces logits of shape (batch, seq_len, 3) for O/B-REF/I-REF | VERIFIED | `test_forward_shape` passes; logits asserted as (2, 16, 3) |
| 2 | Setting use_crf=true switches loss to CRF negative log-likelihood | VERIFIED | `test_crf_toggle_with_labels` passes; CRF path uses `torchcrf.CRF(batch_first=True)` and `-self.crf(...)` |
| 3 | Setting freeze_backbone=true makes all BERT encoder params non-trainable | VERIFIED | `test_freeze_toggle` passes; all `bert.parameters()` have `requires_grad=False` |
| 4 | Setting use_lora=true makes only LoRA adapter + head params trainable | VERIFIED | `test_lora_toggle` passes; `trainable_count < total_count` asserted |
| 5 | Model forward pass works on CPU (both CRF and non-CRF) | VERIFIED | `test_cpu_forward_non_crf` and `test_cpu_forward_crf` pass |
| 6 | Optimizer has two parameter groups with distinct learning rates (backbone < head) | VERIFIED | `test_differential_lr` passes; groups 0 (2e-5) and 1 (1e-4) asserted |
| 7 | Mixed precision resolves correctly per device (fp16/bf16/no) | VERIFIED | `test_cuda_returns_config_value`, `test_cpu_returns_no`, `test_mps_new_torch_returns_bf16`, `test_mps_old_torch_returns_no` all pass |
| 8 | Gradient clipping is applied after backward pass with configurable max_norm | VERIFIED | `test_gradient_clipping` passes; norm bounded to `max_grad_norm + 1e-5` |
| 9 | LR schedule warms up linearly then decays linearly to zero | VERIFIED | `test_lr_schedule_warmup_then_decay` passes; LR at step 0 < 1e-8, peak >= 80% max, end < 10% max |
| 10 | Ensemble mode trains N models; first writes cache, rest read from cache | VERIFIED | `test_ensemble_n_checkpoints` (3 paths returned) and `test_ensemble_cache_handoff` (second Trainer gets cache path) pass |
| 11 | Ensemble majority vote returns correct BIO labels from multiple model predictions | VERIFIED | `test_majority_vote_basic` confirms `[[0,1,2,1],[0,1,1,1],[0,2,2,1]] → [0,1,2,1]` |
| 12 | A trained checkpoint file exists on disk after running `python run.py train` | HUMAN NEEDED | Training loop is fully wired and tested; actual disk artifact requires live run with OPENROUTER_API_KEY |

**Score:** 11/12 truths verified automatically; 1 requires human confirmation

---

## Required Artifacts

| Artifact | Min Lines | Actual Lines | Status | Notes |
|----------|-----------|-------------|--------|-------|
| `src/model/__init__.py` | — | 5 | VERIFIED | Exports `RegulatoryNERModel` |
| `src/model/ner_model.py` | 80 | 195 | VERIFIED | CRF toggle, LoRA toggle, freeze toggle, `get_bert_parameters()`, `get_head_parameters()` |
| `tests/test_ner_model.py` | 60 | 270 | VERIFIED | 9 tests; all pass |
| `requirements.txt` | — | 16 | VERIFIED | Contains `accelerate>=0.27.0`, `pytorch-crf==0.7.2`, `peft>=0.10.0` |
| `src/model/trainer.py` | 150 | 399 | VERIFIED | Trainer, build_optimizer, build_scheduler, save/load_checkpoint, majority_vote, train_ensemble |
| `tests/test_trainer.py` | 120 | 652 | VERIFIED | 19 tests; all pass |
| `run.py` | 20 | 106 | VERIFIED | train/evaluate/predict subcommands; evaluate and predict are correct Phase 4 stubs |
| `config/default.yaml` | — | 43 | VERIFIED | `ensemble.use_gradient_boost: false` present |

---

## Key Link Verification

| From | To | Via | Status | Evidence |
|------|----|-----|--------|---------|
| `src/model/ner_model.py` | `transformers.BertModel / BertForTokenClassification` | `from_pretrained` in `__init__` | WIRED | Line 56: `BertModel.from_pretrained(model_name)`, line 63: `BertForTokenClassification.from_pretrained(model_name, num_labels=...)` |
| `src/model/ner_model.py` | `torchcrf.CRF` | conditional CRF init when `use_crf=True` | WIRED | Line 59: `CRF(self.NUM_LABELS, batch_first=True)` |
| `src/model/ner_model.py` | `peft.get_peft_model` | `_apply_lora` method when `use_lora=True` | WIRED | Line 97/99: `get_peft_model(self.bert, lora_config)` / `get_peft_model(self.bert_tc, lora_config)` |
| `src/model/trainer.py` | `src/model/ner_model.py` | imports `RegulatoryNERModel` | WIRED | Line 26: `from src.model.ner_model import RegulatoryNERModel` |
| `src/model/trainer.py` | `accelerate.Accelerator` | `accelerator.prepare` | WIRED | Lines 250, 265: `self.accelerator.prepare(...)` called for model/optimizer/scheduler and dataloader |
| `src/model/trainer.py` | `src/data/dataset.py` | creates `LLMGeneratedDataset` per epoch | WIRED | Line 25 import + line 258: `LLMGeneratedDataset(config, self.tokenizer, epoch=epoch, cache_path=...)` |
| `run.py` | `src/model/trainer.py` | imports and calls train function | WIRED | Lines 70-75: `from src.model.trainer import Trainer, resolve_mixed_precision, train_ensemble` |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| MODL-01 | 03-01 | BIO token classifier on gbert-large with linear head | SATISFIED | `RegulatoryNERModel` with `NUM_LABELS=3`, uses `BertForTokenClassification` / `BertModel + nn.Linear` |
| MODL-02 | 03-01 | CRF layer via config toggle | SATISFIED | `use_crf` toggle wired; `torchcrf.CRF(batch_first=True)` conditional on `config.model.use_crf` |
| MODL-03 | 03-02 | Differential learning rates | SATISFIED | `build_optimizer` creates two param groups; group 0 backbone (2e-5), group 1 head (1e-4) |
| MODL-04 | 03-02 | Mixed precision with device detection | SATISFIED | `resolve_mixed_precision` returns correct value per device type; CUDA/MPS/CPU all handled |
| MODL-05 | 03-02 | Gradient clipping | SATISFIED | `accelerator.clip_grad_norm_` called every batch step; `max_grad_norm` from config |
| MODL-06 | 03-02 | Linear warmup + decay schedule | SATISFIED | `get_linear_schedule_with_warmup` used; LR trajectory tested and confirmed |
| MODL-07 | 03-01 | Freeze BERT or apply LoRA via config | SATISFIED | `freeze_backbone` and `use_lora` toggles both implemented and tested |
| MODL-08 | 03-01 | Runs on MPS, CUDA, and CPU | SATISFIED | CPU path tested in unit tests; MPS/CUDA paths handled via Accelerate device abstraction |
| ENSM-01 | 03-02 | Bagging ensemble via config | SATISFIED | `ensemble.enabled` + `n_estimators` wired; `train_ensemble` loops over N models |
| ENSM-02 | 03-02 | First model writes cache; others resample from it | SATISFIED | `train_ensemble`: model 0 `cache_path=None`, models 1..N-1 `cache_path=ensemble_base.jsonl` |
| ENSM-03 | 03-02 | Majority vote over BIO predictions | SATISFIED | `majority_vote` uses `Counter.most_common(1)` per position; tested with known inputs |
| ENSM-04 | 03-02 | Gradient-boost variant as config-gated stub | SATISFIED | `use_gradient_boost: false` in config; `train_ensemble` logs warning if enabled; does not error |

All 12 Phase 3 requirements satisfied. No orphaned requirements — all 12 IDs declared in plan frontmatter appear in REQUIREMENTS.md traceability table as Phase 3 / Complete.

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `run.py` | 59 | `"Not yet implemented — see Phase 4"` for evaluate/predict | INFO | Expected and correct — these subcommands are intentionally deferred to Phase 4 per plan spec |

No blockers or warnings. The Phase 4 stubs in `run.py` are intentional and correctly scoped.

---

## Human Verification Required

### 1. End-to-End Training Produces Checkpoint on Disk

**Test:** Set `OPENROUTER_API_KEY`, run `python run.py train` from `/Users/Admin/REG_ML/`, allow at least one epoch to complete.

**Expected:**
- Training starts, log line shows `"Epoch 1/3 — avg loss: X.XXXX"`
- Checkpoint file appears at `checkpoints/epoch_0.pt`
- Log or print output shows two distinct LRs (backbone ~2e-5, head ~1e-4)

**Why human:** The checkpoint-on-disk component of the phase goal is a runtime artifact. It requires a live LLM API call to generate training data (`LLMGeneratedDataset` calls OpenRouter). All code that writes the checkpoint (`save_checkpoint`, `torch.save`) is fully implemented and wired — the infrastructure is sound — but the disk artifact itself cannot be produced without credentials.

### 2. CRF Training Path End-to-End

**Test:** Set `model.use_crf: true` in `config/default.yaml`, run `python run.py train` for one epoch.

**Expected:** Training completes without CRF mask errors; checkpoint saved at `checkpoints/epoch_0.pt`.

**Why human:** CRF path unit tests use mocked tiny BERT to avoid downloads. The real gbert-large constraint (first token must not be -100 in CRF batch) needs end-to-end verification with actual tokenized data from Phase 2.

### 3. Ensemble Training Cache Handoff

**Test:** Set `ensemble.enabled: true`, `ensemble.n_estimators: 2`, run `python run.py train`.

**Expected:** After training, `data/cache/ensemble_base.jsonl` exists (written by model 0); second model reads from it (no new LLM calls for model 1).

**Why human:** The cache handoff logic is tested via mocks. Real file I/O and LLM call elimination for model 1 requires a live run to confirm.

---

## Gaps Summary

No gaps found. All automated must-haves pass. The single human-verification item (checkpoint on disk) is a runtime artifact by design — the training infrastructure is fully implemented, wired, and tested across 83 tests (9 model + 19 trainer + 55 Phase 1-2). The code path from `python run.py train` to `checkpoints/epoch_N.pt` is verified to be fully connected at the code level.

---

**Commits verified:**
- `f288dd2` — test(03-01): RED phase, failing NER model tests
- `bd99053` — feat(03-01): GREEN phase, RegulatoryNERModel implementation
- `3863c8b` — test(03-02): RED phase, failing Trainer/ensemble/run.py tests
- `84bab8c` — feat(03-02): GREEN phase, Trainer + ensemble + run.py

---

_Verified: 2026-03-13_
_Verifier: Claude (gsd-verifier)_
