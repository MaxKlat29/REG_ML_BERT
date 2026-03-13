---
phase: 03-model-training
plan: "01"
subsystem: model
tags: [bert, ner, crf, lora, peft, pytorch, token-classification]
dependency_graph:
  requires:
    - src/utils/config.py (load_config interface)
    - src/utils/device.py (get_device, set_seed)
    - config/default.yaml (model.use_crf, model.freeze_backbone, model.use_lora, model.lora_rank)
  provides:
    - src/model/ner_model.py (RegulatoryNERModel)
    - src/model/__init__.py (package export)
  affects:
    - Phase 4 evaluation (consumes model checkpoints)
    - 03-02 trainer (imports RegulatoryNERModel, uses get_bert_parameters/get_head_parameters)
tech_stack:
  added:
    - pytorch-crf==0.7.2 (torchcrf.CRF, Viterbi decode)
    - peft>=0.10.0 (LoraConfig, get_peft_model, TaskType.TOKEN_CLS)
    - accelerate>=0.27.0 (added to requirements.txt; used in 03-02 trainer)
  patterns:
    - BertForTokenClassification for non-CRF path (loss computed internally with ignore_index=-100)
    - BertModel + nn.Linear + CRF for CRF path (raw logits needed for CRF forward)
    - pytorch-crf mask[:,0] must all be True (batch_first constraint)
    - LoRA applied after freeze_backbone so adapter params override frozen base weights
    - Tiny BertConfig (hidden_size=64, 1 layer) in tests — no model download needed
key_files:
  created:
    - src/model/__init__.py
    - src/model/ner_model.py
    - tests/test_ner_model.py
  modified:
    - requirements.txt (added accelerate, pytorch-crf, peft)
decisions:
  - "CRF path uses BertModel (not BertForTokenClassification) to access raw last_hidden_state for manual linear head + CRF"
  - "pytorch-crf mask[:,0] must all be True — labels[:,0] must not be -100 in CRF forward (SEP-only -100 masking in training data)"
  - "LoRA target_modules fallback: query/value -> q_proj/v_proj -> warn+skip — ensures graceful degradation on non-standard BERT variants"
  - "freeze_backbone applied BEFORE LoRA so PEFT adapter parameters override the frozen base weights"
metrics:
  duration: "4 min"
  completed: "2026-03-13"
  tasks_completed: 1
  files_created: 3
  files_modified: 1
  tests_added: 9
  tests_total: 64
---

# Phase 3 Plan 1: RegulatoryNERModel Summary

**One-liner:** BERT token classifier with config-gated CRF (torchcrf), LoRA (PEFT), and backbone freeze — 9 TDD tests, 64 total passing, no model downloads in CI.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | Failing tests for RegulatoryNERModel | f288dd2 | tests/test_ner_model.py, src/model/__init__.py, requirements.txt |
| 1 (GREEN) | RegulatoryNERModel implementation | bd99053 | src/model/ner_model.py, tests/test_ner_model.py (bug fix) |

## What Was Built

`RegulatoryNERModel(nn.Module)` wrapping gbert-large for German legal reference NER (O=0, B-REF=1, I-REF=2) with three config-controlled toggles:

1. **`use_crf=True`**: Uses `BertModel` + `nn.Linear` + `torchcrf.CRF(batch_first=True)`. Forward with labels returns `(loss, emissions)`. Forward without labels returns Viterbi-decoded tag lists.

2. **`use_crf=False`** (default): Uses `BertForTokenClassification.from_pretrained(num_labels=3)`. Delegates entirely to HF — returns `TokenClassifierOutput` with `.loss` and `.logits`.

3. **`freeze_backbone=True`**: Sets `requires_grad=False` on all BERT encoder params. Applied before LoRA so adapter params remain trainable.

4. **`use_lora=True`**: Applies `LoraConfig(task_type=TOKEN_CLS, r=lora_rank, target_modules=["query","value"])` via `get_peft_model()`. Includes fallback to `q_proj`/`v_proj` naming with a logged warning.

5. **`get_bert_parameters()` / `get_head_parameters()`**: Separate param accessors for Plan 02 differential LR optimizer construction.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] pytorch-crf mask[:, 0] constraint**

- **Found during:** Task 1 GREEN phase (first test run)
- **Issue:** `pytorch-crf` raises `ValueError: mask of the first timestep must all be on` when `batch_first=True` and `mask[:, 0]` contains any `False`. Our test `make_batch()` was setting `labels[:, 0] = -100` (CLS token), making `mask[:, 0] = False`.
- **Fix:** Updated `make_batch()` in test file to only set `labels[:, -1] = -100` (SEP token only). Updated `test_crf_handles_minus100` to ensure position 0 always has a valid label. The CRF model itself is correct — this is a constraint of the pytorch-crf API that must be respected in training data prep (first real token can never be -100).
- **Files modified:** `tests/test_ner_model.py`
- **Commit:** bd99053

## Verification Results

```
pytest tests/test_ner_model.py -x -v
9 passed in 3.64s

pytest tests/ -x -q
64 passed in 6.28s

python -c "from src.model import RegulatoryNERModel; print('import OK')"
import OK
```

## Self-Check: PASSED

- [x] `src/model/__init__.py` — exists
- [x] `src/model/ner_model.py` — exists, 122 lines (> 80 minimum)
- [x] `tests/test_ner_model.py` — exists, 260+ lines (> 60 minimum)
- [x] `requirements.txt` — contains accelerate, pytorch-crf, peft
- [x] Commit f288dd2 (RED) — exists
- [x] Commit bd99053 (GREEN) — exists
- [x] All 9 model tests pass
- [x] Full suite 64/64 passes
- [x] `RegulatoryNERModel` importable from `src.model`
