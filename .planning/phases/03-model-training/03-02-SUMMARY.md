---
phase: 03-model-training
plan: "02"
subsystem: training
tags: [trainer, accelerate, adamw, ensemble, bagging, majority-vote, cli, mixed-precision, gradient-clipping]
dependency_graph:
  requires:
    - src/model/ner_model.py (RegulatoryNERModel, get_bert_parameters, get_head_parameters)
    - src/data/dataset.py (LLMGeneratedDataset)
    - src/utils/config.py (load_config)
    - src/utils/device.py (get_device, set_seed)
    - config/default.yaml (training.*, ensemble.*, data.*)
    - accelerate>=0.27.0 (Accelerator, clip_grad_norm_, backward, prepare, autocast)
    - transformers (get_linear_schedule_with_warmup, BertTokenizerFast)
  provides:
    - src/model/trainer.py (Trainer, train_ensemble, majority_vote, resolve_mixed_precision, build_optimizer, build_scheduler, save_checkpoint, load_checkpoint)
    - run.py (CLI entry point: train/evaluate/predict subcommands)
  affects:
    - Phase 4 evaluation (loads checkpoints from CHECKPOINT_BASE/run_id/epoch_N.pt)
    - Phase 4 inference (imports Trainer, majority_vote for ensemble prediction)
tech_stack:
  added:
    - accelerate (Accelerator.prepare, backward, clip_grad_norm_, autocast, unwrap_model)
    - transformers.get_linear_schedule_with_warmup (linear warmup+decay LambdaLR)
    - torch.optim.AdamW (two param groups with differential LR)
    - argparse (run.py CLI subcommands)
  patterns:
    - Differential LR: backbone param group (lr=2e-5) + head param group (lr=1e-4)
    - resolve_mixed_precision: CUDA returns config value; MPS needs torch>=2.6 for bf16; CPU always "no"
    - Build optimizer BEFORE accelerator.prepare — pass model+optimizer+scheduler together
    - Recreate LLMGeneratedDataset each epoch with epoch= arg for fresh seeds (Pitfall 5)
    - Ensemble bagging: model 0 generates live (no cache), models 1..N read ensemble_base.jsonl
    - majority_vote: collections.Counter.most_common(1) per position, deterministic tie-breaking
    - CHECKPOINT_BASE module constant patchable in tests to redirect checkpoint writes to tmp_path
key_files:
  created:
    - src/model/trainer.py
    - run.py
    - tests/test_trainer.py
  modified:
    - config/default.yaml (added ensemble.use_gradient_boost: false)
decisions:
  - "resolve_mixed_precision: CUDA returns config value; MPS requires torch>=2.6 for bf16; CPU always 'no' — avoids silent dtype errors on Apple Silicon"
  - "Build optimizer before accelerator.prepare — passing all three together avoids prepare overwriting initial LRs"
  - "CHECKPOINT_BASE as module-level constant allows tests to redirect checkpoint writes via patch() without touching config"
  - "Ensemble model 0 writes live-generated data to ensemble_base.jsonl; models 1..N read from it — minimises LLM API calls for N-1 estimators"
  - "use_gradient_boost config key added as false stub — satisfies ENSM-04 without premature implementation"
metrics:
  duration: "6 min"
  completed: "2026-03-13"
  tasks_completed: 2
  files_created: 3
  files_modified: 1
  tests_added: 19
  tests_total: 83
---

# Phase 3 Plan 2: Trainer, Ensemble, and CLI Summary

**One-liner:** AdamW with differential LR, linear warmup+decay, Accelerate-wrapped training loop, bagging ensemble with cache handoff, majority vote aggregation, and run.py CLI — 19 TDD tests, 83 total passing.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | Failing tests for Trainer, ensemble, run.py | 3863c8b | tests/test_trainer.py |
| 1+2 (GREEN) | Trainer implementation + ensemble driver + run.py | 84bab8c | src/model/trainer.py, run.py, config/default.yaml, tests/test_trainer.py (bug fix) |

## What Was Built

### `src/model/trainer.py`

**`resolve_mixed_precision(config, device) -> str`**
Returns `config.training.mixed_precision` for CUDA, `"bf16"` for MPS on torch >= 2.6, `"no"` for MPS on older torch and CPU. Parses `torch.__version__` string for MPS branch.

**`build_optimizer(model, config) -> AdamW`**
Two param groups: backbone parameters (`model.get_bert_parameters()`, `lr=learning_rate_backbone`) and head parameters (`model.get_head_parameters()`, `lr=learning_rate_head`). Filters to `requires_grad=True` only. `weight_decay=0.01`.

**`build_scheduler(optimizer, config, steps_per_epoch) -> LambdaLR`**
`get_linear_schedule_with_warmup` with `num_training_steps = num_epochs * steps_per_epoch`. LR ramps from 0 to peak at `warmup_steps`, then decays linearly to 0.

**`save_checkpoint / load_checkpoint`**
`save_checkpoint` unwraps model via `accelerator.unwrap_model`, writes to `CHECKPOINT_BASE/{run_id}/epoch_{epoch}.pt` with keys: epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict. `load_checkpoint` restores all fields and returns the stored epoch number.

**`class Trainer`**
Main training loop:
1. Build optimizer before `accelerator.prepare` (all 3 together)
2. Each epoch: create fresh `LLMGeneratedDataset(config, tokenizer, epoch=epoch)` — avoids stale IterableDataset
3. For each batch: `accelerator.autocast()` context, forward, `accelerator.backward(loss)`, `accelerator.clip_grad_norm_()`, step, zero_grad
4. Save checkpoint after each epoch; return final checkpoint path

**`majority_vote(predictions) -> list[int]`**
`collections.Counter.most_common(1)` per position over N model predictions. Deterministic tie-breaking.

**`train_ensemble(config, tokenizer, accelerator) -> list[Path]`**
Bagging loop over `n_estimators`:
- Model 0: `cache_path=None` — LLM generates live data; writes to `ensemble_base.jsonl`
- Models 1..N: `cache_path = {cache_dir}/ensemble_base.jsonl` — reads from first model's cache
- Per-estimator seed offset: `seed + i * 1000`
- `use_gradient_boost=True` logs warning "gradient-boost ensemble not yet implemented, falling back to uniform resampling" (ENSM-04 stub)

### `run.py`

CLI with `train`, `evaluate`, `predict` subcommands. Only `train` is implemented:
1. `load_config()` with optional CLI overrides
2. `set_seed`, `get_device`, `resolve_mixed_precision`, `Accelerator(mixed_precision=...)`
3. `BertTokenizerFast.from_pretrained(config.model.name)`
4. If `ensemble.enabled`: `train_ensemble(...)` else `Trainer(...).train()`
5. Prints checkpoint path(s)

`evaluate` and `predict` print "Not yet implemented — see Phase 4" and exit cleanly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `importlib.util.load_from_spec` does not exist in Python 3.14**

- **Found during:** Task 2 GREEN phase, first test run of `TestRunPyTrainSubcommand`
- **Issue:** Test used `importlib.util.load_from_spec(spec)` which is not a real Python API. The correct method is `importlib.util.module_from_spec(spec)` followed by `spec.loader.exec_module(module)`.
- **Fix:** Updated `tests/test_trainer.py` to use `module_from_spec` and `exec_module`.
- **Files modified:** `tests/test_trainer.py`
- **Commit:** 84bab8c (included in GREEN commit)

## Verification Results

```
pytest tests/test_trainer.py -x -v
19 passed in 3.84s

pytest tests/ -x -q
83 passed in 7.14s

python -c "from src.model.trainer import Trainer, train_ensemble, majority_vote; print('OK')"
OK

python run.py --help
usage: run.py [-h] {train,evaluate,predict} ...
```

## Self-Check: PASSED

- [x] `src/model/trainer.py` — exists, 280+ lines (> 150 minimum)
- [x] `tests/test_trainer.py` — exists, 650+ lines (> 120 minimum)
- [x] `run.py` — exists, 110+ lines (> 20 minimum)
- [x] `config/default.yaml` — contains `use_gradient_boost: false`
- [x] Commit 3863c8b (RED) — exists
- [x] Commit 84bab8c (GREEN) — exists
- [x] All 19 trainer tests pass
- [x] Full suite 83/83 passes
- [x] `from src.model.trainer import Trainer, train_ensemble, majority_vote` — importable
- [x] `python run.py --help` — runs without error
- [x] Optimizer has 2 param groups with distinct LRs (backbone < head)
- [x] Mixed precision resolves correctly per device
- [x] Gradient clipping applied after backward with configurable max_norm
- [x] LR schedule warms up linearly then decays linearly to zero
- [x] Ensemble mode trains N models; model 0 writes cache; rest read from it
- [x] Majority vote returns correct BIO labels from multiple model predictions
