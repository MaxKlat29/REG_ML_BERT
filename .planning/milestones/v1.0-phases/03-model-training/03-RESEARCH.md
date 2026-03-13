# Phase 3: Model + Training - Research

**Researched:** 2026-03-13
**Domain:** PyTorch NER token classifier (BERT + CRF + LoRA + Accelerate)
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MODL-01 | BIO token classifier (O, B-REF, I-REF) on gbert-large with linear head | `AutoModelForTokenClassification` with `num_labels=3`; logits from BERT hidden states → linear layer |
| MODL-02 | Optional CRF layer via config toggle to enforce valid BIO transitions | `pytorch-crf` (`CRF` class); forward returns log-likelihood, decode uses Viterbi; mask built from `labels != -100` |
| MODL-03 | Differential LRs: lower for BERT encoder, higher for classification head | AdamW param groups: `model.bert.parameters()` at `lr_backbone`, `model.classifier.parameters()` at `lr_head` |
| MODL-04 | Mixed precision: fp16 CUDA, disabled/bf16 MPS, automatic device detection | `Accelerator(mixed_precision=...)` — pass `"fp16"` on CUDA, `"bf16"` on MPS (requires torch>=2.6), `"no"` as safe fallback |
| MODL-05 | Gradient clipping | `accelerator.clip_grad_norm_(model.parameters(), max_norm)` after `accelerator.backward(loss)` |
| MODL-06 | Linear warmup + linear decay LR schedule | `get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)` from `transformers.optimization` |
| MODL-07 | Freeze BERT or apply LoRA via config toggle | Freeze: `for p in model.bert.parameters(): p.requires_grad = False`; LoRA: PEFT `LoraConfig(task_type=TaskType.TOKEN_CLS, ...)` + `get_peft_model()` |
| MODL-08 | Runs on MPS / CUDA / CPU with automatic detection | Existing `get_device()` in `src/utils/device.py`; Accelerator wraps device routing |
| ENSM-01 | Bagging ensemble via config (n_estimators, bootstrap resampling) | Outer loop: train N models; each with fresh random seed offset; resample from cache using random index selection |
| ENSM-02 | First model writes cache; subsequent models resample from it | `LLMGeneratedDataset` already supports `cache_path` — first model trains live + writes cache; subsequent pass `cache_path` |
| ENSM-03 | Ensemble inference: majority vote over BIO predictions | Collect per-token label predictions from N models; `torch.mode()` or `Counter` per position |
| ENSM-04 | Optional gradient-boost variant with error-weighted retraining | Per-sample loss weighting: compute per-sample loss on val set after model N, weight cache resampling inversely to accuracy |
</phase_requirements>

---

## Summary

Phase 3 builds on the fully operational data pipeline from Phase 2 to produce a trained checkpoint. The model is `deepset/gbert-large` wrapped in `BertForTokenClassification` (or a custom subclass when CRF/LoRA are enabled) with 3 output labels (O=0, B-REF=1, I-REF=2). The existing config already declares all required hyperparameters (`learning_rate_backbone`, `learning_rate_head`, `warmup_steps`, `mixed_precision`, `max_grad_norm`, `use_crf`, `use_lora`, `freeze_backbone`, `ensemble.*`), so Phase 3 is a pure implementation phase — no config schema changes needed.

The training loop uses HuggingFace Accelerate for device-portable mixed precision. Two AdamW parameter groups deliver differential learning rates. `get_linear_schedule_with_warmup` from `transformers.optimization` handles the warmup + linear decay schedule. The CRF toggle swaps the loss function: standard `CrossEntropyLoss` (ignoring -100 positions) vs. `pytorch-crf`'s negative log-likelihood (masking -100 positions). LoRA is applied via PEFT's `get_peft_model()` and is orthogonal to CRF. Ensemble training is an outer loop over the single-model trainer.

**Primary recommendation:** Build `RegulatoryNERModel` as a thin `nn.Module` wrapping `BertForTokenClassification` (or `BertModel` for CRF variant), then build `Trainer` as a plain Python class (not HuggingFace Trainer) to keep full control over the training loop while routing all device/precision ops through Accelerate.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `transformers` | >=4.40 (already in requirements) | gbert-large model, BertTokenizerFast, LR schedules | HuggingFace official; already a project dependency |
| `torch` | >=2.0 (already in requirements) | Tensors, DataLoader, autograd | Project base |
| `accelerate` | >=0.27 | Mixed precision, device routing, gradient management | Official HF library; 5-line integration into any loop |
| `pytorch-crf` | 0.7.2 | CRF layer with Viterbi decode | Only maintained PyPI CRF package; `torchcrf` is an alias |
| `peft` | >=0.10 | LoRA parameter-efficient fine-tuning | Official HF PEFT library; `TaskType.TOKEN_CLS` supported |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `torch.optim.AdamW` | (stdlib torch) | Optimizer with weight decay | Always — standard for transformer fine-tuning |
| `transformers.get_linear_schedule_with_warmup` | (part of transformers) | LR warmup + linear decay | Always — already matches config `warmup_steps` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `accelerate` | Manual `torch.autocast` + device logic | accelerate wraps it cleanly; MPS quirks are handled upstream |
| `pytorch-crf` | Hand-rolled CRF | pytorch-crf is well-tested; BIO constraints are non-trivial to implement correctly |
| `peft` LoRA | Manual weight injection | PEFT has `TaskType.TOKEN_CLS`, tested with BERT; saves 100+ lines |

**Installation:**
```bash
pip install accelerate pytorch-crf peft
```

(Add to `requirements.txt` — these are not present yet.)

---

## Architecture Patterns

### Recommended Project Structure

```
src/
├── data/          # Existing — LLMGeneratedDataset, BIO converter, cache
├── evaluation/    # Existing — metrics, regex baseline
├── utils/         # Existing — config, device
└── model/         # NEW this phase
    ├── __init__.py
    ├── ner_model.py      # RegulatoryNERModel (BERT + head + optional CRF/LoRA)
    └── trainer.py        # Trainer class (train loop, ensemble driver)
run.py                    # NEW — CLI entry point for train/evaluate/predict
```

### Pattern 1: Model Architecture — BertForTokenClassification vs Custom Module

**What:** For the non-CRF path, use `BertForTokenClassification` directly (BERT + linear head, loss computed internally). For the CRF path, use `BertModel` + manual `nn.Linear` head + `CRF` layer.

**Why split:** `BertForTokenClassification.forward()` computes `CrossEntropyLoss` internally with `ignore_index=-100`. When CRF is active, we need raw logits (not loss), so we must use `BertModel` and attach our own head.

**Pattern:**
```python
# Source: HuggingFace transformers docs + pytorch-crf docs
from transformers import BertModel, BertForTokenClassification
from torchcrf import CRF
import torch.nn as nn

class RegulatoryNERModel(nn.Module):
    NUM_LABELS = 3  # O=0, B-REF=1, I-REF=2

    def __init__(self, config):
        super().__init__()
        model_name = config.model.name  # "deepset/gbert-large"

        if config.model.use_crf:
            self.bert = BertModel.from_pretrained(model_name)
            hidden_size = self.bert.config.hidden_size  # 1024 for large
            self.classifier = nn.Linear(hidden_size, self.NUM_LABELS)
            self.crf = CRF(self.NUM_LABELS, batch_first=True)
            self._use_crf = True
        else:
            self.bert_for_tc = BertForTokenClassification.from_pretrained(
                model_name, num_labels=self.NUM_LABELS
            )
            self._use_crf = False

    def forward(self, input_ids, attention_mask, labels=None):
        if self._use_crf:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            emissions = self.classifier(outputs.last_hidden_state)
            # CRF mask: True where labels != -100 (real tokens)
            mask = (labels != -100).bool() if labels is not None else attention_mask.bool()
            # Replace -100 with 0 for CRF (masked positions are ignored)
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0
            if labels is not None:
                loss = -self.crf(emissions, crf_labels, mask=mask, reduction="mean")
                return loss, emissions
            return self.crf.decode(emissions, mask=mask)
        else:
            return self.bert_for_tc(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
```

### Pattern 2: Differential Learning Rate — Two AdamW Parameter Groups

**What:** Separate parameter groups in AdamW for BERT encoder (low LR) vs. classification head (high LR). The LR scheduler applies the same warmup/decay multiplier to both groups, preserving the ratio.

**Example:**
```python
# Source: HuggingFace transformers training docs
from torch.optim import AdamW

def build_optimizer(model, cfg):
    if hasattr(model, 'bert_for_tc'):
        # Non-CRF path: BertForTokenClassification
        bert_params = list(model.bert_for_tc.bert.parameters())
        head_params = list(model.bert_for_tc.classifier.parameters())
    else:
        # CRF path: separate bert + classifier + crf
        bert_params = list(model.bert.parameters())
        head_params = list(model.classifier.parameters()) + list(model.crf.parameters())

    param_groups = [
        {"params": bert_params, "lr": cfg.training.learning_rate_backbone},
        {"params": head_params, "lr": cfg.training.learning_rate_head},
    ]
    return AdamW(param_groups, weight_decay=0.01)
```

### Pattern 3: Accelerate Integration for Mixed Precision + Device Routing

**What:** Accelerate wraps the model, optimizer, scheduler, and dataloader. Mixed precision (fp16/bf16/no) is specified at init. `accelerator.backward(loss)` handles gradient scaling. `accelerator.clip_grad_norm_` handles clipping.

**MPS-specific:** bf16 on MPS requires PyTorch >= 2.6. Since the project already requires torch>=2.0, the safe default is `"no"` for MPS. The config already has `mixed_precision: "bf16"` as default — trainer must override to `"no"` when on MPS and torch < 2.6.

**Example:**
```python
# Source: HuggingFace Accelerate docs
from accelerate import Accelerator
import torch

def resolve_mixed_precision(cfg, device: torch.device) -> str:
    """Determine safe mixed_precision setting for the detected device."""
    requested = cfg.training.mixed_precision  # "fp16", "bf16", or "no"
    if device.type == "mps":
        # bf16 on MPS needs torch >= 2.6; fp16 needs >= 2.8
        major, minor = [int(x) for x in torch.__version__.split(".")[:2]]
        if requested == "bf16" and (major, minor) >= (2, 6):
            return "bf16"
        return "no"  # safe fallback
    return requested  # CUDA handles fp16/bf16 natively

accelerator = Accelerator(mixed_precision=resolve_mixed_precision(cfg, device))
model, optimizer, dataloader, scheduler = accelerator.prepare(
    model, optimizer, dataloader, scheduler
)

# Training loop
for batch in dataloader:
    with accelerator.autocast():
        loss, _ = model(**batch)
    accelerator.backward(loss)
    accelerator.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

### Pattern 4: LR Scheduler — Linear Warmup + Linear Decay

**What:** `get_linear_schedule_with_warmup` from `transformers.optimization`. Increases LR linearly from 0 → initial_lr over `warmup_steps`, then decreases linearly from initial_lr → 0 over remaining training steps.

**Example:**
```python
# Source: transformers.optimization docs (verified)
from transformers import get_linear_schedule_with_warmup

num_training_steps = num_epochs * steps_per_epoch
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=cfg.training.warmup_steps,
    num_training_steps=num_training_steps,
)
```

**Note:** Because AdamW has two param groups with different LRs, the scheduler multiplies each group's LR by the same factor. The ratio between backbone and head LR is preserved throughout training.

### Pattern 5: LoRA Configuration for TOKEN_CLS

**What:** PEFT `LoraConfig` with `task_type=TaskType.TOKEN_CLS`. Targets `query` and `value` attention matrices in BERT. Applied after model creation, before optimizer construction.

**Example:**
```python
# Source: HuggingFace PEFT docs (token classification guide)
from peft import get_peft_model, LoraConfig, TaskType

def apply_lora(model, cfg):
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=cfg.model.lora_rank,          # default 16
        lora_alpha=cfg.model.lora_rank * 2,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
    )
    return get_peft_model(model, lora_config)
```

**Important:** LoRA and CRF are orthogonal. LoRA can be applied to the BERT encoder in both CRF and non-CRF paths. Apply LoRA before wrapping with Accelerate.

### Pattern 6: Ensemble Driver — Bagging over Single-Model Trainer

**What:** The ensemble is not a separate architecture — it's an outer loop that calls the single-model trainer N times, varying the seed and cache path.

**Cache protocol (ENSM-02):**
- Model 0: `cache_path=None` → generates live via LLM → writes to `data/cache/ensemble_0.jsonl`
- Model 1..N-1: `cache_path="data/cache/ensemble_0.jsonl"` → reads from cache, resamples with random seed

**Example:**
```python
def train_ensemble(cfg, tokenizer):
    cache_path = Path(cfg.data.cache_dir) / "ensemble_base.jsonl"
    checkpoints = []
    for i in range(cfg.ensemble.n_estimators):
        seed = cfg.project.seed + i * 1000
        # First model generates + caches; rest read from cache
        train_cache = None if i == 0 else str(cache_path)
        write_cache = str(cache_path) if i == 0 else None
        ckpt = train_single_model(cfg, tokenizer, seed, train_cache, write_cache)
        checkpoints.append(ckpt)
    return checkpoints
```

### Pattern 7: Checkpoint Saving

**What:** Save full training state dict (model weights, optimizer, scheduler, epoch) using `torch.save`. Use HuggingFace `save_pretrained` for portability.

**Example:**
```python
import torch
from pathlib import Path

def save_checkpoint(model, optimizer, scheduler, epoch, cfg, run_id: str = ""):
    ckpt_dir = Path("checkpoints") / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Unwrap from PEFT/Accelerate for clean save
    raw_model = accelerator.unwrap_model(model)
    torch.save({
        "epoch": epoch,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": dict(cfg),
    }, ckpt_dir / f"epoch_{epoch}.pt")
```

### Anti-Patterns to Avoid

- **Using HuggingFace `Trainer` class:** It adds complexity for CRF and ensemble scenarios; full control is preferable here.
- **Calling `loss.backward()` directly:** Always use `accelerator.backward(loss)` — direct backward skips gradient scaling for mixed precision.
- **`model.to(device)` after `accelerator.prepare()`:** Accelerate handles device placement; manual `.to()` afterward breaks MPS/CUDA logic.
- **Passing -100 labels directly to CRF:** CRF requires valid tag indices; mask must be built from `labels != -100` before converting -100 → 0 in the label tensor.
- **Building optimizer after `accelerator.prepare()`:** Build optimizer with param groups first, then pass both model and optimizer to `prepare()` together so Accelerate wraps the optimizer correctly.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Viterbi decoding for CRF | Custom Viterbi | `pytorch-crf` `CRF.decode()` | Viterbi has non-trivial edge cases with batched variable-length sequences |
| LR warmup + decay | Manual lambda scheduler | `get_linear_schedule_with_warmup` | Already in `transformers`, correct behavior with multiple param groups |
| Mixed precision scaling | `torch.cuda.amp.GradScaler()` directly | `accelerate` `Accelerator` | Accelerate handles MPS/CUDA/CPU differences and scaler lifecycle |
| LoRA weight injection | Manual rank decomposition | PEFT `get_peft_model()` | PEFT handles attention module targeting, merge/unmerge, and save/load |
| BIO transition constraints | Manual post-processing | CRF layer | CRF enforces valid transitions as a trained component, not a hard rule |

**Key insight:** The CRF layer is deceptively complex — the forward pass computes a log-partition function via dynamic programming, and the decode pass runs Viterbi. Both must handle variable-length sequences via masking. `pytorch-crf` (PyPI: `pytorch-crf`) is the only well-maintained PyTorch CRF package and is the correct choice.

---

## Common Pitfalls

### Pitfall 1: CRF Receives -100 Labels

**What goes wrong:** `pytorch-crf`'s `forward(emissions, tags, mask)` requires `tags` to be valid label indices (0..N-1). If -100 from the BIO converter leaks in, CRF raises `ValueError` or produces wrong loss.

**Why it happens:** The BIO converter sets special tokens to label -100 (correct for CrossEntropyLoss), but CRF does not use `ignore_index`.

**How to avoid:** Always build a boolean mask `mask = (labels != -100)` BEFORE modifying the labels tensor. Then do `labels = labels.clone(); labels[labels == -100] = 0`. The mask ensures position 0 substitution never contributes to the CRF loss.

**Warning signs:** `IndexError: index out of range` or `AssertionError` from torchcrf during training.

### Pitfall 2: MPS Mixed Precision OOM or Silent Errors

**What goes wrong:** Using `mixed_precision="fp16"` on MPS with torch < 2.8 causes either silent dtype mismatches or OOM on large batches.

**Why it happens:** MPS fp16 support was only added in torch 2.8 (bf16 in 2.6). Older versions silently fall through to float32 or crash.

**How to avoid:** Use `resolve_mixed_precision()` helper that inspects `torch.__version__` and falls back to `"no"` for MPS if version requirements are not met. Config default `"bf16"` is appropriate for MPS on torch>=2.6.

**Warning signs:** `NotImplementedError: The operator 'aten::...' is not implemented for 'MPS'` during forward pass.

### Pitfall 3: LoRA + CRF Parameter Group Collision

**What goes wrong:** When both LoRA and CRF are enabled, `get_peft_model()` wraps the BERT module but the CRF and linear head remain outside PEFT. If you build optimizer param groups from the original model attributes before applying LoRA, `bert_params` will include non-trainable frozen base weights.

**Why it happens:** `get_peft_model()` replaces linear layers in BERT with LoRA adapters and freezes base weights. The model attribute references change.

**How to avoid:** Apply LoRA first, then build optimizer param groups using `model.parameters()` filtered by `requires_grad=True`. Separate backbone from head by name prefix, not by direct attribute reference.

**Warning signs:** `optimizer has param groups where no parameters require grad` warning, or head LR not appearing in logs.

### Pitfall 4: Scheduler Steps with Two Param Groups

**What goes wrong:** `get_linear_schedule_with_warmup` operates on the optimizer's learning rates. After warmup, both param groups decay linearly toward 0, preserving their ratio. This is correct. However, if you create the scheduler before `accelerator.prepare()`, it may not be properly wrapped as an `AcceleratedScheduler`.

**Why it happens:** Accelerate wraps the scheduler to synchronize step calls across distributed processes. Creating it outside `prepare()` skips this.

**How to avoid:** Pass all four objects (model, optimizer, dataloader, scheduler) together to `accelerator.prepare()`.

### Pitfall 5: IterableDataset Epoch Argument Drift

**What goes wrong:** `LLMGeneratedDataset` takes `epoch` as a constructor argument (not updated dynamically). If you reuse the same dataset object across epochs, all epochs generate identical data (same seeds).

**Why it happens:** The seed formula `epoch * 10000 + batch_idx * 100 + worker_id` needs the epoch number baked in at dataset creation.

**How to avoid:** Recreate the dataset (and DataLoader) at the start of each epoch, passing the current epoch number.

### Pitfall 6: BertTokenizerFast vs AutoTokenizer on gbert-large

**What goes wrong:** As documented in STATE.md (02-02 decision): `transformers` 5.x `AutoTokenizer` fails on gbert-large without `tokenizer.json`. Phase 3 must use `BertTokenizerFast.from_pretrained(model_name)` consistently.

**How to avoid:** Always use `BertTokenizerFast` for gbert-large. This is already established in Phase 2 code.

---

## Code Examples

Verified patterns from official sources:

### Accelerator Init + Prepare

```python
# Source: https://huggingface.co/docs/accelerate/package_reference/accelerator
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16")  # or "fp16" / "no"
model, optimizer, train_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, scheduler
)
```

### Linear Warmup + Decay Schedule

```python
# Source: https://huggingface.co/docs/transformers/main_classes/optimizer_schedules
from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=cfg.training.warmup_steps,      # e.g. 100
    num_training_steps=num_epochs * steps_per_epoch,
)
```

### CRF Loss Computation (verified against pytorch-crf 0.7.2 docs)

```python
# Source: https://pytorch-crf.readthedocs.io/en/stable/
from torchcrf import CRF

crf = CRF(num_tags=3, batch_first=True)

# emissions: (batch, seq_len, num_tags)
# labels: (batch, seq_len) — with -100 for ignored positions
mask = (labels != -100).bool()          # True = real token
clean_labels = labels.clone()
clean_labels[clean_labels == -100] = 0  # replace -100 AFTER mask is built
loss = -crf(emissions, clean_labels, mask=mask, reduction="mean")

# Inference / decode
predictions = crf.decode(emissions, mask=mask)  # list of lists
```

### PEFT LoRA for Token Classification

```python
# Source: https://huggingface.co/docs/peft/en/task_guides/token-classification-lora
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    r=16,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # verify << 100% trainable
```

### Gradient Clipping with Accelerate

```python
# Source: https://huggingface.co/docs/accelerate/package_reference/accelerator
accelerator.backward(loss)
accelerator.clip_grad_norm_(model.parameters(), max_norm=cfg.training.max_grad_norm)
optimizer.step()
scheduler.step()
optimizer.zero_grad()
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual `torch.autocast` + `GradScaler` | `Accelerate` wraps all of it | 2021+ | Single code path for CUDA/MPS/CPU |
| MPS: fp16 not supported | MPS: bf16 supported (torch>=2.6), fp16 (torch>=2.8) | 2024-2025 | Can enable bf16 on Apple Silicon |
| HuggingFace Trainer for custom training | Plain loop + Accelerate | — | Full control over CRF loss, ensemble, custom logging |
| `AutoTokenizer` on gbert-large | `BertTokenizerFast` directly | transformers 5.x | Avoids tokenizer.json requirement |

**Deprecated/outdated:**
- `transformers.AdamW` (old path): use `torch.optim.AdamW` — the transformers version is a thin wrapper that is now deprecated in favor of torch native.
- `torch.cuda.amp.autocast()` directly: use `accelerator.autocast()` to stay device-portable.

---

## Open Questions

1. **LoRA + CRF interaction on gbert-large specifically**
   - What we know: PEFT LoRA works with BertModel for TOKEN_CLS per official docs
   - What's unclear: Whether `target_modules=["query", "value"]` matches gbert-large's internal attention module naming exactly (some models use different attribute names)
   - Recommendation: Add a `model.print_trainable_parameters()` call after LoRA application; if it shows 0 trainable params, inspect module names with `[n for n, _ in model.named_modules()]` and adjust `target_modules`

2. **Steps per epoch for IterableDataset**
   - What we know: `LLMGeneratedDataset` generates `samples_per_batch` samples per worker; total depends on DataLoader `num_workers` and how many batches the dataset yields
   - What's unclear: The exact `steps_per_epoch` count needed for `num_training_steps` in the scheduler — this is required before calling `get_linear_schedule_with_warmup`
   - Recommendation: Use `cfg.data.samples_per_batch * num_workers` as `steps_per_epoch`; or use a fixed estimate (e.g., 100) with a note that the schedule is approximate for IterableDataset

3. **Gradient-boost ensemble (ENSM-04) complexity**
   - What we know: Standard ENSM-04 requires computing per-sample loss on the first model's predictions to weight resampling
   - What's unclear: Whether this is worth full implementation vs. a simplified version (e.g., oversample wrong predictions by 2x)
   - Recommendation: Implement as config-gated stub; if `ensemble.use_gradient_boost: false` (add to config), skip the error-weight step and just do uniform resampling

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (already configured) |
| Config file | `pytest.ini` — `testpaths = tests`, `asyncio_mode = auto` |
| Quick run command | `pytest tests/test_ner_model.py tests/test_trainer.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MODL-01 | Model forward pass returns logits shape (B, S, 3) | unit | `pytest tests/test_ner_model.py::test_forward_shape -x` | Wave 0 |
| MODL-02 | CRF enabled: loss computed; CRF disabled: cross-entropy loss | unit | `pytest tests/test_ner_model.py::test_crf_toggle -x` | Wave 0 |
| MODL-03 | Optimizer has 2 param groups with distinct LRs | unit | `pytest tests/test_trainer.py::test_differential_lr -x` | Wave 0 |
| MODL-04 | `resolve_mixed_precision` returns correct value per device/torch version | unit | `pytest tests/test_trainer.py::test_mixed_precision_resolution -x` | Wave 0 |
| MODL-05 | Gradient clipping called with correct max_norm | unit | `pytest tests/test_trainer.py::test_gradient_clipping -x` | Wave 0 |
| MODL-06 | Scheduler LR increases during warmup, decreases after | unit | `pytest tests/test_trainer.py::test_lr_schedule_shape -x` | Wave 0 |
| MODL-07 | Freeze: BERT params require_grad=False; LoRA: only adapter params trainable | unit | `pytest tests/test_ner_model.py::test_freeze_toggle test_lora_toggle -x` | Wave 0 |
| MODL-08 | Model can be instantiated and forward-passed on CPU (CI proxy) | unit | `pytest tests/test_ner_model.py::test_cpu_forward -x` | Wave 0 |
| ENSM-01 | N checkpoints saved when ensemble.enabled=true | unit | `pytest tests/test_trainer.py::test_ensemble_n_checkpoints -x` | Wave 0 |
| ENSM-02 | First model writes cache; second model reads from cache path | unit | `pytest tests/test_trainer.py::test_ensemble_cache_handoff -x` | Wave 0 |
| ENSM-03 | Majority vote function returns correct labels for 3-model toy case | unit | `pytest tests/test_trainer.py::test_ensemble_majority_vote -x` | Wave 0 |
| ENSM-04 | Gradient-boost variant: stub present; toggled by config key | unit | `pytest tests/test_trainer.py::test_ensemble_gradient_boost_stub -x` | Wave 0 |

**Testing strategy note:** All model tests should use tiny BERT (`bert-base-uncased` or CPU mock weights) to avoid downloading gbert-large in CI. Use `unittest.mock.patch` to replace `from_pretrained` calls, or use a `BertConfig`-based randomly initialized model for fast unit tests.

### Sampling Rate

- **Per task commit:** `pytest tests/test_ner_model.py tests/test_trainer.py -x`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_ner_model.py` — covers MODL-01, MODL-02, MODL-07, MODL-08
- [ ] `tests/test_trainer.py` — covers MODL-03, MODL-04, MODL-05, MODL-06, ENSM-01, ENSM-02, ENSM-03, ENSM-04
- [ ] `src/model/__init__.py` — package init
- [ ] `run.py` — CLI entry point (at minimum `train` subcommand stub)
- [ ] `requirements.txt` update — add `accelerate`, `pytorch-crf`, `peft`

---

## Sources

### Primary (HIGH confidence)

- [pytorch-crf 0.7.2 docs](https://pytorch-crf.readthedocs.io/en/stable/) — CRF API: forward/decode params, mask behavior, `pytorch-crf` PyPI package name
- [HuggingFace transformers optimization docs](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules) — `get_linear_schedule_with_warmup` exact signature, SchedulerType enum
- [HuggingFace Accelerate docs](https://huggingface.co/docs/accelerate/package_reference/accelerator) — Accelerator init, `mixed_precision` parameter, `prepare()`, `backward()`, `clip_grad_norm_`
- [HuggingFace PEFT LoRA docs](https://huggingface.co/docs/peft/en/task_guides/token-classification-lora) — `LoraConfig`, `TaskType.TOKEN_CLS`, `target_modules` for BERT

### Secondary (MEDIUM confidence)

- WebSearch: Accelerate MPS bf16 support requires torch>=2.6, fp16 requires torch>=2.8 (corroborated by multiple sources including GitHub issue #1432)
- WebSearch: `transformers.AdamW` deprecated in favor of `torch.optim.AdamW` (multiple sources confirm)
- WebSearch + HuggingFace forums: `BertForTokenClassification` uses internal CrossEntropyLoss with ignore_index=-100; CRF path requires `BertModel` for raw logits

### Tertiary (LOW confidence)

- The exact torch version requirement for MPS bf16 (2.6) vs fp16 (2.8) was from a search result summary, not directly verified from a PyTorch changelog — treat as guidance, verify at runtime.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — pytorch-crf, accelerate, peft are all official maintained packages verified via docs
- Architecture: HIGH — patterns derived from official docs; CRF mask handling verified against pytorch-crf API
- Pitfalls: HIGH — most pitfalls derived from existing STATE.md decisions (tokenizer issue) + verified pytorch-crf API constraints (no -100 support)

**Research date:** 2026-03-13
**Valid until:** 2026-04-13 (accelerate/peft are stable; MPS torch version table may shift)
