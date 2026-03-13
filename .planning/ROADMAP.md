# Roadmap: REG_ML — Regulatory Reference Extraction Pipeline

## Overview

Four phases that follow the natural dependency chain of an ML pipeline: lay the foundation and establish the regex benchmark first, then validate the riskiest component (data generation and BIO alignment), then build and train the model, and finally wire evaluation and inference together for the PoC verdict. Each phase produces something independently verifiable. Optional enhancements (CRF, ensemble) are included in Phase 3 as config-toggle features so they don't require a separate phase.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation** - Project scaffold, YAML config, device detection, and regex baseline
- [ ] **Phase 2: Data Pipeline** - LLM data generator, BIO converter, IterableDataset, and frozen gold test set
- [ ] **Phase 3: Model + Training** - gbert-large token classifier, training loop, and optional CRF/ensemble
- [ ] **Phase 4: Evaluation + Inference** - Full eval harness, baseline comparison, CLI inference, and PoC packaging

## Phase Details

### Phase 1: Foundation
**Goal**: Project runs, config is YAML-driven, device detection works on all hardware, and the regex baseline is producing real benchmark numbers
**Depends on**: Nothing (first phase)
**Requirements**: CONF-01, CONF-02, CONF-03, CONF-04, EVAL-02, EVAL-03, DOCS-01, DOCS-02
**Success Criteria** (what must be TRUE):
  1. User runs `pip install -r requirements.txt && export OPENROUTER_API_KEY=...` and the project is fully operational in under 10 minutes
  2. User runs a smoke test and the device auto-detected matches hardware (MPS on M1, CUDA on RTX, CPU otherwise)
  3. User runs the regex baseline on a set of sample regulatory sentences and receives Precision/Recall/F1 output covering all major reference types (§, Art., Abs., Tz., Nr., lit., Satz, Anhang, Verordnung)
  4. User edits `config.yaml` to change any hyperparameter and the change is picked up without touching code; CLI override `--model.use_crf=false` works
**Plans**: 2 plans

Plans:
- [x] 01-01: Project scaffold, YAML config layer (OmegaConf), CLI entry point skeleton, device detection, fixed-seed setup
- [x] 01-02: Regex baseline (all German legal reference patterns), entity-level evaluation wrapper using seqeval, README + .env.example

### Phase 2: Data Pipeline
**Goal**: LLM-generated training data flows into the model input format with verified BIO label correctness, and the gold test set is frozen on disk before any model training begins
**Depends on**: Phase 1
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, DATA-07, DATA-08, DATA-09, DATA-10, DATA-11, GOLD-01, GOLD-02, GOLD-03
**Success Criteria** (what must be TRUE):
  1. User triggers a data generation smoke test and receives a batch of tokenized samples with BIO labels; the round-trip check (text → BIO → decoded spans) matches the original LLM-tagged spans
  2. Special tokens ([CLS], [SEP], [PAD]) show label -100 in every generated sample; no sample has an O-label for a special token
  3. User runs the gold set generator script once; `gold_test_set.json` is written to disk with all entries marked `needs_review: true` and containing both positive and negative examples
  4. LLM generation retries on rate-limit/timeout with exponential backoff; a simulated failure does not crash the training loop
  5. Cached data from disk is usable as a reproducible training source (enabling ensemble resampling)
**Plans**: 3 plans

Plans:
- [ ] 02-01-PLAN.md — Async LLM client (OpenRouter/Gemini Flash, httpx, tenacity retry, ref-tag parser, domain rotation)
- [ ] 02-02-PLAN.md — BIO converter (char-span to token-level BIO via offset_mapping), JSONL disk cache, IterableDataset with worker sharding
- [ ] 02-03-PLAN.md — Gold test set builder (LLM-generated, fixed seed, JSON persistence, needs_review flag, positive/negative mix)

### Phase 3: Model + Training
**Goal**: A trained gbert-large token classifier checkpoint exists on disk, produced by a training loop that uses differential learning rates, warmup, and mixed precision, with CRF and ensemble available as config toggles
**Depends on**: Phase 2
**Requirements**: MODL-01, MODL-02, MODL-03, MODL-04, MODL-05, MODL-06, MODL-07, MODL-08, ENSM-01, ENSM-02, ENSM-03, ENSM-04
**Success Criteria** (what must be TRUE):
  1. User runs `python run.py train` and training completes at least one epoch without error on both MPS and CUDA; a checkpoint is saved to disk
  2. Training log shows two distinct learning rates in the optimizer (lower for BERT encoder, higher for classification head) and a warmup + decay LR schedule
  3. User sets `model.use_crf: true` in config and training runs with CRF enabled; user sets it to `false` and training runs without CRF — both paths produce a checkpoint
  4. User sets `ensemble.enabled: true` and trains N models; subsequent models resample from the cached data written by the first model
  5. Training runs on Apple Silicon (MPS) without OOM at batch_size=4 using mixed precision (bf16 or disabled); CUDA uses fp16
**Plans**: TBD

Plans:
- [ ] 03-01: RegulatoryNERModel (gbert-large + linear head, optional CRF layer, LoRA/freeze toggle via config)
- [ ] 03-02: Trainer (differential LR AdamW, warmup + linear decay, mixed precision via Accelerate, gradient clipping, checkpoint saving, per-epoch logging, ensemble driver)

### Phase 4: Evaluation + Inference
**Goal**: The PoC delivers its verdict — a comparison table showing whether the ML model beats the regex baseline on recall over the frozen gold test set — and a CLI that converts arbitrary German text to reference spans
**Depends on**: Phase 3
**Requirements**: EVAL-01, EVAL-04, EVAL-05, EVAL-06, INFR-01, INFR-02, INFR-03, DOCS-03
**Success Criteria** (what must be TRUE):
  1. User runs `python run.py evaluate` and receives a side-by-side table of ML model vs regex baseline showing entity-level Precision, Recall, and F1 on the gold test set
  2. Evaluation output includes per-reference-type breakdown (§, Art., Tz., etc.) and a false-positive / false-negative dump file for error analysis
  3. Evaluation reports both exact match and partial match (IoU > 0.5) scores
  4. User runs `python run.py predict --text "Gemäß § 25a Abs. 1 KWG gilt folgendes"` and receives character-offset spans with confidence scores in under 5 seconds
  5. User runs batch prediction on a list of texts and all return valid span output; all public methods and classes have Google-style docstrings with type hints
**Plans**: TBD

Plans:
- [ ] 04-01: Evaluator (seqeval entity-level P/R/F1, per-type breakdown, exact + IoU scoring, FP/FN dump, side-by-side ML vs regex comparison table)
- [ ] 04-02: Predictor + CLI integration (load checkpoint, text → span list with confidence scores, batch prediction, run.py train/evaluate/predict subcommands, Google-style docstrings across all modules)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 2/2 | Complete    | 2026-03-13 |
| 2. Data Pipeline | 1/3 | In Progress|  |
| 3. Model + Training | 0/2 | Not started | - |
| 4. Evaluation + Inference | 0/2 | Not started | - |
