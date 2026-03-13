# Architecture Patterns

**Domain:** German Regulatory NER — Token Classification Pipeline (PoC)
**Researched:** 2026-03-13
**Confidence:** HIGH — well-established BERT NER pipeline patterns; domain-specific details (gbert-large, online LLM data generation) cross-referenced against PROJECT.md requirements

---

## Recommended Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     CLI Entry Point                             │
│          train.py / evaluate.py / predict.py / run.py          │
│               (reads config.yaml, dispatches to modules)        │
└────────┬──────────────┬────────────────────────────────────────┘
         │              │
         ▼              ▼
┌─────────────────┐  ┌──────────────────────────────────────────┐
│  Config Layer   │  │             Data Layer                   │
│  config/        │  │  data_generation/ + dataset/             │
│  - config.yaml  │  │  LLMDataGenerator → BIOConverter →       │
│  - schema.py    │  │  RegulatoryDataset (IterableDataset)      │
│  - loader.py    │  └──────────────────┬───────────────────────┘
└─────────────────┘                     │
                                        ▼
                          ┌─────────────────────────┐
                          │       Model Layer        │
                          │  models/                 │
                          │  gbert-large backbone    │
                          │  + Linear Head           │
                          │  + optional CRF          │
                          └────────────┬─────────────┘
                                       │
                          ┌────────────┴─────────────┐
                          ▼                           ▼
             ┌────────────────────┐    ┌──────────────────────────┐
             │   Training Loop    │    │    Inference Engine       │
             │   training/        │    │    inference/             │
             │   Trainer class    │    │    Predictor class        │
             │   optimizer/sched  │    │    text → span list       │
             └────────┬───────────┘    └──────────────────────────┘
                      │
                      ▼
             ┌────────────────────┐
             │  Evaluation Layer  │
             │  evaluation/       │
             │  entity-level P/R/F1│
             │  + regex baseline  │
             └────────────────────┘
```

---

## Component Boundaries

| Component | Responsibility | Inputs | Outputs | Communicates With |
|-----------|---------------|--------|---------|-------------------|
| **Config Loader** | Parse YAML, validate schema, expose typed config object | config.yaml file path | TrainingConfig dataclass | All other components (read-only) |
| **LLM Data Generator** | Call OpenRouter API, produce raw samples as {text, spans[]} dicts | Config (LLM model, prompt templates, batch size) | Raw JSON samples | BIO Converter |
| **BIO Converter** | Map character-level reference spans to token-level BIO labels | Raw sample {text, spans[]}, tokenizer | {input_ids, attention_mask, labels} tensors | LLM Data Generator, Dataset, Tokenizer |
| **Regulatory Dataset** | PyTorch IterableDataset; orchestrate generation + conversion on-the-fly | LLM Generator + BIO Converter + Config | Batched tensor dicts | Training Loop, DataLoader |
| **Gold Test Set Builder** | Generate + cache a fixed evaluation set via LLM; write to disk | Config (gold set size, seed) | JSON file on disk | Evaluation Layer |
| **Model** | gbert-large + Linear classification head + optional CRF | Tokenized tensors | Logit or CRF emission scores | Training Loop, Inference Engine |
| **Trainer** | Orchestrate training loop: forward, loss, backward, optimizer step, warmup | Model + Dataset + Config | Checkpoints on disk | Model, Dataset, Evaluator |
| **Evaluator** | Compute entity-level P/R/F1 on gold set; run regex baseline | Gold set + Model or Regex | Metrics dict, comparison table | Trainer (called at eval steps), CLI |
| **Inference Engine** | Load checkpoint, tokenize raw text, decode predictions to character spans | Raw German text string | List of {start, end, text} reference spans | Model, Tokenizer |
| **Regex Baseline** | Pattern-match German legal references without ML | Raw German text string | List of {start, end, text} reference spans | Evaluator |
| **Device Manager** | Detect MPS / CUDA / CPU and set torch.device accordingly | Runtime environment | torch.device | Model, Trainer, Inference Engine |

---

## Data Flow

### Training Flow

```
config.yaml
    │
    ▼
Config Loader ──────────────────────────────────────────────────────┐
    │                                                                │
    ▼                                                                │
LLM Data Generator                                                   │
  │  POST /chat/completions (OpenRouter)                             │
  │  prompt: "Generate German regulatory text with annotations"      │
  │  response: [{text: "...", spans: [{start, end, label}]}]         │
  ▼                                                                  │
BIO Converter                                                        │
  │  tokenizer.encode(text, return_offsets_mapping=True)             │
  │  align char spans → token indices                                │
  │  assign B-REF / I-REF / O per token                             │
  ▼                                                                  │
RegulatoryDataset (IterableDataset)                                  │
  │  yields {input_ids, attention_mask, labels, token_type_ids}      │
  ▼                                                                  │
DataLoader (PyTorch)                                                 │
  │  batching, padding to max_length                                 │
  ▼                                                                  │
Trainer ←───────────────────────────────────────────────────────────┘
  │  gbert-large forward pass
  │  [CLS][tok1][tok2]...[SEP] → hidden states (1024d per token)
  │  Linear(1024 → 3) → logits [O, B-REF, I-REF]
  │  optional: CRF layer → Viterbi decode
  │  CrossEntropyLoss (ignore padding index -100)
  │  AdamW (differential LR: backbone 2e-5, head 1e-4)
  │  warmup scheduler
  │  mixed precision (torch.cuda.amp / torch.mps)
  ▼
Checkpoint saved to disk (model weights + config snapshot)
  │
  ▼
Evaluator (called every N steps)
  │  decode predictions → entity spans
  │  compare vs gold annotations (seqeval or custom entity-level)
  │  log P / R / F1
  ▼
Comparison vs Regex Baseline
  │  run regex on same gold set
  │  print comparison table: ML vs Regex P/R/F1
```

### Inference Flow

```
Raw German text (CLI input)
    │
    ▼
Tokenizer (gbert-large tokenizer)
  │  encode with offset mapping
  ▼
Model (loaded from checkpoint)
  │  forward pass → logits
  │  argmax (or CRF Viterbi) → token label sequence
  ▼
Span Decoder
  │  walk label sequence
  │  collect B-REF...I-REF runs
  │  map token offsets back to character offsets
  ▼
Output: [{start: int, end: int, text: str}, ...]
    │
    ▼
CLI prints extracted references
```

### Gold Set Generation Flow (one-time, then cached)

```
Config (gold_set_size, seed)
    │
    ▼
LLM Data Generator (with fixed seed prompt)
    │
    ▼
BIO Converter
    │
    ▼
gold_test_set.json (written to disk, manually reviewable)
  structure: [{text, spans, bio_labels, tokens}]
    │
    ▼
Evaluator reads from disk during eval (never re-generated unless deleted)
```

---

## Module Structure

```
reg_ml/
├── config/
│   ├── __init__.py
│   ├── schema.py          # Pydantic or dataclass config models
│   └── loader.py          # load_config(path) → TrainingConfig
│
├── data/
│   ├── __init__.py
│   ├── generator.py       # LLMDataGenerator: calls OpenRouter, returns raw samples
│   ├── bio_converter.py   # BIOConverter: char spans → token BIO labels
│   ├── dataset.py         # RegulatoryDataset(IterableDataset)
│   └── gold_set.py        # GoldSetBuilder: generate + persist test set
│
├── models/
│   ├── __init__.py
│   ├── ner_model.py       # RegulatoryNERModel: gbert-large + head + optional CRF
│   └── crf.py             # CRF layer (conditional random field)
│
├── training/
│   ├── __init__.py
│   ├── trainer.py         # Trainer class: full training loop
│   └── optimizers.py      # differential LR, AdamW, scheduler setup
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py       # entity-level P/R/F1 computation
│   └── baseline.py        # RegexBaseline: rule-based German legal ref extractor
│
├── inference/
│   ├── __init__.py
│   └── predictor.py       # Predictor: text → reference span list
│
├── utils/
│   ├── __init__.py
│   ├── device.py          # get_device(): MPS / CUDA / CPU detection
│   ├── seeding.py         # set_all_seeds(seed)
│   └── logging.py         # structured logging setup
│
├── config.yaml            # All hyperparameters, no hardcoded values in code
└── run.py                 # CLI entry point: train / evaluate / predict subcommands
```

---

## Patterns to Follow

### Pattern 1: Tokenizer-Aligned BIO Labeling

BERT tokenizers split words into subword pieces. A single word like "§" may tokenize to one token, but "Abs." may split across multiple. The BIO conversion must:
1. Tokenize with return_offsets_mapping=True
2. For each token, check if its char offset range overlaps a labeled span
3. Assign B-REF to the first overlapping token, I-REF to continuations, O elsewhere
4. Assign -100 (PyTorch ignore index) to special tokens ([CLS], [SEP], padding)

```python
def align_labels_to_tokens(
    text: str,
    spans: list[dict],         # [{start, end}]
    tokenizer,
    max_length: int = 512,
) -> dict:
    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
    )
    offset_mapping = enc["offset_mapping"]
    labels = []
    for token_start, token_end in offset_mapping:
        if token_start == token_end:          # special token
            labels.append(-100)
            continue
        label = "O"
        for span in spans:
            if token_start >= span["start"] and token_end <= span["end"]:
                label = "B-REF" if token_start == span["start"] else "I-REF"
                break
        labels.append(LABEL2ID[label])
    enc["labels"] = labels
    return enc
```

Confidence: HIGH — canonical HuggingFace NER tokenization pattern.

### Pattern 2: IterableDataset with Online Generation

Since data is LLM-generated on-the-fly, use torch.utils.data.IterableDataset rather than a map-style dataset. This avoids pre-generating the entire dataset.

```python
class RegulatoryDataset(IterableDataset):
    def __init__(self, generator, converter, samples_per_epoch):
        self.generator = generator
        self.converter = converter
        self.samples_per_epoch = samples_per_epoch

    def __iter__(self):
        for _ in range(self.samples_per_epoch):
            raw = self.generator.generate_one()
            yield self.converter.convert(raw)
```

Workers must use different random seeds to avoid duplicate samples across DataLoader workers.

Confidence: HIGH — standard PyTorch pattern for streaming datasets.

### Pattern 3: Differential Learning Rates

gbert-large's pretrained weights need much smaller updates than the randomly initialized classification head.

```python
optimizer = AdamW([
    {"params": model.bert.parameters(), "lr": config.backbone_lr},    # e.g. 2e-5
    {"params": model.classifier.parameters(), "lr": config.head_lr},  # e.g. 1e-4
])
```

Confidence: HIGH — standard fine-tuning practice, described in BERT paper and HuggingFace tutorials.

### Pattern 4: Entity-Level Evaluation (not token-level)

Token-level accuracy inflates scores because O tokens dominate. Use entity-level F1:
- A predicted entity is correct only if its entire span exactly matches the gold span
- seqeval library implements this directly
- Report separately: exact-match P/R/F1 and partial-overlap recall (more useful for regulatory context)

Confidence: HIGH — seqeval is the standard NER evaluation library.

### Pattern 5: Self-Contained Checkpoint Schema

Save everything needed to resume or run inference without the training config file:

```python
{
    "model_state_dict": model.state_dict(),
    "config_snapshot": asdict(config),      # embed config so inference is self-contained
    "label_map": LABEL2ID,
    "tokenizer_name": "deepset/gbert-large",
    "epoch": epoch,
    "step": global_step,
    "best_f1": best_f1,
}
```

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Padding Token Labels as Real Labels

**What goes wrong:** Padding positions get assigned label ID 0 (often "O"), causing the model to learn from meaningless padding tokens.

**Why bad:** Inflates loss signal. The model updates weights based on padding — this degrades performance.

**Instead:** Always assign -100 to padding positions and special tokens. PyTorch's CrossEntropyLoss ignores index -100 by default.

### Anti-Pattern 2: Map-Style Dataset That Pre-Generates All Data

**What goes wrong:** Calling the LLM API to pre-generate 10,000 samples before training starts.

**Why bad:** Blocks training start, incurs upfront API cost, and the dataset cannot dynamically adjust to training progress.

**Instead:** Use IterableDataset with online generation. Generate N samples per epoch. Cache the gold test set to disk (it must be fixed), but training data stays dynamic.

### Anti-Pattern 3: Single Learning Rate for All Parameters

**What goes wrong:** Using AdamW(model.parameters(), lr=1e-4) — same LR for backbone and head.

**Why bad:** Too high for pretrained weights (destroys learned German representations), too low for the random head (slow convergence).

**Instead:** Differential LR: backbone at ~2e-5, head at ~5e-5 to 1e-4.

### Anti-Pattern 4: Greedy Decoding Without CRF When Transitions Matter

**What goes wrong:** Model predicts O, I-REF, I-REF — an I-REF starting a sequence without a B-REF.

**Why bad:** Produces invalid BIO sequences. Post-processing to fix these is brittle.

**Instead:** Enable the CRF option in config. CRF's Viterbi algorithm enforces valid transitions at training and inference time. For PoC validation, start without CRF, measure invalid sequence rate, add CRF if it is non-trivial.

### Anti-Pattern 5: Mutable Global State for Device or Tokenizer

**What goes wrong:** device = get_device() at module import time; tokenizer loaded as module-level global.

**Why bad:** Breaks multi-process DataLoader (workers fork before device detection), makes testing difficult.

**Instead:** Instantiate device and tokenizer inside class constructors or pass them as constructor arguments. Use get_device() as a pure function called at runtime.

### Anti-Pattern 6: Evaluating on Training-Distribution LLM Data

**What goes wrong:** Using freshly-generated LLM samples (same distribution as training data) as the test set.

**Why bad:** Overfits evaluation to LLM generation quirks. The model may learn LLM formatting patterns rather than genuine German regulatory text patterns.

**Instead:** The gold test set must be generated once with a fixed seed, optionally human-reviewed, and then frozen on disk. Never regenerate it. Treat it as a static held-out set.

---

## Build Order (Phase Dependencies)

The architecture has clear dependency chains. Build in this order:

```
Phase 1 — Foundation (no dependencies)
├── Config Layer          (config/schema.py, config/loader.py, config.yaml)
├── Device Manager        (utils/device.py)
└── Seeding utils         (utils/seeding.py)

Phase 2 — Data Pipeline (depends on: Config, Tokenizer via HuggingFace)
├── LLM Data Generator    (data/generator.py) — OpenRouter API calls
├── BIO Converter         (data/bio_converter.py) — needs tokenizer + spans
└── IterableDataset       (data/dataset.py) — wraps Generator + Converter

Phase 3 — Model (depends on: Config, Device)
├── NER Model             (models/ner_model.py) — gbert-large + head
└── CRF Layer             (models/crf.py) — optional, can stub first

Phase 4 — Baseline (depends on: nothing except regex)
└── Regex Baseline        (evaluation/baseline.py) — independent, build early for benchmarking

Phase 5 — Training (depends on: Data Pipeline, Model, Config)
└── Trainer               (training/trainer.py, training/optimizers.py)

Phase 6 — Evaluation (depends on: Model, Gold Set Builder, Baseline)
├── Gold Set Builder      (data/gold_set.py) — depends on Generator + Converter
└── Evaluator             (evaluation/evaluator.py) — entity-level metrics

Phase 7 — Inference (depends on: Model, Tokenizer)
└── Predictor             (inference/predictor.py)

Phase 8 — CLI Integration (depends on: all above)
└── run.py                (train / evaluate / predict subcommands)
```

Key dependency insight: the Regex Baseline (Phase 4) is fully independent and should be built early so evaluation comparisons are available as soon as the ML model produces any output.

---

## Scalability Considerations

This is a PoC. Scalability is not the primary concern, but these boundaries should not be crossed during PoC without a conscious decision:

| Concern | PoC Scope | If Productionized Later |
|---------|-----------|------------------------|
| Data generation throughput | Sequential API calls, ~60 samples/min | Async batch calls or local LLM (vLLM) |
| Model inference latency | Single-text CLI, ~100ms per call | ONNX export, batched REST API |
| Checkpoint storage | Single best checkpoint + latest | MLflow / W&B artifact tracking |
| Multi-GPU training | Not needed (single M1 or RTX GPU) | DistributedDataParallel |
| Ensemble | Config toggle, sequential | Parallel ensemble member training |

---

## Critical Design Decisions for PoC

| Decision | Architectural Impact |
|----------|---------------------|
| IterableDataset (not map-style) | No __len__ defined; DistributedSampler cannot be used without custom iterator splitting |
| Online generation | num_workers > 0 in DataLoader requires careful seeding per worker (worker_init_fn) to avoid duplicated samples |
| CRF optional via config | Model class must have two forward paths; loss function differs (CRF NLL vs CrossEntropy) — plan for this from the start |
| Fixed gold test set | Gold set written to data/gold_test_set.json on first run; subsequent runs load from disk; protected from accidental overwrite |
| Differential LR | Optimizer setup must separate model.bert.parameters() from model.classifier.parameters() — model class design must support this separation clearly |
| gbert-large 1024d | Memory footprint: ~340MB model weights; batch_size=16 with max_length=512 needs ~8-12GB VRAM; batch size may need reduction for 8GB GPU configs |

---

## Sources

- HuggingFace Transformers documentation: token classification tutorial (canonical BIO alignment pattern)
- PyTorch documentation: IterableDataset and DataLoader worker seeding
- Original BERT paper (Devlin et al., 2018): differential learning rate rationale
- seqeval library documentation: entity-level NER evaluation
- deepset/gbert-large model card on HuggingFace Hub
- Lample et al., 2016 (LSTM-CRF for NER): CRF transition constraint rationale
- PROJECT.md: all project-specific constraints and requirements

**Confidence assessment:**
- BIO tokenization alignment pattern: HIGH (canonical HuggingFace pattern)
- IterableDataset for online generation: HIGH (standard PyTorch)
- Differential learning rates: HIGH (well-established fine-tuning practice)
- CRF layer design: MEDIUM (implementation varies; torchcrf vs custom are both used)
- Online LLM generation data quality: MEDIUM (novel approach, fewer established patterns)
- gbert-large specific memory requirements: MEDIUM (based on model size calculation, not benchmarked)
