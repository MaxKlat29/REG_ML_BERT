# Stack Research: German Regulatory Reference Extraction

## Core ML Stack

### Base Model: deepset/gbert-large
- **Version:** Latest on HuggingFace (safetensors available)
- **Why:** Best pretrained German BERT. 1024d hidden size, WordPiece tokenizer, 58k+ downloads. GermEval14 NER F1: 88.16%. MIT license.
- **Confidence:** HIGH
- **Alternatives considered:**
  - `dbmdz/bert-base-german-cased` — smaller (768d), less capacity for complex spans
  - `xlm-roberta-large` — multilingual but diluted German performance vs dedicated model
  - `deepset/gbert-base` — 768d, use only if OOM on gbert-large

### PyTorch: torch >= 2.0
- **Current:** 2.10.0
- **Why:** Native MPS (Apple Silicon) support since 2.0. CUDA support. AMP (automatic mixed precision) for fp16.
- **Confidence:** HIGH
- **Note:** MPS backend has quirks with fp16 — use bf16 or disable AMP on MPS.

### HuggingFace Transformers: >= 4.35
- **Current:** 5.3.0
- **Why:** `AutoModelForTokenClassification` is the standard surface for BIO tagging. `DataCollatorForTokenClassification` handles padding + label alignment. Requires Python >= 3.10.
- **Confidence:** HIGH
- **Key classes:** `AutoTokenizer`, `AutoModelForTokenClassification`, `DataCollatorForTokenClassification`

### Accelerate: >= 0.25
- **Current:** 1.13.0
- **Why:** Transparent device dispatch (MPS/CUDA/CPU). No device-specific code needed. Handles mixed precision setup.
- **Confidence:** HIGH

## CRF Layer

### Recommendation: Custom inline implementation (~80 lines)
- **Why:** Both `pytorch-crf` (last release 2019) and `torchcrf` (last release 2020) are unmaintained. With only 3 labels (O, B-REF, I-REF), a custom CRF is simple and avoids dependency risk.
- **Confidence:** HIGH
- **What NOT to use:** `pytorch-crf` (pip: pytorch-crf >= 0.7.2) — works but unmaintained, potential compatibility issues with future PyTorch versions.
- **Alternative:** If custom CRF is too much effort for PoC, the `pytorch-crf` package does work and is stable enough for a PoC.

## Data Generation & HTTP

### httpx: >= 0.25
- **Current:** 0.28.x
- **Why:** Async HTTP client for OpenRouter API calls. Native async/await, connection pooling, timeout handling. Preferred over `requests` for async workloads.
- **Confidence:** HIGH
- **Alternative considered:** `openai` SDK (2.26.0) — works with OpenRouter's OpenAI-compatible endpoint, handles retries automatically. Either works; httpx gives more control over retry/backoff logic.

## Config Management

### OmegaConf: >= 2.3
- **Why:** YAML → structured config with dot-notation access, CLI overrides (`--model.use_crf=false`), merge semantics. Standard in ML projects (used by Hydra).
- **Confidence:** HIGH
- **Alternative considered:** Pydantic 2.x for config validation at startup — heavier but gives type checking. OmegaConf is simpler for YAML-driven workflow.

## Evaluation

### seqeval: >= 1.2.2
- **Why:** Standard library for entity-level (span-level) F1/Precision/Recall on BIO sequences. Used in official HuggingFace NER tutorials. Drop-in metrics calculation.
- **Confidence:** HIGH

### scikit-learn: >= 1.3
- **Why:** Classification reports, confusion matrices, support utilities. Standard ML toolkit.
- **Confidence:** HIGH

## LoRA (Optional)

### PEFT: >= 0.6.0
- **Current:** 0.15.x
- **Why:** Parameter-efficient fine-tuning. Useful if gbert-large is too large for full fine-tuning on M1 memory. LoRA rank 16 reduces trainable params by ~95%.
- **Confidence:** HIGH

## Utilities

| Package | Version | Purpose |
|---------|---------|---------|
| pyyaml | >= 6.0 | YAML parsing (OmegaConf dependency) |
| tqdm | latest | Progress bars for training/generation |
| python-dotenv | latest | Load .env for OPENROUTER_API_KEY |
| wandb | latest (optional) | Experiment tracking |
| numpy | >= 1.24 | Array operations, seeding |

## What NOT to Use

| Package | Why Not |
|---------|---------|
| spaCy | Overkill for single-entity BIO tagging; adds heavy dependency |
| flair | Good NER framework but hides too much; less control over training loop |
| Hydra | Too complex for PoC config; OmegaConf alone is sufficient |
| MLflow | Explicit out-of-scope; wandb optional is enough |
| datasets (HuggingFace) | Map-style, not designed for online/streaming IterableDataset pattern |
| sentencepiece | gbert-large uses WordPiece, not SentencePiece |

## Device Compatibility Matrix

| Device | PyTorch Backend | fp16/AMP | Batch Size (gbert-large) | Notes |
|--------|----------------|----------|--------------------------|-------|
| Apple M1 16GB | MPS | Disable or bf16 | 4-8 | Use gradient accumulation to simulate larger batch |
| Apple M1 32GB | MPS | Disable or bf16 | 8-16 | Comfortable |
| RTX 3090 24GB | CUDA | fp16 ✓ | 16-32 | Sweet spot |
| RTX 4090 24GB | CUDA | fp16 ✓ | 16-32 | Fast |
| CPU | CPU | No | 2-4 | Very slow, only for testing |

---
*Researched: 2026-03-13*
