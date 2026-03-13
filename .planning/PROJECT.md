# REG_ML — Regulatory Reference Extraction Pipeline

## What This Is

A Python ML pipeline (PoC) that automatically extracts legal references from German regulatory text blocks using token classification with BIO labels. From text like "Gemäß § 25a Abs. 1 KWG gilt folgendes", the model identifies and extracts "§ 25a Abs. 1 KWG" as a reference span. Built as a proof-of-concept — v1.0 shipped with full pipeline from data generation to CLI inference.

## Core Value

Reliably find every legal reference in German regulatory text (recall over precision) — missing a reference is worse than flagging a false positive.

## Requirements

### Validated

- ✓ Token classification model (gbert-large + Linear Head) with BIO labels — v1.0
- ✓ Optional CRF layer for label transition constraints — v1.0
- ✓ Online training data generation via LLM (OpenRouter API) — v1.0
- ✓ BIO conversion from character-level spans to token-level labels — v1.0
- ✓ PyTorch IterableDataset with on-the-fly LLM data generation — v1.0
- ✓ Training loop with differential learning rates, warmup, mixed precision — v1.0
- ✓ Optional ensemble (bagging with cached data) — v1.0
- ✓ Entity-level evaluation (Precision, Recall, F1) — v1.0
- ✓ Regex baseline as benchmark comparison — v1.0
- ✓ CLI inference: text in → reference spans out — v1.0
- ✓ Gold test set generation via LLM (manually reviewable) — v1.0
- ✓ YAML-driven config — no hardcoded hyperparameters — v1.0
- ✓ Cross-platform: Apple Silicon (MPS) + NVIDIA (CUDA) + CPU fallback — v1.0

### Active

- [ ] GPU training via SSH — real training run on CUDA hardware
- [ ] End-to-end evaluation: ML model vs regex on gold test set (the actual PoC verdict)
- [ ] Gold test set manual review (needs_review flags)

### Out of Scope

- Docker/containerization — PoC doesn't need it
- REST API / FastAPI endpoints — no serving layer
- Frontend / UI — CLI only
- CI/CD pipeline — manual workflow
- MLflow or complex experiment tracking — wandb optional only
- Mobile/edge deployment — desktop/server only

## Context

Shipped v1.0 PoC with ~6,500 LOC Python, 146 tests, 42 requirements satisfied.
Tech stack: Python 3.10+, PyTorch, HuggingFace Transformers, OmegaConf, httpx, seqeval, Accelerate, pytorch-crf, PEFT.
Training verified on MPS (loss decreasing) but MPS too slow for real training — GPU needed.
Next: SSH to GPU machine, real training, then `python run.py evaluate` for the PoC verdict.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| BIO tagging over span extraction | Simpler, well-understood approach for NER-style tasks | ✓ Good — clean implementation |
| Online LLM data generation | No annotated German regulatory corpus available | ✓ Good — diverse samples generated |
| gbert-large as base | Best available pretrained German BERT; 1024d | ✓ Good — but heavy on MPS |
| Recall over precision | Missing a legal reference has higher cost | — Pending real eval |
| CRF optional (config toggle) | Enforces BIO transitions, adds complexity | — Pending real eval |
| Ensemble optional (config toggle) | Bagging can improve robustness | — Pending real eval |
| BertTokenizerFast (not Auto) | AutoTokenizer lacks offset_mapping on gbert-large | ✓ Good — key discovery |
| Dual-path model architecture | BertForTokenClassification vs BertModel+CRF | ✓ Good — clean separation |
| Accelerate for device handling | Single code path CUDA/MPS/CPU | ✓ Good — simplified trainer |

## Constraints

- **Base model**: deepset/gbert-large (1024d hidden size) — pretrained German BERT
- **Data API**: OpenRouter (https://openrouter.ai/api/v1/chat/completions), auth via OPENROUTER_API_KEY env var
- **LLM for data**: Configurable, default google/gemini-2.0-flash-001
- **Python**: 3.10+
- **Immediacy**: Must be runnable end-to-end immediately after setup (pip install + env var)
- **Reproducibility**: Fixed seeds everywhere (PyTorch, NumPy, LLM generation per batch)

---
*Last updated: 2026-03-13 after v1.0 milestone*
