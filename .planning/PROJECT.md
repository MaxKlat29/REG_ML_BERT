# REG_ML — Regulatory Reference Extraction Pipeline

## What This Is

A Python ML pipeline (PoC) that automatically extracts legal references from German regulatory text blocks using token classification with BIO labels. From text like "Gemäß § 25a Abs. 1 KWG gilt folgendes", the model identifies and extracts "§ 25a Abs. 1 KWG" as a reference span. Built for product evaluation — proving the approach works before productionizing.

## Core Value

Reliably find every legal reference in German regulatory text (recall over precision) — missing a reference is worse than flagging a false positive.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Token classification model (gbert-large + Linear Head) with BIO labels (O, B-REF, I-REF)
- [ ] Optional CRF layer for label transition constraints
- [ ] Online training data generation via LLM (OpenRouter API)
- [ ] BIO conversion from character-level spans to token-level labels
- [ ] PyTorch IterableDataset with on-the-fly LLM data generation
- [ ] Training loop with differential learning rates, warmup, mixed precision
- [ ] Optional ensemble (bagging with cached data)
- [ ] Entity-level evaluation (Precision, Recall, F1)
- [ ] Regex baseline as benchmark comparison
- [ ] CLI inference: text in → reference spans out
- [ ] Gold test set generation via LLM (manually reviewable)
- [ ] YAML-driven config — no hardcoded hyperparameters
- [ ] Cross-platform: Apple Silicon (MPS) + NVIDIA (CUDA) + CPU fallback

### Out of Scope

- Docker/containerization — PoC doesn't need it
- REST API / FastAPI endpoints — no serving layer
- Frontend / UI — CLI only
- CI/CD pipeline — manual workflow
- MLflow or complex experiment tracking — wandb optional only
- Mobile/edge deployment — desktop/server only

## Context

- **Domain**: German regulatory text across all legal areas (BGB, HGB, KWG, MaRisk, DORA, DSGVO, CRR, MiFID II, Solvency II, VAG, WpHG, KAGB, Basel III, StGB, Steuerrecht, etc.)
- **Reference types**: § references, Artikel, Absatz, Anhang, Verordnungen, Richtlinien, Textziffer (Tz.), lit., Nr., Satz — anything pointing to another legal source
- **Data strategy**: No static training dataset. Training data generated on-the-fly by LLM via OpenRouter. ~60% samples with references, ~40% without. Gold test set LLM-generated + manually validated.
- **Product context**: This PoC evaluates whether the approach is viable for a broader product. Must demonstrate the ML model beats a regex baseline, especially on recall.
- **Hardware**: Primary dev on Apple M1 (MPS backend), colleague runs on RTX GPU (CUDA). Both must work seamlessly with automatic device detection.

## Constraints

- **Base model**: deepset/gbert-large (1024d hidden size) — pretrained German BERT
- **Data API**: OpenRouter (https://openrouter.ai/api/v1/chat/completions), auth via OPENROUTER_API_KEY env var
- **LLM for data**: Configurable, default google/gemini-2.0-flash-001
- **Python**: 3.10+
- **Immediacy**: Must be runnable end-to-end immediately after setup (pip install + env var)
- **Reproducibility**: Fixed seeds everywhere (PyTorch, NumPy, LLM generation per batch)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| BIO tagging over span extraction | Simpler, well-understood approach for NER-style tasks; works well with BERT | — Pending |
| Online LLM data generation | No annotated German regulatory corpus available; LLM can generate diverse examples on-the-fly | — Pending |
| gbert-large as base | Best available pretrained German BERT; 1024d gives strong representations | — Pending |
| Recall over precision | Missing a legal reference has higher cost than a false positive in regulatory context | — Pending |
| CRF optional | Adds complexity but enforces valid BIO transitions; config toggle lets us A/B test | — Pending |
| Ensemble optional | Bagging can improve robustness but adds training time; config toggle for PoC flexibility | — Pending |

---
*Last updated: 2026-03-13 after initialization*
