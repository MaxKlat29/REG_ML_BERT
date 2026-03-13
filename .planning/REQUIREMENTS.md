# Requirements: REG_ML — Regulatory Reference Extraction Pipeline

**Defined:** 2026-03-13
**Core Value:** Reliably find every legal reference in German regulatory text (recall over precision)

## v1 Requirements

### Model Core

- [ ] **MODL-01**: User can train a BIO token classifier (O, B-REF, I-REF) on gbert-large with a linear classification head
- [ ] **MODL-02**: User can enable/disable CRF layer via config toggle to enforce valid BIO transitions
- [ ] **MODL-03**: Model uses differential learning rates (lower for BERT encoder, higher for classification head)
- [ ] **MODL-04**: Training uses mixed precision (fp16 on CUDA, disabled/bf16 on MPS) with automatic device detection
- [ ] **MODL-05**: Training uses gradient clipping to prevent exploding gradients
- [ ] **MODL-06**: Training uses linear warmup + linear decay learning rate schedule
- [ ] **MODL-07**: User can freeze BERT or apply LoRA via config toggle as alternative to full fine-tuning
- [ ] **MODL-08**: Model runs on Apple Silicon (MPS), NVIDIA GPU (CUDA), and CPU with automatic detection

### Data Pipeline

- [x] **DATA-01**: Training data is generated on-the-fly by an LLM via OpenRouter API (no static dataset required)
- [x] **DATA-02**: LLM generates German regulatory text blocks with `<ref>...</ref>` tagged reference spans
- [x] **DATA-03**: Character-level reference spans are converted to token-level BIO labels using tokenizer offset_mapping
- [x] **DATA-04**: BIO conversion correctly handles BERT subword tokenization (first subtoken gets label, rest configurable)
- [x] **DATA-05**: Special tokens ([CLS], [SEP], [PAD]) receive label -100 (ignored by loss)
- [x] **DATA-06**: PyTorch IterableDataset generates training batches on-the-fly via LLM calls
- [x] **DATA-07**: LLM generation uses fixed seed per batch (epoch * 10000 + batch_idx * 100 + offset) for reproducibility
- [x] **DATA-08**: LLM generation includes retry logic with exponential backoff and configurable rate limiting
- [x] **DATA-09**: Generated data is validated — character offsets verified against actual text content
- [x] **DATA-10**: LLM prompt rotates across regulatory domains (BGB, KWG, MaRisk, DORA, DSGVO, CRR, etc.) for diversity
- [x] **DATA-11**: Generated data can be cached to disk for ensemble training and reproducibility

### Gold Test Set

- [ ] **GOLD-01**: User can generate a gold test set via CLI script (LLM-generated, persisted as JSON)
- [ ] **GOLD-02**: Gold test set samples are marked as "needs_review" for manual validation
- [ ] **GOLD-03**: Gold test set contains mix of positive (with references) and negative (no references) examples

### Evaluation

- [ ] **EVAL-01**: Evaluation reports entity-level Precision, Recall, and F1 (not token-level)
- [x] **EVAL-02**: Regex baseline extracts references using pattern matching (§, Artikel, Abs., Anhang, Verordnung, etc.)
- [x] **EVAL-03**: Regex baseline is evaluated with same metrics as ML model for direct comparison
- [ ] **EVAL-04**: Evaluation reports per-reference-type metrics (§ references, Artikel, Tz., etc.)
- [ ] **EVAL-05**: Evaluation dumps false positives and false negatives to file for error analysis
- [ ] **EVAL-06**: Evaluation supports both exact match and partial match (IoU > 0.5) scoring

### Ensemble

- [ ] **ENSM-01**: User can enable bagging ensemble via config (n_estimators models with bootstrap resampling)
- [ ] **ENSM-02**: First model generates and caches training data; subsequent models resample from cache
- [ ] **ENSM-03**: Ensemble inference uses majority vote over BIO predictions at token level
- [ ] **ENSM-04**: User can optionally enable gradient-boost-cached variant with error-weighted retraining

### Inference

- [ ] **INFR-01**: User can run CLI prediction on arbitrary German text and get reference spans with character offsets
- [ ] **INFR-02**: Predictions include confidence scores (softmax probability or CRF marginals)
- [ ] **INFR-03**: User can run batch prediction on multiple texts

### Config & Setup

- [x] **CONF-01**: All hyperparameters controlled via single YAML config file (no hardcoded values)
- [x] **CONF-02**: Config supports CLI overrides (e.g., `--model.use_crf=false`)
- [x] **CONF-03**: Seeds are set for PyTorch, NumPy, and LLM generation for full reproducibility
- [x] **CONF-04**: Project runs after pip install + single env var (OPENROUTER_API_KEY), setup under 10 minutes

### Documentation

- [x] **DOCS-01**: README.md with project description, setup guide, usage examples, and Mermaid pipeline diagram
- [x] **DOCS-02**: .env.example with OPENROUTER_API_KEY placeholder
- [ ] **DOCS-03**: All classes and public methods have Google-style docstrings with type hints

## v2 Requirements

### Entity Linking
- **LINK-01**: Map extracted reference spans to canonical law database entries (e.g., "§ 25a KWG" → structured record)

### Nested References
- **NEST-01**: Support nested/overlapping reference spans (e.g., reference within a reference)

### Advanced Ensemble
- **ENSM-05**: Automatic ensemble member selection based on validation performance
- **ENSM-06**: Stacking ensemble with meta-learner

### Monitoring
- **MNTR-01**: Track data generation quality metrics over training runs
- **MNTR-02**: Detect and alert on model performance degradation

## Out of Scope

| Feature | Reason |
|---------|--------|
| REST API / FastAPI serving | PoC — CLI inference sufficient for evaluation |
| Frontend / UI / dashboard | Engineers evaluate via CLI; no end-user interface needed |
| Docker / containerization | Cross-platform solved via device detection; no deployment target |
| CI/CD pipeline | Manual workflow; no automated deployment |
| MLflow / complex experiment tracking | wandb optional is sufficient; avoid over-engineering |
| Multi-GPU / distributed training | Single device sufficient for PoC dataset sizes |
| Custom tokenizer training | gbert-large tokenizer already trained on German text |
| Real-time streaming inference | Batch CLI is sufficient |
| Hyperparameter search (Optuna) | Manual config variants; PoC needs one good config |
| Annotation UI | LLM generates labeled data; manual spot-check of gold set |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CONF-01 | Phase 1 | Complete |
| CONF-02 | Phase 1 | Complete |
| CONF-03 | Phase 1 | Complete |
| CONF-04 | Phase 1 | Complete |
| EVAL-02 | Phase 1 | Complete |
| EVAL-03 | Phase 1 | Complete |
| DOCS-01 | Phase 1 | Complete |
| DOCS-02 | Phase 1 | Complete |
| DATA-01 | Phase 2 | Complete |
| DATA-02 | Phase 2 | Complete |
| DATA-03 | Phase 2 | Complete |
| DATA-04 | Phase 2 | Complete |
| DATA-05 | Phase 2 | Complete |
| DATA-06 | Phase 2 | Complete |
| DATA-07 | Phase 2 | Complete |
| DATA-08 | Phase 2 | Complete |
| DATA-09 | Phase 2 | Complete |
| DATA-10 | Phase 2 | Complete |
| DATA-11 | Phase 2 | Complete |
| GOLD-01 | Phase 2 | Pending |
| GOLD-02 | Phase 2 | Pending |
| GOLD-03 | Phase 2 | Pending |
| MODL-01 | Phase 3 | Pending |
| MODL-02 | Phase 3 | Pending |
| MODL-03 | Phase 3 | Pending |
| MODL-04 | Phase 3 | Pending |
| MODL-05 | Phase 3 | Pending |
| MODL-06 | Phase 3 | Pending |
| MODL-07 | Phase 3 | Pending |
| MODL-08 | Phase 3 | Pending |
| ENSM-01 | Phase 3 | Pending |
| ENSM-02 | Phase 3 | Pending |
| ENSM-03 | Phase 3 | Pending |
| ENSM-04 | Phase 3 | Pending |
| EVAL-01 | Phase 4 | Pending |
| EVAL-04 | Phase 4 | Pending |
| EVAL-05 | Phase 4 | Pending |
| EVAL-06 | Phase 4 | Pending |
| INFR-01 | Phase 4 | Pending |
| INFR-02 | Phase 4 | Pending |
| INFR-03 | Phase 4 | Pending |
| DOCS-03 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 42 total
- Mapped to phases: 42
- Unmapped: 0

---
*Requirements defined: 2026-03-13*
*Last updated: 2026-03-13 after 01-02 completion (EVAL-02, EVAL-03, DOCS-01, DOCS-02 complete)*
