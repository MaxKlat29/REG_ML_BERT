# Milestones

## v1.0 PoC (Shipped: 2026-03-13)

**Phases completed:** 4 phases, 9 plans, 146 tests
**Timeline:** 2026-03-13 (single day, ~6 hours)
**LOC:** ~6,500 Python
**Commits:** 52

**Key accomplishments:**
1. YAML-driven config (OmegaConf) with CLI overrides and cross-platform device detection (MPS/CUDA/CPU)
2. Regex baseline covering all 10 German legal reference types with entity-level P/R/F1 evaluation
3. Online LLM data generation via OpenRouter with BIO token alignment via offset_mapping
4. gbert-large token classifier with optional CRF, LoRA, backbone freeze — all config-toggled
5. Training loop with differential LR, warmup+decay, Accelerate mixed precision, ensemble bagging
6. Full evaluation harness: ML vs regex comparison, per-type breakdown, IoU partial match, FP/FN dump
7. CLI predictor with char-offset spans and confidence scores, batch prediction support

**42/42 requirements satisfied. 0 gaps.**

**Key decisions:**
- BIO tagging (not span extraction) — simpler, well-understood for NER
- Online LLM data generation — no annotated German regulatory corpus exists
- gbert-large as base — best German BERT available
- Recall priority — missing a legal reference worse than false positive
- Dual-path model: BertForTokenClassification (non-CRF) / BertModel+Linear+CRF
- BertTokenizerFast (not AutoTokenizer) — only way to get offset_mapping on gbert-large

**Known issues:**
- MPS too slow for real training (~20min for 3 samples) — GPU per SSH needed
- CRF benefit on short spans uncertain — needs real training to validate
- Gold test set needs manual review before final verdict

---
