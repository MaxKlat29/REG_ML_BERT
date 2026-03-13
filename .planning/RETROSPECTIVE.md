# Retrospective

## Milestone: v1.0 — PoC

**Shipped:** 2026-03-13
**Phases:** 4 | **Plans:** 9 | **Tests:** 146 | **Commits:** 52

### What Was Built
- YAML-driven config system with OmegaConf + CLI overrides
- Cross-platform device detection (MPS/CUDA/CPU) with automatic mixed precision
- Regex baseline covering 10 German legal reference types
- Async LLM data generator (OpenRouter/Gemini Flash) with retry, domain rotation, JSONL cache
- BIO token-level converter via offset_mapping alignment
- Gold test set builder with needs_review flags
- gbert-large token classifier with CRF/LoRA/freeze config toggles
- Trainer: differential LR, warmup+decay, Accelerate, gradient clipping, checkpoints
- Ensemble driver with bagging and majority vote
- Evaluator: ML vs regex comparison, per-type breakdown, IoU partial match, FP/FN dump
- Predictor: char-offset spans with confidence, batch prediction, CLI integration

### What Worked
- **GSD methodology**: 4 phases planned + executed + verified in a single day — structured approach prevented scope creep
- **TDD with tiny BertConfig**: All model tests used BertConfig(hidden_size=64, layers=1) — instant, no downloads, deterministic
- **Wave-based parallel execution**: Plans 01 and 02 in each phase often ran in parallel subagents
- **offset_mapping for BIO alignment**: Robust solution for German compound words and subword tokenization
- **Config-toggle approach**: CRF, LoRA, ensemble all behind config flags — complexity opt-in

### What Was Inefficient
- **MPS training debugging**: Spent significant time discovering MPS async dispatch behavior (backward() returns fast, optimizer.step() blocks ~700s)
- **OOM iterations on MPS**: Multiple rounds of config reduction (batch 4→2→1, seq 512→256→64) before gbert-large fit in 9GB MPS memory
- **gradient_checkpointing + MPS**: Enabled for memory savings but made async dispatch even slower — net negative on MPS

### Patterns Established
- **Dual-path model architecture**: BertForTokenClassification (non-CRF) vs BertModel + Linear + CRF — clean separation via config toggle
- **Online LLM data generation**: Generate training data on-the-fly, no static dataset needed
- **JSONL cache for reproducibility**: First model writes, ensemble members resample from cache
- **Config overlay system**: default.yaml → gpu.yaml → CLI overrides (OmegaConf merge chain)
- **flush=True logging**: Essential when MPS async dispatch makes timing unpredictable

### Key Lessons
1. **MPS is async**: Operations dispatch to Metal queue; Python sees "done" but GPU still computing. Timing at Python level is misleading.
2. **gbert-large needs GPU**: 335M params + AdamW states exceed MPS 9GB limit at useful batch sizes. Real training must go to CUDA.
3. **BertTokenizerFast required**: AutoTokenizer on gbert-large returns slow tokenizer without offset_mapping. Must explicitly use BertTokenizerFast.
4. **pytorch-crf mask gotcha**: batch_first=True requires mask[:,0]=True — CLS position must stay unmasked even with label -100.
5. **load_dotenv() must be first**: Without it as first call in entry point, .env never loads and API key is empty.

### Cost Observations
- Model mix: ~80% Opus, ~20% Sonnet (subagents)
- Sessions: 1 major session (PoC complete in one day)
- Notable: Entire v1.0 PoC from zero to 146 tests in ~6 hours

## Cross-Milestone Trends

| Metric | v1.0 |
|--------|------|
| Phases | 4 |
| Plans | 9 |
| Tests | 146 |
| Requirements | 42/42 |
| Days | 1 |
| Key risk | MPS too slow for real training |
