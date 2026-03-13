# Project Research Summary

**Project:** REG_ML — German Regulatory Reference Extraction
**Domain:** German Legal NER / BIO Token Classification (PoC)
**Researched:** 2026-03-13
**Confidence:** MEDIUM-HIGH

## Executive Summary

REG_ML is a proof-of-concept NLP system that extracts regulatory references (e.g., "§ 25a KWG", "Art. 6 Abs. 1 DSGVO") from German legal text using BERT-based token classification. The established approach is to fine-tune a pretrained German BERT model (deepset/gbert-large) with a BIO tagging scheme — three labels (O, B-REF, I-REF) over tokenized input. The critical constraint for this PoC is that no publicly annotated German regulatory corpus exists, so training data must be generated synthetically via an LLM (OpenRouter/Gemini Flash). This shapes every architectural decision: online generation via IterableDataset, a separately cached gold test set, and a fixed-seed reproducibility strategy.

The recommended approach is a phased build that validates the riskiest assumptions early. The data pipeline — LLM generation, character-span-to-BIO alignment, and gold set construction — is the most technically risky component and must be built and verified before any model training begins. A regex baseline should be built in parallel since it is fully independent and defines the bar the ML model must beat. Once data and baseline are working, the model training loop follows well-established BERT fine-tuning patterns with differential learning rates and optional mixed precision. CRF and ensemble enhancements are deferred to post-baseline validation.

The primary risk is evaluation validity: if the gold test set is not independently generated and manually reviewed, the PoC may report inflated metrics that do not reflect real-world performance. Secondary risks are the BIO alignment bug (silent label corruption from subword tokenization) and OOM on Apple M1 with large batch sizes. These are all well-understood and preventable with explicit validation steps built into the data pipeline phase.

---

## Key Findings

### Recommended Stack

The stack is built on deepset/gbert-large as the backbone model — the strongest dedicated German BERT available, with 1024-dimensional hidden states and 88%+ F1 on GermEval14 NER. PyTorch 2.x + HuggingFace Transformers 4.35+ are the standard surface for BIO token classification, with HuggingFace's `DataCollatorForTokenClassification` handling padding and label alignment. The Accelerate library provides transparent MPS/CUDA/CPU dispatch so no device-specific code branches are needed.

CRF libraries (pytorch-crf, torchcrf) are both unmaintained since 2019-2020. The recommendation is a custom ~80-line CRF implementation to avoid dependency risk, or the pytorch-crf package as an acceptable PoC shortcut. OmegaConf is the right config layer — lightweight YAML with dot-notation access and CLI overrides, without Hydra's overhead. seqeval is the standard entity-level NER evaluation library and must be used instead of token-level metrics.

**Core technologies:**
- `deepset/gbert-large`: BERT backbone — best pretrained German model, 1024d, MIT license
- `torch >= 2.0`: Training runtime — native MPS support, AMP, required for M1 compatibility
- `transformers >= 4.35`: HuggingFace surface — `AutoModelForTokenClassification`, tokenizer, data collator
- `accelerate >= 0.25`: Device dispatch — transparent MPS/CUDA/CPU, no device-specific code needed
- `httpx >= 0.25`: Async HTTP — LLM API calls to OpenRouter with connection pooling and retry control
- `omegaconf >= 2.3`: Config management — YAML with CLI overrides, no hardcoded hyperparameters
- `seqeval >= 1.2.2`: Evaluation — entity-level P/R/F1, the only valid NER evaluation metric
- `peft >= 0.6.0` (optional): LoRA — if gbert-large exceeds M1 memory, reduces trainable params by ~95%

### Expected Features

**Must have (table stakes):**
- BIO token classification (O, B-REF, I-REF) — standard NER framing, baseline expectation
- Entity-level Precision / Recall / F1 via seqeval — token-level is insufficient and misleading
- Regex baseline as benchmark — ML claims are meaningless without a rule-based comparison target
- CLI inference (text in → spans out) — required for demo and evaluation
- YAML config with no hardcoded hyperparameters — reproducibility and A/B testing requirement
- Fixed-seed reproducibility — same config + seed must produce same metrics within 0.5%
- Cross-platform device detection (MPS / CUDA / CPU) — two developers, two hardware targets
- LLM-generated training data pipeline — the only viable data strategy; no public corpus exists
- Gold test set (generated once, manually reviewed, frozen on disk) — held-out evaluation set
- Character-span to BIO token alignment — critical bridge between LLM output and model input
- Training loop with differential learning rates — backbone 2e-5, head 1e-4; prevents catastrophic forgetting
- Mixed precision training — required for practical speed; bfloat16 on MPS, fp16 on CUDA

**Should have (differentiators):**
- Optional CRF layer (config toggle) — enforces valid BIO transitions; add if invalid sequences are frequent
- Error analysis output (false positives / false negatives to file) — high-value for PoC writeup
- Warmup schedule + cosine decay — better convergence than flat LR; standard for BERT fine-tuning
- Gradient clipping (max_norm=1.0) — defensive practice, especially relevant with CRF
- LLM data generation quality checks — verify char offsets before training, log invalid sample rate
- Configurable negative sample ratio — 60/40 is a hypothesis; expose as YAML param
- Sample diversity controls in LLM prompts — domain rotation (BGB, KWG, DSGVO, etc.)

**Defer (v2+):**
- Optional bagging ensemble — high complexity, adds significant training time; add only if single-model variance is high
- Reference type breakdown in evaluation — requires typed LLM annotations
- Span confidence scores — useful for downstream product filtering, not for PoC evaluation
- REST API / serving layer — out of scope; CLI is sufficient
- Hyperparameter search (Optuna / Ray Tune) — PoC needs one good config, not the optimal one

### Architecture Approach

The architecture is a clean layered pipeline with 8 modules: Config, Data (Generator + BIO Converter + Dataset + Gold Set Builder), Models (NER model + optional CRF), Training (Trainer + optimizer setup), Evaluation (Evaluator + Regex Baseline), Inference (Predictor), and a CLI entry point. The key structural insight is that the Regex Baseline is fully independent of the ML stack and should be built early. The data layer uses PyTorch IterableDataset for online LLM generation — no pre-generation step, samples are produced dynamically per epoch, while the gold test set is generated exactly once and frozen on disk.

**Major components:**
1. **Config Layer** (`config/`) — YAML-driven, typed dataclass, single source of truth for all hyperparameters
2. **LLM Data Generator** (`data/generator.py`) — async OpenRouter calls, produces `{text, spans[]}` dicts
3. **BIO Converter** (`data/bio_converter.py`) — maps character-level spans to token-level BIO labels via offset_mapping
4. **RegulatoryDataset** (`data/dataset.py`) — IterableDataset wrapping generator + converter; no pre-generation
5. **Gold Set Builder** (`data/gold_set.py`) — generates and persists the fixed evaluation set on first run
6. **RegulatoryNERModel** (`models/ner_model.py`) — gbert-large + linear head + optional CRF; two distinct forward paths
7. **Trainer** (`training/trainer.py`) — differential LR AdamW, warmup scheduler, mixed precision, checkpointing
8. **Evaluator + Regex Baseline** (`evaluation/`) — entity-level metrics; baseline computed on same gold set for fair comparison
9. **Predictor** (`inference/predictor.py`) — load checkpoint, text → character span list
10. **run.py** — CLI entry with `train` / `evaluate` / `predict` subcommands

### Critical Pitfalls

1. **Subword BIO alignment bug** — German compounds tokenize into many subwords; silent label corruption if offset mapping is not used correctly. Prevention: test BIO converter with round-trip verification (text → BIO → spans == original spans) before any training.

2. **LLM-to-LLM evaluation inflation** — training and test data from the same LLM distribution produces artifically high F1; model learns LLM writing patterns, not real regulatory text. Prevention: gold test set must be generated with a fixed seed, frozen to disk, and manually reviewed.

3. **Token-level vs entity-level evaluation** — token-level F1 is misleading because O tokens dominate. A model predicting all-O gets high token accuracy. Prevention: use seqeval entity-level metrics exclusively; never report token-level accuracy as a primary metric.

4. **Special token label masking** — [CLS], [SEP], [PAD] must receive label -100, not O. Assigning O dilutes the loss signal. Prevention: verify in BIO converter unit tests that all special tokens get -100.

5. **gbert-large OOM + wrong learning rate** — 1024d model can OOM on M1 16GB; single LR for backbone and head causes catastrophic forgetting. Prevention: differential LR (2e-5 backbone, 1e-4 head), start with batch_size=4 on M1, use gradient accumulation.

6. **IterableDataset warmup steps mismatch** — total training steps are unknown without `__len__`; warmup_ratio cannot be computed. Prevention: use fixed `warmup_steps` (not ratio); log actual LR at each step.

---

## Implications for Roadmap

Based on the combined research, the architecture imposes a clear dependency chain. The suggested phase structure follows that chain and front-loads validation of the highest-risk unknowns.

### Phase 1: Foundation + Regex Baseline
**Rationale:** Zero dependencies — can be built immediately. Config layer and regex baseline are fully independent and provide the comparison target that validates all later work. Building baseline first reveals how hard the problem actually is before investing in ML.
**Delivers:** Working project scaffold, device detection, seeding, YAML config, and a legitimate regex benchmark covering all major German legal reference patterns (§, Art., Abs., Tz., lit., Nr., Satz, Anhang, Verordnung).
**Addresses features:** YAML config, cross-platform device detection, regex baseline, fixed-seed reproducibility
**Avoids pitfalls:** Regex baseline too weak (Pitfall 8) — build it thoroughly from the start

### Phase 2: Data Pipeline + Gold Test Set
**Rationale:** This is the highest-risk phase. The BIO alignment bug is the most common source of silent failure in BERT NER systems. Online LLM generation is novel and needs validation before any model training. Gold set must be frozen before model training begins to prevent evaluation leakage.
**Delivers:** LLM data generator (OpenRouter/Gemini Flash), BIO converter with round-trip tests, IterableDataset, and a frozen gold_test_set.json with manually reviewed examples.
**Addresses features:** LLM-generated training data pipeline, character-span to BIO alignment, gold test set, LLM data quality checks
**Avoids pitfalls:** Subword BIO alignment bug (Pitfall 1), LLM-to-LLM evaluation inflation (Pitfall 2), special token label masking (Pitfall 4), max-sequence truncation (Pitfall 10), German Unicode normalization (Pitfall 13)
**Needs research-phase:** Yes — BIO alignment edge cases, OpenRouter async retry strategy, IterableDataset worker seeding

### Phase 3: Model + Training Loop
**Rationale:** Once data pipeline is validated, model and training are straightforward BERT fine-tuning with well-documented patterns. CRF is optional and stubbed — not wired in until Phase 4.
**Delivers:** RegulatoryNERModel (gbert-large + linear head), Trainer with differential LR + warmup + AMP, checkpoint saving, per-epoch metric logging.
**Addresses features:** Differential learning rates, mixed precision, gradient clipping, warmup schedule
**Avoids pitfalls:** OOM + wrong LR (Pitfall 6), IterableDataset warmup mismatch (Pitfall 7), fp16 NaN on MPS (Pitfall 11), O-class imbalance (Pitfall 12)
**Standard patterns:** Yes — canonical BERT fine-tuning; skip research-phase

### Phase 4: Evaluation Harness + Baseline Comparison
**Rationale:** With a trained model and frozen gold set, entity-level evaluation can now run. Baseline comparison produces the core PoC verdict: does ML beat regex on recall?
**Delivers:** Evaluator (seqeval entity-level P/R/F1), side-by-side ML vs regex comparison table, error analysis output (false positives / negatives to file).
**Addresses features:** Entity-level P/R/F1, error analysis output, regex baseline comparison
**Avoids pitfalls:** Token-level vs entity-level evaluation (Pitfall 3), evaluation on training-distribution data (Pitfall 6 variant)
**Standard patterns:** Yes — seqeval is well-documented; skip research-phase

### Phase 5: CLI Inference + PoC Packaging
**Rationale:** Final integration step. Inference depends on a trained model checkpoint. CLI is required for demo. Self-contained checkpoint schema ensures inference works without the training config file.
**Delivers:** Predictor (checkpoint → text → reference spans), run.py CLI with train/evaluate/predict subcommands, setup documentation (pip install + OPENROUTER_API_KEY).
**Addresses features:** CLI inference, instant runability (<10 min setup), self-contained checkpoint
**Standard patterns:** Yes — straightforward argparse/typer integration

### Phase 6: Optional Enhancements (CRF + Ensemble)
**Rationale:** Only warranted if Phase 4 evaluation reveals specific failure modes. CRF if invalid BIO sequences are frequent; ensemble if single-model variance is high. These are config-toggle features, not new infrastructure.
**Delivers:** Wired CRF layer (custom ~80 lines), optional bagging ensemble with epoch-based cache management.
**Addresses features:** Optional CRF layer, optional ensemble, span confidence scores (CRF-off mode)
**Avoids pitfalls:** CRF not enforcing BIO constraints (Pitfall 5), ensemble cache disk overflow (Pitfall 14)
**Gate condition:** Only enter this phase if Phase 4 results indicate need. Do not build speculatively.

### Phase Ordering Rationale

- **Baseline first** — reveals the actual difficulty of the problem before ML investment; defines the success bar
- **Data pipeline before model** — BIO alignment is the highest-risk failure mode; validating it early prevents wasted training runs on corrupted data
- **Gold set frozen before training** — prevents evaluation leakage; must be immutable by the time any model sees training data
- **CRF/ensemble last** — these are enhancements, not foundations; adding them early adds complexity without PoC value
- **IterableDataset pattern** — drives the decision to compute warmup_steps as fixed integer, not ratio; cascades into Phase 3 design

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2:** BIO alignment edge cases for German compound words and legal terminology; OpenRouter async rate limits and retry behavior; IterableDataset worker seeding to prevent duplicate samples across DataLoader workers

Phases with standard patterns (skip research-phase):
- **Phase 1:** Config and regex patterns are well-documented; regex for German legal syntax is straightforward
- **Phase 3:** BERT fine-tuning patterns are canonical and well-documented in HuggingFace tutorials
- **Phase 4:** seqeval library is standard; evaluation loop is straightforward
- **Phase 5:** CLI integration is trivial; argparse/typer are well-documented

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All libraries are well-established; version recommendations verified against current releases; MPS behavior caveats well-documented |
| Features | MEDIUM | Core NLP/ML features are well-established; German regulatory domain specifics are training-knowledge only; no public benchmark to validate against |
| Architecture | HIGH | BIO tokenization alignment is canonical HuggingFace pattern; IterableDataset and differential LR are standard PyTorch/BERT fine-tuning; CRF implementation has some variance across approaches |
| Pitfalls | HIGH | All identified pitfalls are well-known failure modes in BERT NER projects; some domain-specific risks (LLM data quality, regulatory text diversity) are inference rather than verified sources |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **German legal NER benchmarks:** No verified state-of-the-art numbers for this specific task (regulatory reference extraction). The PoC will establish its own baseline via the regex comparison. Not a blocker — the PoC is its own benchmark.
- **CRF marginal benefit on short spans:** Unclear whether CRF improves 2-5 token spans. Resolve empirically in Phase 4: measure invalid sequence rate before deciding to build Phase 6.
- **MPS bfloat16 stability:** Known improvement in PyTorch 2.x but edge cases remain. Handle by auto-detecting device and disabling AMP on MPS as a default-safe behavior, with opt-in via config.
- **LLM generation throughput:** OpenRouter/Gemini Flash rate limits and async retry behavior are not benchmarked. Phase 2 should include a throughput test before committing to online generation for training.
- **Gold set size vs LLM cost:** 100+ examples is the minimum credibility threshold. Cost and feasibility of generating a larger set should be validated in Phase 2.

---

## Sources

### Primary (HIGH confidence)
- HuggingFace Transformers documentation — BIO alignment pattern, token classification tutorial, DataCollatorForTokenClassification
- PyTorch documentation — IterableDataset, DataLoader worker seeding, CrossEntropyLoss ignore_index
- seqeval library documentation — entity-level NER evaluation
- Original BERT paper (Devlin et al., 2018) — differential learning rate rationale
- deepset/gbert-large model card — model architecture, GermEval14 benchmark numbers
- Lample et al., 2016 (LSTM-CRF for NER) — CRF transition constraint rationale

### Secondary (MEDIUM confidence)
- Brain patterns: `openrouter-structured-extraction.md`, `llm-judge-fewshot-learning.md` — LLM data generation approach
- Project context: `/Users/Admin/REG_ML/.planning/PROJECT.md` — scope, hardware targets, constraints
- Training knowledge (up to Aug 2025): German BERT fine-tuning practices, MPS mixed precision behavior

### Tertiary (LOW confidence)
- CRF effectiveness on short-span BIO tasks — inference from general NER literature; needs empirical validation within the PoC
- German regulatory reference type distribution — no public statistics; prompt diversity strategy is a hypothesis

---
*Research completed: 2026-03-13*
*Ready for roadmap: yes*
