# Feature Landscape

**Domain:** Regulatory Reference Extraction — German Legal NER (BIO Token Classification)
**Project:** REG_ML PoC
**Researched:** 2026-03-13
**Confidence note:** Web search unavailable. Findings based on training knowledge (cutoff Aug 2025) plus project context. Confidence flagged per section.

---

## Table Stakes

Features users (and evaluators of this PoC) expect to be present. Missing = the PoC cannot be taken seriously as a viable product signal.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| BIO token classification (O, B-REF, I-REF) | Standard NER framing; reviewers will expect this as baseline approach | Low | Already decided — gbert-large + linear head |
| Entity-level Precision / Recall / F1 | Without these, you cannot claim the model "works"; recall is the primary KPI | Low | seqeval library handles this; token-level F1 is a trap — use entity-level only |
| Regex baseline as benchmark | Any ML claim is meaningless without a rule-based baseline to beat; PoC credibility depends on this | Medium | Regex for §, Artikel, Abs., Nr., Tz., lit., Satz patterns — German legal syntax is relatively regular |
| CLI inference: text in → spans out | Evaluators must be able to run it; no CLI = no demo | Low | Simple argparse or typer interface |
| YAML config (no hardcoded hyperparameters) | Reproducibility and A/B testing require config-driven runs | Low | Single config.yaml: lr, epochs, batch_size, model_name, crf_enabled, ensemble_enabled |
| Fixed-seed reproducibility | Results that cannot be reproduced are not PoC results | Low | Seed all: PyTorch, NumPy, LLM generation (temperature=0 or seeded prompt) |
| Cross-platform device detection (MPS / CUDA / CPU) | Two developers, two hardware targets; if one cannot run it, it breaks collaboration | Low | Auto-detect via torch.device("mps") / torch.cuda.is_available() / fallback |
| LLM-generated training data pipeline | No annotated German regulatory corpus exists publicly; this is the only viable data strategy | Medium | OpenRouter + Gemini Flash; ~60/40 positive/negative split; online generation via IterableDataset |
| Gold test set (LLM-generated + manually reviewed) | PoC evaluation must be on held-out data the model never trained on; manual review prevents gaming | Medium | Generate once, persist as JSON/JSONL; do not regenerate between runs |
| Character-span to BIO token alignment | Required bridge between LLM output (character spans) and model input (tokenized BIO labels) | Medium | Tricky with subword tokenization — first subword gets B/I, subsequent get I; off-by-one bugs are common |
| Training loop with differential learning rates | Fine-tuning BERT requires lower LR for pretrained layers vs. new head; single LR causes catastrophic forgetting | Low-Medium | 2-group param groups: encoder vs. classification head |
| Mixed precision training | Required for practical training speed on both MPS and CUDA | Low | torch.autocast; note MPS support for fp16 is partial — bfloat16 preferred on M1 |
| pip install + env var setup (instant runability) | PoC that requires more than 10 minutes setup to run is not evaluated | Low | requirements.txt or pyproject.toml; single OPENROUTER_API_KEY env var |

---

## Differentiators

Features that set this PoC apart from a minimal implementation. Not required for basic viability, but valued for demonstrating rigor and production-readiness signal.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Optional CRF layer | Enforces valid BIO transition probabilities (no I-REF after O without B-REF); measurable recall/precision effect | Medium | torchcrf or pytorch-crf; config toggle; A/B comparison table in results |
| Optional bagging ensemble | Multiple models trained on different LLM-generated batches; reduces variance, improves robustness | High | Config toggle; requires cached data strategy for reproducibility; adds significant training time |
| Reference type breakdown in evaluation | Beyond aggregate F1 — per-type metrics (§, Artikel, Tz., etc.) reveal which reference types the model handles well vs. misses | Medium | Requires typed annotations from LLM OR post-hoc regex classification of extracted spans |
| LLM data generation quality checks | Verifying generated samples are correct before training prevents garbage-in/garbage-out | Medium | Spot-check: re-extract spans from generated text, verify char offsets match; log percent invalid samples |
| Warmup schedule + cosine decay | Better convergence than flat LR; standard practice for BERT fine-tuning | Low | HuggingFace get_linear_schedule_with_warmup or manual implementation |
| Error analysis output | After evaluation, dump false positives and false negatives to a file for qualitative review | Low | High signal-to-noise for improving prompts or model; invaluable for PoC writeup |
| Configurable negative sample ratio | 60/40 pos/neg is a hypothesis — being able to tune this reveals model sensitivity | Low | Single YAML param; no code changes needed |
| Span confidence scores (CRF-off mode) | Softmax probability on B-REF token as proxy confidence; useful for downstream filtering | Low-Medium | Not standard output but easy to add to inference CLI |
| Sample diversity controls in LLM prompts | Regulatory domains (BGB, KWG, DSGVO, etc.) should be sampled proportionally; otherwise model overfits to §-heavy financial law | Medium | Domain rotation in prompt templates; track per-domain sample counts |
| Gradient clipping | Prevents exploding gradients, especially with CRF layer; standard defensive practice | Low | torch.nn.utils.clip_grad_norm_ with max_norm=1.0 |

---

## Anti-Features

Features to explicitly NOT build for this PoC. Building them would waste time, add complexity, and dilute focus from the core PoC question.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| REST API / FastAPI serving layer | Adds DevOps surface area; not the PoC question | CLI inference is sufficient for evaluation |
| Frontend / UI / visualization dashboard | UI work is weeks of effort; PoC is evaluated by engineers, not end users | Print results to stdout; optionally write JSON |
| Docker / containerization | Adds complexity; cross-platform already solved via device detection | Document pip install + env var setup clearly |
| MLflow / full experiment tracking | Over-engineering for PoC; wandb optional is the right call | Print per-epoch metrics to stdout + optional CSV log |
| Multi-label / nested entity support | German legal refs are non-overlapping spans; nesting adds modeling complexity for no PoC benefit | Flat BIO is sufficient; note nested refs as future work |
| Entity linking / resolution | Mapping "§ 25a KWG" to a canonical law database entry is a separate, harder problem | Extract spans only; link in a future phase |
| Custom tokenizer training | gbert-large tokenizer is already trained on German text; retraining is not justified | Use tokenizer as-is with truncation and padding |
| Real-time / streaming inference | PoC operates on document batches, not live streams | Batch CLI inference is sufficient |
| Hyperparameter search (Optuna / Ray Tune) | PoC needs one good configuration, not the optimal one | Manual grid: 3-5 config variants at most |
| CI/CD pipeline | No deployment target; automated tests would require a gold dataset which is PoC output | Manual training + eval run is fine |
| Multi-GPU / distributed training | gbert-large on single GPU/MPS is sufficient for PoC dataset sizes | Single-device training only |
| Automatic data labeling UI / annotation tool | LLM generates labeled data; no human annotation interface needed | LLM prompt + manual spot-check of gold set |

---

## Feature Dependencies

```
LLM data generation pipeline
  -> Character-span to BIO token alignment
    -> Training loop
      -> Entity-level evaluation
        -> Baseline comparison (regex benchmark)
          -> PoC verdict (ML beats regex on recall)

YAML config
  -> All tunable features (CRF toggle, ensemble toggle, LR, epochs, etc.)

Gold test set
  -> Entity-level evaluation (must be separate from training data)
  -> LLM data generation pipeline (reuses same generation infrastructure)

CLI inference
  -> Trained model checkpoint (training loop must complete)
  -> Optionally: span confidence scores

Optional CRF layer
  -> Training loop (added loss term)
  -> Entity-level evaluation (compare CRF vs. no-CRF via YAML toggle)

Optional ensemble
  -> Multiple training runs with cached data
  -> Entity-level evaluation (ensemble vs. single model)
```

---

## MVP Recommendation

For the PoC to answer its core question ("does ML beat regex for German legal reference extraction?"), prioritize in this order:

1. **Regex baseline** — implement first; this is your comparison target and reveals what you are actually trying to beat
2. **LLM data generation + BIO alignment** — the data strategy is the hardest unsolved problem; validate it early
3. **Gold test set** — generate and manually review before any training; prevents evaluation leakage
4. **Training loop (no CRF, no ensemble)** — simple model first; add complexity only if simple model fails
5. **Entity-level evaluation + error analysis** — core evaluation harness; error analysis is free and high-value
6. **CLI inference** — required for demo; add last since it depends on a trained model

**Defer for post-PoC:**
- CRF layer: add only if entity-level F1 plateaus and BIO violations are frequent in error analysis
- Ensemble: add only if single-model variance is high across runs
- Reference type breakdown: valuable but requires typed LLM annotations; add if time permits
- Span confidence scores: useful for downstream product but not for PoC evaluation

---

## PoC Success Criteria

The PoC is successful when all of the following hold:

| Criterion | Target | Notes |
|-----------|--------|-------|
| Entity-level Recall (ML model) | Greater than regex baseline recall | Primary claim; missing references is the failure mode |
| Entity-level F1 (ML model) | Greater than regex baseline F1 | Secondary; should not sacrifice too much precision |
| Gold test set size | >= 100 examples (mix of positive/negative) | Minimum for statistical credibility |
| Reproducibility | Same config + seed -> same metrics within 0.5% | Required for PoC credibility |
| Setup time | < 10 minutes from clone to first training run | Required for colleague onboarding |

---

## Research Gaps

The following areas could not be verified against current literature (web search unavailable):

- **State-of-the-art German legal NER benchmarks**: Training knowledge suggests GermEval datasets exist but do not cover regulatory reference spans specifically. Confidence: LOW.
- **gbert-large vs. newer German models**: Training data suggests gbert-large is the right call, but newer instruction-tuned models may have shifted the landscape. Project has already decided on gbert-large. Confidence: MEDIUM.
- **CRF effectiveness on short-span BIO tasks**: Training knowledge suggests CRF helps most when spans are long or nested; for 2-5 token spans like legal references, benefit may be marginal. Needs empirical validation within the PoC. Confidence: LOW.
- **MPS mixed precision behavior with bfloat16**: Known instability in early MPS implementations; current PyTorch state (>=2.0) has improved this but edge cases remain. Confidence: MEDIUM.

---

## Sources

- Project context: `/Users/Admin/REG_ML/.planning/PROJECT.md`
- Brain patterns: `openrouter-structured-extraction.md`, `llm-judge-fewshot-learning.md`
- Training knowledge: HuggingFace Transformers documentation, seqeval library, PyTorch AMP, German BERT fine-tuning practices (up to Aug 2025)
- Confidence: MEDIUM overall — core NLP/ML claims are well-established; German regulatory domain specifics are training-knowledge only
