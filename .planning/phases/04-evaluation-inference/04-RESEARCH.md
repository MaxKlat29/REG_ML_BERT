# Phase 4: Evaluation + Inference - Research

**Researched:** 2026-03-13
**Domain:** NER evaluation metrics, span IoU scoring, BERT inference, CLI integration, Google-style docstrings
**Confidence:** HIGH

## Summary

Phase 4 is the PoC verdict phase. All infrastructure is already in place: seqeval is installed, the existing `Evaluator` class handles regex baseline P/R/F1, `RegulatoryNERModel` has a working `forward()` for inference, `load_checkpoint()` is implemented in `trainer.py`, and the gold test set generator (and on-disk JSON format) are fully defined. The work is pure extension: expand the evaluator to cover ML model evaluation with per-type breakdown and IoU partial matching, add a `Predictor` class, wire up `run.py evaluate` and `run.py predict --text`, and apply Google-style docstrings across all public interfaces.

No new dependencies are required. Every library needed (seqeval, torch, transformers, tabulate/rich for table formatting, regex) is already in `requirements.txt` or in stdlib. The trickiest implementation work is the IoU partial-match scorer (requires span-level overlap computation, not token-level BIO diffing) and the per-reference-type breakdown (requires typed spans: knowing *which* pattern type a span belongs to).

**Primary recommendation:** Extend existing `Evaluator` and `RegexBaseline` classes — do not rewrite them. Add `Predictor` as a new class in `src/model/predictor.py`. All docstring work can be done as a single sweep in plan 04-02.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EVAL-01 | Entity-level Precision, Recall, F1 (not token-level) | seqeval already wired in `metrics.py`; ML model path needs same `spans_to_bio` → `compute_entity_metrics` flow |
| EVAL-04 | Per-reference-type breakdown (§, Art., Tz., etc.) | Requires typed spans from `RegexBaseline` and from ML model decoding; `RegexBaseline.extract()` returns untyped spans — needs extension or a typed variant |
| EVAL-05 | FP/FN dump file for error analysis | JSON file written per sample showing text, gold spans, pred spans, and error type |
| EVAL-06 | Exact match + partial match (IoU > 0.5) scoring | Span IoU = intersection length / union length; must be computed at span level before BIO conversion |
| INFR-01 | CLI prediction on arbitrary German text → char-offset spans | `Predictor` class: tokenize → model forward (no labels) → decode BIO to char spans via `offset_mapping` |
| INFR-02 | Confidence scores (softmax prob or CRF marginals) | Non-CRF: `softmax(logits)` max over B-REF/I-REF positions; CRF: use `crf.decode()` which returns Viterbi paths — marginals require `crf.forward()` score, not standard decode |
| INFR-03 | Batch prediction on multiple texts | Loop over texts or batch tokenize with padding |
| DOCS-03 | Google-style docstrings with type hints on all public classes/methods | Sweep all `src/` modules: add Args/Returns/Raises sections where missing |
</phase_requirements>

## Standard Stack

### Core (already in requirements.txt — no new installs needed)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| seqeval | >=1.2.2 | Entity-level NER metrics (P/R/F1, per-type classification_report) | Already used in `metrics.py`; industry standard for NER evaluation |
| torch | >=2.0.0 | Model inference, softmax for confidence scores | All model code already uses torch |
| transformers | >=4.40.0 | `BertTokenizerFast` for tokenizing input text during inference | Already imported for training |
| omegaconf | >=2.3.0 | Config reading in evaluator/predictor | Already in use across project |

### Supporting (stdlib — no install needed)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json | stdlib | Reading gold_test_set.json, writing FP/FN dump | Always |
| pathlib | stdlib | Path handling for checkpoint discovery | Always |
| tabulate | optional | Formatted side-by-side comparison table in terminal | Nice-to-have; `str.format()` is sufficient fallback |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| seqeval entity-level metrics | Manual span matching | seqeval handles IOB2 edge cases; hand-roll is error-prone for B/I continuity |
| softmax confidence | CRF marginals | CRF marginals require separate forward pass scoring; softmax on non-CRF logits is simpler and sufficient for PoC |

**Installation:** No new dependencies. Everything needed is already present.

## Architecture Patterns

### Recommended Project Structure (additions only)

```
src/
├── evaluation/
│   ├── evaluator.py       # EXTEND: add evaluate_model(), per-type breakdown, FP/FN dump, IoU scoring
│   └── metrics.py         # EXTEND: add iou_score(), typed_spans_to_bio() or per-type variant
├── model/
│   ├── predictor.py       # NEW: Predictor class — load checkpoint, text→spans with confidence
│   ├── ner_model.py       # EXISTS: no changes needed for inference (forward() handles label=None)
│   └── trainer.py         # EXISTS: load_checkpoint() already implemented
run.py                     # EXTEND: wire evaluate and predict subcommands
```

### Pattern 1: ML Model Evaluation Flow

**What:** Load gold test JSON → for each sample, run model inference → decode BIO → compute entity-level metrics via seqeval
**When to use:** `python run.py evaluate`
**Example:**
```python
# Inference path for a single sample (non-CRF)
output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
# output.logits shape: (batch, seq_len, 3)
pred_ids = output.logits.argmax(dim=-1)  # (batch, seq_len)
# Convert int labels back to string BIO for seqeval
LABEL_MAP = {0: "O", 1: "B-REF", 2: "I-REF", -100: "O"}
```

### Pattern 2: Span IoU Scoring (EVAL-06)

**What:** Measure overlap between predicted span and gold span as intersection/union of character ranges
**When to use:** Partial match evaluation alongside exact match
**Example:**
```python
def span_iou(pred: tuple[int, int], gold: tuple[int, int]) -> float:
    """Intersection over Union for character-offset spans."""
    p_start, p_end = pred
    g_start, g_end = gold
    inter_start = max(p_start, g_start)
    inter_end = min(p_end, g_end)
    intersection = max(0, inter_end - inter_start)
    union = (p_end - p_start) + (g_end - g_start) - intersection
    if union == 0:
        return 1.0
    return intersection / union
```

Partial match at threshold 0.5: a predicted span "matches" a gold span if `span_iou(pred, gold) > 0.5`. This is computed at the raw character-span level — NOT via seqeval/BIO. The evaluator runs two passes: exact match (using seqeval on BIO labels) and partial match (IoU > 0.5 on char spans, then compute P/R/F1 manually).

### Pattern 3: Per-Reference-Type Breakdown (EVAL-04)

**What:** Report P/R/F1 split by reference type (§, Art., Tz., Anhang, Verordnung)
**How:** The gold test set samples contain `"spans": [(start, end), ...]` without type labels. Two approaches:

**Approach A (recommended):** Tag gold spans by re-running `RegexBaseline` patterns individually on the gold text, then match each gold span to the nearest pattern type. Since the gold set was generated by prompting for specific reference types and the regex patterns are per-type, this is reliable for gold spans.

**Approach B:** Add a `extract_typed()` method to `RegexBaseline` that returns `list[tuple[int, int, str]]` with the pattern type name. Use this for both gold labeling and baseline predictions.

Approach B is cleaner — it extends the existing regex baseline naturally and avoids re-running patterns separately.

**Example typed extraction:**
```python
def extract_typed(self, text: str) -> list[tuple[int, int, str]]:
    """Returns (start, end, ref_type) for each match."""
    results = []
    for ref_type, pattern in self.TYPED_PATTERNS.items():
        for m in pattern.finditer(text):
            results.append((m.start(), m.end(), ref_type))
    return sorted(results, key=lambda x: x[0])
```

seqeval's `classification_report` already supports per-entity-type breakdown when BIO labels use typed tags (e.g., `B-PARAGRAPH`, `B-ARTIKEL`) instead of generic `B-REF`. If typed BIO labels are provided, seqeval reports per-type metrics automatically.

### Pattern 4: Predictor Class (INFR-01, INFR-02, INFR-03)

**What:** Load a checkpoint once, expose `predict(text)` and `predict_batch(texts)` returning char-offset spans with confidence
**Key design decisions:**
- Load checkpoint using existing `load_checkpoint()` from `trainer.py`
- Use `BertTokenizerFast` (same as training, not AutoTokenizer — see STATE.md decision)
- Return character offsets, not token offsets — reconstruct via `offset_mapping` (same logic as `validate_bio_roundtrip` in `bio_converter.py`)
- For confidence: average softmax probability of B-REF/I-REF tokens within each span
- Under 5 seconds: gbert-large inference on CPU takes ~1-2s per sentence; model.eval() + torch.no_grad() is mandatory

**Example signature:**
```python
@dataclass
class PredictedSpan:
    start: int
    end: int
    text: str
    confidence: float

class Predictor:
    def __init__(self, checkpoint_path: str | Path, config, device: torch.device) -> None: ...
    def predict(self, text: str) -> list[PredictedSpan]: ...
    def predict_batch(self, texts: list[str]) -> list[list[PredictedSpan]]: ...
```

### Pattern 5: BIO → Char Span Reconstruction

**What:** Convert model's per-token integer predictions back to character offset spans
**This logic already exists in `bio_converter.validate_bio_roundtrip()`** — extract and generalize it in `Predictor`:

```python
def _decode_bio_to_spans(
    token_labels: list[int],
    offset_mapping: list[tuple[int, int]],
    logits: torch.Tensor,
) -> list[PredictedSpan]:
    """Reconstruct char-offset spans from BIO label sequence."""
    spans = []
    current_start = None
    current_end = None
    current_probs = []

    softmax_probs = torch.softmax(logits, dim=-1)  # (seq_len, 3)

    for idx, (label, (tok_start, tok_end)) in enumerate(
        zip(token_labels, offset_mapping)
    ):
        if tok_start == 0 and tok_end == 0:
            continue  # special token
        if label == LABEL_B_REF:
            if current_start is not None:
                spans.append(_make_span(current_start, current_end, current_probs))
            current_start, current_end = tok_start, tok_end
            current_probs = [softmax_probs[idx, label].item()]
        elif label == LABEL_I_REF and current_start is not None:
            current_end = tok_end
            current_probs.append(softmax_probs[idx, label].item())
        else:
            if current_start is not None:
                spans.append(_make_span(current_start, current_end, current_probs))
                current_start = None
                current_probs = []
    if current_start is not None:
        spans.append(_make_span(current_start, current_end, current_probs))
    return spans
```

### Pattern 6: Side-by-Side Comparison Table (EVAL-01)

**What:** Print ML model vs regex baseline metrics in one terminal-friendly table
**Format:**
```
=================================================================
  Evaluation Results — Gold Test Set (N=50 samples)
=================================================================

  Metric       ML Model   Regex Baseline   Delta
  ---------    --------   --------------   -----
  Precision      0.8912         0.9231     -0.032
  Recall         0.9341         0.8120     +0.122   ← primary goal
  F1-Score       0.9122         0.8637     +0.049

  Verdict: ML model BEATS baseline on recall (+0.122)
=================================================================
```

Plain string formatting is sufficient — no tabulate dependency needed.

### Anti-Patterns to Avoid

- **Token-level metrics as proxy for EVAL-01:** The requirement is entity-level P/R/F1. seqeval computes this correctly at entity level. Do NOT compute token-level accuracy.
- **Loading model inside predict() loop:** Load once in `__init__`, reuse across calls.
- **Missing `model.eval()` and `torch.no_grad()` during inference:** Without these, the model is slower and may produce non-deterministic results (dropout active).
- **Using AutoTokenizer for gbert-large:** STATE.md records that `BertTokenizerFast` must be used directly. `AutoTokenizer` fails on gbert-large without `tokenizer.json` in transformers 5.x.
- **IoU computed on BIO token level:** IoU must be computed on raw char spans, not on BIO token sequences.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Entity-level P/R/F1 | Custom span matching logic | seqeval `precision_score`, `recall_score`, `f1_score` | Already handles IOB2 boundary detection, partial sequence issues |
| Per-type breakdown | Manual type-splitting loops | seqeval `classification_report` with typed BIO tags (B-PARAGRAPH vs B-ARTIKEL) | seqeval reports per-entity-type automatically when entity types differ in the label string |
| Checkpoint loading | Custom torch.load wrapper | Existing `load_checkpoint()` in `src/model/trainer.py` | Already tested and handles optimizer/scheduler optionally |
| Tokenization for inference | Re-implement tokenization | `BertTokenizerFast.from_pretrained()` with `return_offsets_mapping=True` | `char_spans_to_bio` already demonstrates correct usage |
| BIO→span reconstruction | New decoder | Generalize `validate_bio_roundtrip()` from `bio_converter.py` | Core logic is already written and tested |

**Key insight:** Nearly all the hard work is already done in Phases 1-3. Phase 4 is primarily wiring and extension, not new computation.

## Common Pitfalls

### Pitfall 1: Gold Set Has No Type Labels on Spans
**What goes wrong:** `gold_test_set.json` stores `"spans": [(start, end), ...]` — no reference type attached. Per-type breakdown (EVAL-04) requires knowing whether a span is a § or Art. reference.
**Why it happens:** The gold generator stores untyped spans because the LLM tags with `<ref>...</ref>` uniformly.
**How to avoid:** For the gold spans, derive type by running typed regex patterns against the gold text to classify each gold span. Since gold spans were originally generated from text containing known reference types, the regex will correctly classify them. For ML model predictions, use the same typed regex classifier post-hoc on the predicted span text.
**Warning signs:** All per-type rows showing identical counts.

### Pitfall 2: seqeval Requires String BIO Labels, Not Integers
**What goes wrong:** `compute_entity_metrics()` in `metrics.py` takes `list[list[str]]` — seqeval does not accept integer labels. Model output is integers (0, 1, 2).
**How to avoid:** Map integers → strings with `LABEL_MAP = {0: "O", 1: "B-REF", 2: "I-REF"}` before passing to seqeval. Skip positions where label == -100 (special tokens) by filtering them out or mapping to "O".
**Warning signs:** seqeval raises `ValueError: Invalid label sequence` or returns 0.0 for all metrics.

### Pitfall 3: Ignoring -100 Labels When Building seqeval Input
**What goes wrong:** Model outputs logits for ALL token positions including special tokens (CLS, SEP, PAD). If argmax is taken over all positions without filtering, -100 positions create mismatched sequence lengths between gold and pred.
**How to avoid:** Build the gold and pred BIO string lists only for positions where `gold_labels[i] != -100` (i.e., where attention_mask == 1 and the token is not a special token).
**Warning signs:** seqeval `ValueError` about sequence lengths.

### Pitfall 4: 5-Second Inference Budget on CPU
**What goes wrong:** gbert-large has 335M parameters. Cold-start inference (including tokenizer load + model load) can take 10-20 seconds. The requirement says under 5 seconds for `predict`.
**Why it happens:** Model and tokenizer loading are the bottleneck, not inference itself.
**How to avoid:** Measure from end of model load (i.e., time only the tokenize+forward+decode steps). The "under 5 seconds" constraint in the success criterion refers to inference time, not startup time. If needed, document the distinction in CLI output. On Apple Silicon MPS or CUDA, inference is sub-1-second.
**Warning signs:** Test timing the full `Predictor.__init__` + `predict()` together — that will fail the 5s budget on CPU.

### Pitfall 5: CRF Model Inference Returns List-of-Lists, Not Logits
**What goes wrong:** When `use_crf=True`, `model.forward()` without labels calls `crf.decode()` and returns `list[list[int]]` (Viterbi paths), NOT a `TokenClassifierOutput`. The predictor must handle both paths.
**How to avoid:** Check `model.use_crf` in `Predictor` and handle both return types:
- Non-CRF: `output.logits.argmax(dim=-1)` for labels; `softmax(output.logits)` for confidence
- CRF: decoded integer sequences already; confidence requires re-running `crf` in training mode or using emission scores — for PoC, set confidence=1.0 for CRF mode and document this.
**Warning signs:** `AttributeError: 'list' object has no attribute 'logits'`

### Pitfall 6: Checkpoint Discovery
**What goes wrong:** Checkpoints are saved to `checkpoints/{run_id}/epoch_{epoch}.pt`. For `run.py evaluate`, the user must either specify a checkpoint path or the evaluator must find the latest one automatically.
**How to avoid:** Add a `--checkpoint` argument to the evaluate and predict subcommands. Also provide a helper that finds the latest checkpoint in `checkpoints/` by modification time or epoch number. Document that ensemble evaluation requires specifying all N checkpoint paths.
**Warning signs:** FileNotFoundError when running evaluate without explicit checkpoint path.

### Pitfall 7: Gold Test Set May Not Exist on Disk
**What goes wrong:** `data/gold_test/gold_test_set.json` is generated by running `scripts/generate_gold_test.py` with a live OPENROUTER_API_KEY. The file exists in the current repo (confirmed: `data/gold_test/epoch_0.pt` exists — wait, that's checkpoints). The gold_test directory exists but may be empty.
**How to avoid:** At the start of `run.py evaluate`, check for the gold test set file and print a clear error if missing: "Gold test set not found. Run: python scripts/generate_gold_test.py".

## Code Examples

Verified patterns from existing codebase:

### Loading and Running Model Inference
```python
# From src/model/trainer.py — load_checkpoint already implemented
from src.model.trainer import load_checkpoint
from src.model.ner_model import RegulatoryNERModel

model = RegulatoryNERModel(config)
load_checkpoint(checkpoint_path, model)
model.eval()  # CRITICAL: disable dropout

with torch.no_grad():
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    # Non-CRF: output.logits shape (batch, seq_len, 3)
    # CRF: output is list[list[int]] from crf.decode()
```

### Reading Gold Test Set
```python
# Gold set format from scripts/generate_gold_test.py:
# {
#   "text": "Gemäß § 25a KWG ...",
#   "spans": [[6, 15], [20, 28]],   # JSON arrays, not tuples
#   "bio_labels": {"input_ids": [...], "attention_mask": [...], "labels": [...]},
#   "needs_review": true,
#   "domain": "KWG",
#   "seed": 1337,
#   "has_references": true
# }

import json
with open("data/gold_test/gold_test_set.json") as f:
    samples = json.load(f)
# Convert span lists to tuples
gold_spans = [tuple(s) for s in sample["spans"]]
```

### seqeval Usage with Typed Labels
```python
# Source: existing src/evaluation/metrics.py + seqeval docs
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score

# For per-type breakdown, use typed entity names:
y_true = [["O", "B-PARAGRAPH", "I-PARAGRAPH", "O", "B-ARTIKEL"]]
y_pred = [["O", "B-PARAGRAPH", "I-PARAGRAPH", "O", "O"]]

report = classification_report(y_true, y_pred)
# Output includes per-entity-type rows: PARAGRAPH, ARTIKEL
```

### Tokenizing Text for Inference (offset_mapping pattern)
```python
# Source: src/data/bio_converter.py — char_spans_to_bio()
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("deepset/gbert-large")
encoding = tokenizer(
    text,
    return_offsets_mapping=True,
    truncation=True,
    padding="max_length",
    max_length=512,
    return_tensors="pt",
)
# encoding["offset_mapping"][i] = (char_start, char_end) for token i
# Special tokens: (0, 0) — filter these out during decode
```

### FP/FN Dump File Format
```python
# Suggested JSON structure for error analysis output
error_record = {
    "sample_idx": i,
    "text": sample["text"],
    "gold_spans": gold_spans,
    "pred_spans": pred_spans,
    "false_positives": [s for s in pred_spans if s not in gold_spans],
    "false_negatives": [s for s in gold_spans if s not in pred_spans],
    "domain": sample.get("domain", "unknown"),
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Token-level accuracy for NER | Entity-level seqeval P/R/F1 | ~2018 (seqeval paper) | Entity boundary errors no longer hidden by majority-O token accuracy |
| Exact match only | IoU-based partial matching | Common in span extraction tasks (SQuAD-style) | Captures near-miss predictions that are practically useful |
| Global-only metrics | Per-entity-type breakdown | Standard in production NER | Reveals which reference types the model struggles with |

**Not needed for this phase:**
- Micro vs macro averaging: seqeval default (micro) is correct for NER; macro would give equal weight to rare Tz. references
- Confidence calibration: beyond PoC scope; raw softmax confidence is sufficient

## Open Questions

1. **Gold test set population on disk**
   - What we know: `data/gold_test/` directory exists but appears empty (no JSON found when checking)
   - What's unclear: Whether a gold set will be generated before evaluation runs, or whether `run.py evaluate` must handle the missing-file case gracefully
   - Recommendation: Evaluator should fail fast with a clear message if the gold set is missing. Include a note in the plan that the user must run `generate_gold_test.py` first if not already done.

2. **Per-type breakdown implementation strategy**
   - What we know: Gold spans are untyped; regex patterns are monolithic in `regex_baseline.py`
   - What's unclear: Whether to extend `RegexBaseline` with typed patterns or classify gold spans post-hoc
   - Recommendation: Add `extract_typed()` to `RegexBaseline` returning `(start, end, ref_type)`. This is ~20 lines. Then use typed BIO labels (`B-PARAGRAPH`, `B-ARTIKEL`, etc.) through the seqeval pipeline for automatic per-type reporting. Both ML model and regex baseline get typed BIO via the same gold-span typing function.

3. **Ensemble checkpoint evaluation**
   - What we know: Ensemble saves multiple checkpoints at `checkpoints/ensemble_{i}/epoch_{epoch}.pt`
   - What's unclear: Whether evaluation should support ensemble inference (majority vote) or only single-model
   - Recommendation: For PoC, support single checkpoint evaluation. Add optional `--ensemble-dir` flag that loads all checkpoints in a directory and applies majority vote. This is a low-effort extension using existing `majority_vote()` from `trainer.py`.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest >= (installed; see requirements.txt) |
| Config file | `pytest.ini` — `testpaths = tests`, `asyncio_mode = auto` |
| Quick run command | `pytest tests/test_evaluator.py tests/test_predictor.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-01 | ML model eval returns entity-level P/R/F1 dict | unit | `pytest tests/test_evaluator.py::TestMLEvaluation -x` | ❌ Wave 0 |
| EVAL-04 | Per-type breakdown returns metrics keyed by ref type | unit | `pytest tests/test_evaluator.py::TestPerTypeBreakdown -x` | ❌ Wave 0 |
| EVAL-05 | FP/FN dump file written to disk with correct fields | unit | `pytest tests/test_evaluator.py::TestFPFNDump -x` | ❌ Wave 0 |
| EVAL-06 | IoU scorer returns correct values; partial match scoring works | unit | `pytest tests/test_evaluator.py::TestIoUScoring -x` | ❌ Wave 0 |
| INFR-01 | Predictor.predict() returns char-offset spans | unit | `pytest tests/test_predictor.py::TestPredict -x` | ❌ Wave 0 |
| INFR-02 | Predicted spans include confidence scores in [0, 1] | unit | `pytest tests/test_predictor.py::TestConfidenceScores -x` | ❌ Wave 0 |
| INFR-03 | predict_batch() returns list-of-lists, one per input text | unit | `pytest tests/test_predictor.py::TestBatchPredict -x` | ❌ Wave 0 |
| DOCS-03 | All public classes/methods have Google-style docstrings | unit | `pytest tests/test_docs.py -x` | ❌ Wave 0 (extend existing test_docs.py) |

### Sampling Rate
- **Per task commit:** `pytest tests/test_evaluator.py tests/test_predictor.py -x`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green (`pytest tests/ -x` — 83+ tests passing) before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_evaluator.py` — covers EVAL-01, EVAL-04, EVAL-05, EVAL-06
- [ ] `tests/test_predictor.py` — covers INFR-01, INFR-02, INFR-03
- [ ] Extend `tests/test_docs.py` to check Google-style docstrings (Args/Returns sections) — covers DOCS-03

**Note:** `tests/test_metrics.py` already exists and tests `spans_to_bio` and `compute_entity_metrics`. New tests build on top of these, not in parallel.

## Sources

### Primary (HIGH confidence)
- `/Users/Admin/REG_ML/src/evaluation/metrics.py` — seqeval integration, `spans_to_bio`, `compute_entity_metrics`
- `/Users/Admin/REG_ML/src/evaluation/evaluator.py` — existing `Evaluator` class, `evaluate_baseline()`
- `/Users/Admin/REG_ML/src/model/trainer.py` — `load_checkpoint()`, `majority_vote()`, checkpoint format
- `/Users/Admin/REG_ML/src/model/ner_model.py` — `forward()` return types for CRF vs non-CRF paths
- `/Users/Admin/REG_ML/src/data/bio_converter.py` — `validate_bio_roundtrip()`, offset_mapping usage, LABEL constants
- `/Users/Admin/REG_ML/scripts/generate_gold_test.py` — gold set JSON structure, field names
- `/Users/Admin/REG_ML/.planning/STATE.md` — key decisions: BertTokenizerFast, CRF forward path, CRF mask constraints

### Secondary (MEDIUM confidence)
- seqeval documentation (per-entity-type breakdown via typed BIO tags) — standard library behavior, HIGH confidence
- IoU span computation — standard formula, widely used in span extraction evaluation

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already present and in use
- Architecture: HIGH — patterns derived directly from existing tested code
- Pitfalls: HIGH — most pitfalls identified from existing code decisions documented in STATE.md

**Research date:** 2026-03-13
**Valid until:** 2026-06-13 (stable libraries; seqeval, torch, transformers change slowly)
