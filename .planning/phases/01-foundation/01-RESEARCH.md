# Phase 1: Foundation — Research

**Researched:** 2026-03-13
**Domain:** Python project scaffold, OmegaConf config layer, device detection, regex NER baseline, seqeval evaluation
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CONF-01 | All hyperparameters controlled via single YAML config file (no hardcoded values) | OmegaConf load + structured dataclass schema pattern documented |
| CONF-02 | Config supports CLI overrides (e.g., `--model.use_crf=false`) | OmegaConf.from_cli() + OmegaConf.merge() pattern documented |
| CONF-03 | Seeds are set for PyTorch, NumPy, and LLM generation for full reproducibility | Standard torch/numpy seed API documented; LLM seed is config value only in Phase 1 |
| CONF-04 | Project runs after pip install + single env var (OPENROUTER_API_KEY), setup under 10 minutes | python-dotenv load_dotenv() pattern; requirements.txt structure documented |
| EVAL-02 | Regex baseline extracts references using pattern matching (§, Artikel, Abs., Anhang, Verordnung, etc.) | jura_regex + german-legal-reference-parser patterns researched; all target reference types identified |
| EVAL-03 | Regex baseline is evaluated with same metrics as ML model for direct comparison | seqeval classification_report with BIO label conversion documented; wrapper pattern defined |
| DOCS-01 | README.md with project description, setup guide, usage examples, and Mermaid pipeline diagram | Mermaid syntax documented; content scope defined |
| DOCS-02 | .env.example with OPENROUTER_API_KEY placeholder | Simple file, one-liner |
</phase_requirements>

---

## Summary

Phase 1 is a pure scaffold and baseline phase — no ML training, no LLM calls. It lays the foundation that all subsequent phases build on. The work divides cleanly into two plans: (1) project scaffold, config layer, device detection, and seed setup; (2) regex baseline with seqeval evaluation wrapper, plus README and .env.example.

The technical stack for this phase is well-established and carries HIGH confidence throughout. OmegaConf 2.3 is the right config tool: it provides YAML loading, dot-notation access, CLI overrides via `OmegaConf.from_cli()`, and optional structured config validation via `@dataclass`. The pattern is `load YAML → merge with CLI args → access via dot notation`. No Hydra needed. The regex baseline covers all required German legal reference types; existing open-source projects (jura_regex, german-legal-reference-parser) confirm the complete pattern set. seqeval is the correct entity-level evaluation library — token-level metrics are explicitly wrong for this task. Device detection uses the standard three-way check: CUDA → MPS → CPU, using `torch.cuda.is_available()` and `torch.backends.mps.is_available()`.

The project is currently a near-empty greenfield (only `.gitignore` and `.env` exist). The full directory tree must be created from scratch. The `data/gold_test/` and `data/cache/` directories contain only `.gitkeep` files. `requirements.txt` is the install surface — no `pyproject.toml` needed for this PoC.

**Primary recommendation:** Build scaffold and config layer in Plan 01-01 (independent of regex work), then build regex + evaluation + docs in Plan 01-02. The two plans are independent and can be reviewed separately.

---

## Standard Stack

### Core (Phase 1 only)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| omegaconf | >= 2.3 | YAML config with CLI overrides and dot-notation access | Standard ML config layer; used by Hydra; handles merge semantics cleanly |
| torch | >= 2.0 | Device detection and seed setting only in Phase 1 | Native MPS support since 2.0; required for device detection API |
| seqeval | >= 1.2.2 | Entity-level NER Precision/Recall/F1 | Standard HuggingFace NER evaluation library; entity-level not token-level |
| python-dotenv | latest | Load OPENROUTER_API_KEY from .env | Standard 12-factor app env loading |
| numpy | >= 1.24 | Seed setting only in Phase 1 | Required for reproducibility alongside torch |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pyyaml | >= 6.0 | OmegaConf dependency for YAML parsing | Automatic dependency; do not call directly |
| re | stdlib | Regex engine for baseline | Standard library; no install needed |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| omegaconf | pydantic + typer | Pydantic gives stricter type validation but heavier; OmegaConf is simpler for YAML-driven ML configs |
| omegaconf | configparser | configparser has no hierarchical config or CLI merge; not suitable |
| python-dotenv | manual os.environ | os.getenv() works but dotenv handles local dev .env file transparently |
| re (stdlib) | regex (PyPI) | regex module supports overlapping matches (`overlapped=True`) for §§ chains; jura_regex requires it; stdlib re does not. **Use the `regex` PyPI package** if §§ multi-section references need to be captured correctly |

**Installation:**
```bash
pip install omegaconf torch seqeval python-dotenv numpy pyyaml regex
```

Full requirements.txt for Phase 1 (MVP subset — more libraries added in later phases):
```
omegaconf>=2.3.0
torch>=2.0.0
seqeval>=1.2.2
python-dotenv
numpy>=1.24.0
pyyaml>=6.0
regex
```

---

## Architecture Patterns

### Recommended Project Structure
```
regulatory-ref-extraction/
├── config/
│   └── default.yaml          # All hyperparameters; single source of truth
├── src/
│   ├── __init__.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py         # seqeval wrapper: spans → BIO → entity-level P/R/F1
│   │   ├── regex_baseline.py  # RegexBaseline class: text → span list
│   │   └── evaluator.py       # Evaluator: runs baseline, computes metrics, formats report
│   └── utils/
│       ├── __init__.py
│       └── config.py          # load_config(argv) → OmegaConf DictConfig
├── scripts/
│   ├── evaluate.py            # CLI entry: load config, run evaluator, print report
│   └── predict.py             # CLI entry skeleton (full implementation in Phase 4)
├── data/
│   ├── gold_test/
│   │   └── .gitkeep
│   └── cache/
│       └── .gitkeep
├── requirements.txt
├── README.md
└── .env.example
```

### Pattern 1: OmegaConf Config Loading with CLI Overrides

**What:** Load `default.yaml`, merge with CLI args passed as `key=value` or `key.nested=value`, return a single DictConfig object.
**When to use:** Every entry point (`scripts/evaluate.py`, `scripts/predict.py`, later `run.py`) calls this once at startup.

```python
# Source: https://omegaconf.readthedocs.io/en/2.3_branch/usage.html
# src/utils/config.py

import sys
from omegaconf import OmegaConf, DictConfig

def load_config(config_path: str = "config/default.yaml", overrides: list[str] | None = None) -> DictConfig:
    """Load YAML config and apply CLI overrides.

    Args:
        config_path: Path to default.yaml
        overrides: List of "key=value" strings (from sys.argv[1:] or test fixtures)

    Returns:
        Merged DictConfig with dot-notation access
    """
    base = OmegaConf.load(config_path)
    if overrides:
        cli = OmegaConf.from_dotlist(overrides)
    else:
        # Parse sys.argv[1:] — skip program name
        cli = OmegaConf.from_cli(sys.argv[1:])
    return OmegaConf.merge(base, cli)
```

CLI usage example:
```bash
python scripts/evaluate.py model.use_crf=false training.batch_size=8
```

**Note:** `OmegaConf.from_cli()` and `OmegaConf.from_dotlist()` parse `key.nested=value` syntax. The `merge()` call ensures CLI values override YAML defaults without modifying the file.

### Pattern 2: Device Detection (Three-Way)

**What:** Check CUDA → MPS → CPU in order. Return a `torch.device` and a string label for logging.
**When to use:** Called once at startup in any script that uses PyTorch (Phase 1: smoke test only; Phase 3: training).

```python
# Source: https://docs.pytorch.org/docs/stable/notes/mps.html

import torch

def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU.

    Returns:
        torch.device ready for .to(device) calls
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device
```

**Important:** `torch.backends.mps.is_available()` requires macOS 12.3+ and a PyTorch build with MPS enabled. It returns `False` silently on unsupported platforms — the fallback to CPU is automatic and correct.

### Pattern 3: Fixed Seed Setup

**What:** Set seeds for PyTorch, NumPy. LLM seed is stored in config and passed to API calls in Phase 2.
**When to use:** Called once at startup before any stochastic operation.

```python
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Set seeds for full reproducibility across PyTorch, NumPy, and Python random.

    Args:
        seed: Integer seed from config (e.g., cfg.training.seed)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS does not have a separate seed API — torch.manual_seed() covers it
```

### Pattern 4: Regex Baseline — German Legal References

**What:** A class that takes raw text and returns a list of `(start, end)` character-offset spans for detected legal references.
**When to use:** EVAL-02/EVAL-03 — compared against ML model output using same seqeval metrics.

Core pattern sourced from jura_regex (kiersch/jura_regex) and german-legal-reference-parser (lavis-nlp):

```python
# Source: https://github.com/kiersch/jura_regex
# Source: https://github.com/lavis-nlp/german-legal-reference-parser

import regex  # Use 'regex' PyPI package, not stdlib 're', for overlapped=True

# Pattern covers: §, §§ (multi-section), Art., Artikel, Abs., Nr., lit., Satz, Anhang, Verordnung, EU references
GERMAN_LEGAL_REF_PATTERN = regex.compile(
    r"""
    (?:
        # § and §§ patterns (single and multi-section)
        §§?\s*\d+(?:\w\b)?
        (?:
            (?:\s*(?:Abs\.|Absatz)\s*\d+(?:\w\b)?)?
            (?:\s*(?:S\.|Satz)\s*\d+)?
            (?:\s*(?:Nr\.|Nummer)\s*\d+(?:\w\b)?)?
            (?:\s*lit\.\s*[a-z])?
            (?:\s*(?:Tz\.|Teilziffer)\s*\d+)?
        )*
        (?:\s*[A-Z][A-Za-z]*[A-Z]\w*)?    # law abbreviation (KWG, DSGVO, BGB, etc.)
        |
        # Art. / Artikel patterns
        (?:Art\.|Artikel)\s*\d+(?:\w\b)?
        (?:
            (?:\s*(?:Abs\.|Absatz)\s*\d+(?:\w\b)?)?
            (?:\s*(?:S\.|Satz)\s*\d+)?
            (?:\s*(?:Nr\.|Nummer)\s*\d+(?:\w\b)?)?
            (?:\s*lit\.\s*[a-z])?
            (?:\s*(?:Tz\.|Teilziffer)\s*\d+)?
        )*
        (?:\s*[A-Z][A-Za-z]*[A-Z]\w*)?    # law abbreviation
        |
        # Anhang (Appendix references)
        Anhang\s*(?:[IVX]+|\d+)(?:\s*[A-Z][A-Za-z]*[A-Z]\w*)?
        |
        # Verordnung (Regulation references without § / Art. prefix)
        (?:EU-)?Verordnung\s*(?:Nr\.\s*)?\d+/\d+(?:/[A-Z]+)?
    )
    """,
    regex.VERBOSE | regex.UNICODE,
)
```

**Reference types covered:**
| Type | Example | Pattern element |
|------|---------|----------------|
| § paragraph | § 25a KWG | `§§?\s*\d+` + law abbrev |
| §§ multi-section | §§ 3, 4 UWG | `§§` with number |
| Abs. (Absatz) | Abs. 1 | `(?:Abs\.\|Absatz)` |
| Nr. (Nummer) | Nr. 3a | `Nr\.\s*\d+(?:\w\b)?` |
| lit. (litera) | lit. b | `lit\.\s*[a-z]` |
| Satz | S. 2 | `S\.\|Satz` |
| Tz. (Teilziffer) | Tz. 4 | `Tz\.\|Teilziffer` |
| Art. / Artikel | Art. 6 DSGVO | `Art\.\|Artikel` |
| Anhang | Anhang II | `Anhang\s*(?:[IVX]+\|\d+)` |
| Verordnung | Verordnung 2016/679 | `Verordnung\s*\d+/\d+` |
| EU references | EU-Verordnung 648/2012 | `(?:EU-)?Verordnung` |

### Pattern 5: seqeval Integration — Span-to-BIO Conversion

**What:** Convert character-offset span predictions to BIO label sequences for seqeval.
**Why needed:** seqeval operates on token-level BIO label lists, not character spans. The metrics.py module must bridge between span output (regex and later ML model) and seqeval input.

```python
# Source: https://github.com/chakki-works/seqeval

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

def spans_to_bio(text_tokens: list[str], span_offsets: list[tuple[int, int]]) -> list[str]:
    """Convert character-level spans to BIO token labels.

    Args:
        text_tokens: List of whitespace-split tokens (for baseline evaluation only)
        span_offsets: List of (start, end) character-offset spans

    Returns:
        List of BIO labels, one per token
    """
    # Implementation: map each token's character range to B-REF / I-REF / O
    ...

def compute_entity_metrics(
    y_true: list[list[str]],
    y_pred: list[list[str]],
) -> dict:
    """Compute entity-level P/R/F1 using seqeval.

    Args:
        y_true: List of BIO label sequences (ground truth), one per sample
        y_pred: List of BIO label sequences (predicted), one per sample

    Returns:
        Dict with precision, recall, f1, report string
    """
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "report": classification_report(y_true, y_pred),
    }
```

**Critical:** seqeval expects labels in IOB2 format: `B-REF` for the first token of a span, `I-REF` for continuation tokens, `O` for non-reference tokens. For a single entity type (REF), the type prefix is required — `B-REF` not just `B`.

### Anti-Patterns to Avoid
- **Using token-level accuracy as a metric:** O tokens dominate; all-O classifier gets >90% token accuracy on regulatory text. Always use seqeval entity-level metrics.
- **Using stdlib `re` for §§ patterns:** `re` does not support `overlapped=True`. Multi-section patterns like `§§ 3, 4 UWG` require the `regex` PyPI package.
- **Hardcoding any value in code:** Every threshold, seed value, path, or model name belongs in `config/default.yaml`. No exceptions.
- **Running `load_dotenv()` globally on import:** Call it once at entry point (`scripts/evaluate.py`, `scripts/predict.py`), not in library modules.
- **Using `OmegaConf.update()` to mutate config after load:** Treat config as immutable after `load_config()` returns. Pass the DictConfig down as a function argument.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Entity-level NER metrics | Custom span-matching loop | seqeval `classification_report` | Handles edge cases: partial matches, adjacent spans, I-without-B sequences |
| YAML merge with CLI override | Custom argparse + yaml.load | OmegaConf `load()` + `from_cli()` + `merge()` | Handles nested dot-notation, type coercion, MISSING sentinel |
| Device detection boilerplate | if/elif chains per file | Single `get_device()` utility in utils/ | Prevents divergence between scripts; MPS/CUDA checks have subtleties |
| .env loading | Manual `open(".env")` parsing | python-dotenv `load_dotenv()` | Handles quotes, comments, export prefix, and doesn't override real env vars |

**Key insight:** seqeval is specifically built for sequence label evaluation and handles the BIO bookkeeping that custom implementations routinely get wrong (e.g., I-without-B treated as new span vs error).

---

## Common Pitfalls

### Pitfall 1: Token-Level vs Entity-Level Evaluation
**What goes wrong:** Computing accuracy or F1 over individual tokens instead of complete entity spans. Token-level F1 is misleading — the O class dominates and inflates scores.
**Why it happens:** It's simpler to compare label arrays element-by-element. Token-level is the default for many generic classification metrics.
**How to avoid:** Use seqeval `classification_report()` exclusively. Never use `sklearn.metrics.classification_report` for NER evaluation.
**Warning signs:** F1 above 0.95 on first run with no training; "accuracy" reported instead of "entity F1".

### Pitfall 2: seqeval Expects Nested Lists
**What goes wrong:** Passing a flat list of labels instead of a list-of-lists. seqeval works on batches of sequences.
**Why it happens:** Single-sample evaluation paths flatten the structure by accident.
**How to avoid:** Always wrap single samples: `y_true = [label_list]` not `y_true = label_list`.
**Warning signs:** `TypeError: string indices must be integers` from seqeval internals.

### Pitfall 3: OmegaConf CLI Override Syntax
**What goes wrong:** Using `--model.use_crf=false` (with double-dash) in `OmegaConf.from_cli()`. OmegaConf's CLI parser expects `model.use_crf=false` without leading dashes.
**Why it happens:** Developers familiar with argparse expect GNU-style `--flag` syntax.
**How to avoid:** Document in README that overrides use `key.path=value` syntax, not `--key.path=value`. Strip leading `--` in `load_config()` if needed for user convenience.
**Warning signs:** `OmegaConf.from_cli()` raises `omegaconf.errors.OmegaConfBaseException` or silently ignores the override.

### Pitfall 4: MPS Available But torch.backends.mps.is_built() Not Checked
**What goes wrong:** `torch.backends.mps.is_available()` can return False even on Apple Silicon if the PyTorch build doesn't have MPS compiled in.
**Why it happens:** Some conda/pip installs of PyTorch don't include MPS support.
**How to avoid:** The three-way check (CUDA → MPS → CPU) handles this correctly because `is_available()` already incorporates the build check. Just log the detected device so the user can verify.
**Warning signs:** Running on CPU unexpectedly on an M1 Mac.

### Pitfall 5: §§ Multi-Section Reference Not Matched
**What goes wrong:** `§§ 3, 4 UWG` (double §, comma-separated sections) treated as two separate spans or one broken span.
**Why it happens:** Most simple regex patterns only handle single `§` followed by one number.
**How to avoid:** Use the `regex` PyPI package (not `re`) for the baseline. Pattern must handle `§§` as a prefix for multi-norm references. Consider a two-pass approach: first capture the full `§§ N, M law` span, then optionally decompose internally.
**Warning signs:** Recall drops significantly on text with multi-norm citations.

### Pitfall 6: .env Not Loaded Before Config Access
**What goes wrong:** `os.getenv("OPENROUTER_API_KEY")` returns `None` because `load_dotenv()` was never called. Happens if config loading runs before dotenv initialization.
**Why it happens:** Import order — if config module accesses env vars at module level, dotenv hasn't run yet.
**How to avoid:** Call `load_dotenv()` as the very first statement in each entry point script, before any imports that access env vars.
**Warning signs:** `OPENROUTER_API_KEY` is `None` in config even when `.env` file exists.

---

## Code Examples

### default.yaml — Complete Structure for Phase 1+

```yaml
# config/default.yaml
# All hyperparameters. No hardcoded values in code.

project:
  name: "regulatory-ref-extraction"
  seed: 42

device:
  auto_detect: true  # If false, force 'cpu'

model:
  name: "deepset/gbert-large"
  use_crf: false
  freeze_backbone: false
  use_lora: false
  lora_rank: 16

training:
  batch_size: 4
  learning_rate_backbone: 2.0e-5
  learning_rate_head: 1.0e-4
  warmup_steps: 100
  max_grad_norm: 1.0
  num_epochs: 3
  mixed_precision: "bf16"  # "fp16", "bf16", or "no"

data:
  max_seq_length: 512
  samples_per_batch: 8
  negative_sample_ratio: 0.4
  cache_dir: "data/cache"
  gold_test_dir: "data/gold_test"
  llm_seed: 1337
  llm_model: "google/gemini-flash-1.5"

ensemble:
  enabled: false
  n_estimators: 3

evaluation:
  output_dir: "evaluation_output"
```

### Entry Point: scripts/evaluate.py (skeleton)

```python
# scripts/evaluate.py
import sys
from dotenv import load_dotenv

load_dotenv()  # Must be first — before any config or model imports

from src.utils.config import load_config
from src.utils.device import get_device, set_seed

def main():
    cfg = load_config()
    device = get_device()
    set_seed(cfg.project.seed)
    print(f"Device: {device}")
    # Phase 1: regex baseline evaluation runs here

if __name__ == "__main__":
    main()
```

### seqeval Usage — Entity-Level Report

```python
# Source: https://github.com/chakki-works/seqeval
from seqeval.metrics import classification_report

y_true = [["O", "B-REF", "I-REF", "O", "B-REF", "O"]]
y_pred = [["O", "B-REF", "I-REF", "O", "O",     "O"]]

print(classification_report(y_true, y_pred))
# Output:
#              precision    recall  f1-score   support
#         REF       1.00      0.50      0.67         2
#   micro avg       1.00      0.50      0.67         2
#   macro avg       1.00      0.50      0.67         2
#weighted avg       1.00      0.50      0.67         2
```

Note: seqeval strips the B-/I- prefix and reports by entity type. Label must be `B-REF`/`I-REF`, not `B`/`I`.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `configparser` / `argparse` for ML configs | OmegaConf / Hydra | ~2019-2021 | Hierarchical YAML + CLI overrides become standard in ML |
| Token-level F1 for NER | Entity-level F1 via seqeval | ~2018 (CoNLL standard) | Token metrics are now considered invalid for NER evaluation |
| `re` stdlib for all regex | `regex` PyPI package for complex patterns | Ongoing | `regex` supports overlapping matches, better Unicode; stdlib `re` insufficient for §§ chains |
| Manual device strings in every function | Accelerate / single `get_device()` utility | PyTorch 2.0 + MPS support | Device-agnostic code is now expected, not optional |

**Deprecated/outdated:**
- `pytorch-crf` / `torchcrf`: Both last released 2019-2020, unmaintained. Do not use in Phase 3.
- `datasets` (HuggingFace) for this project: Map-style, not designed for online IterableDataset streaming. Out of scope.
- Token accuracy as NER metric: Never valid for entity extraction; O-token dominance gives false confidence.

---

## Open Questions

1. **seqeval with single entity type — label naming**
   - What we know: seqeval requires `B-REF`/`I-REF` format, not bare `B`/`I`
   - What's unclear: Whether seqeval `mode="strict"` changes behavior for the simple 2-class (REF, O) case
   - Recommendation: Use default mode (no `scheme` arg) for Phase 1 baseline; add `mode="strict", scheme=IOB2` in Phase 4 when the evaluation harness is more mature

2. **Regex precision vs recall tradeoff for baseline**
   - What we know: The baseline intentionally favors recall (catch all references, even ambiguous ones)
   - What's unclear: Exact false positive rate for law abbreviation matching (e.g., "KWG" appearing as a standalone abbreviation without a § prefix)
   - Recommendation: Start with a permissive pattern, measure FP rate on sample data in Phase 1, tighten in Phase 4's error analysis

3. **OmegaConf `from_cli()` with argparse compatibility**
   - What we know: OmegaConf expects `key=value` not `--key=value`
   - What's unclear: Whether Phase 4's `run.py` should use argparse subcommands alongside OmegaConf overrides
   - Recommendation: In Phase 1, use pure OmegaConf CLI syntax. Phase 4 will address the `run.py train/evaluate/predict` subcommand integration.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (no config file yet — Wave 0 gap) |
| Config file | `pytest.ini` or `pyproject.toml [tool.pytest.ini_options]` — see Wave 0 |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CONF-01 | `load_config()` reads default.yaml with no overrides | unit | `pytest tests/test_config.py::test_load_default_config -x` | Wave 0 |
| CONF-02 | `load_config(overrides=["model.use_crf=false"])` overrides YAML default | unit | `pytest tests/test_config.py::test_cli_override -x` | Wave 0 |
| CONF-03 | `set_seed(42)` produces identical random outputs across two calls | unit | `pytest tests/test_config.py::test_seed_reproducibility -x` | Wave 0 |
| CONF-04 | `pip install -r requirements.txt` resolves without conflict (smoke) | manual | manual — run once per environment | N/A |
| EVAL-02 | RegexBaseline.extract() returns spans for all 10 reference types | unit | `pytest tests/test_regex_baseline.py::test_all_reference_types -x` | Wave 0 |
| EVAL-03 | seqeval metrics wrapper returns correct P/R/F1 for known input | unit | `pytest tests/test_metrics.py::test_entity_metrics -x` | Wave 0 |
| DOCS-01 | README.md exists and contains Mermaid code block | unit | `pytest tests/test_docs.py::test_readme_exists -x` | Wave 0 |
| DOCS-02 | .env.example exists and contains OPENROUTER_API_KEY | unit | `pytest tests/test_docs.py::test_env_example -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/ -x -q`
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/__init__.py` — package init
- [ ] `tests/test_config.py` — covers CONF-01, CONF-02, CONF-03
- [ ] `tests/test_regex_baseline.py` — covers EVAL-02 (all 10 reference types with fixture sentences)
- [ ] `tests/test_metrics.py` — covers EVAL-03 (known BIO input → expected P/R/F1)
- [ ] `tests/test_docs.py` — covers DOCS-01, DOCS-02 (file existence checks)
- [ ] `tests/conftest.py` — shared fixtures (sample sentences with known references)
- [ ] `pytest.ini` or `pyproject.toml` — pytest config with testpaths, markers
- [ ] Framework install: `pip install pytest` — not yet in requirements.txt

---

## Sources

### Primary (HIGH confidence)
- [OmegaConf 2.3 Usage Docs](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html) — load, merge, from_cli patterns
- [OmegaConf 2.3 Structured Configs](https://omegaconf.readthedocs.io/en/2.3_branch/structured_config.html) — @dataclass schema + merge validation
- [PyTorch MPS Backend Docs](https://docs.pytorch.org/docs/stable/notes/mps.html) — is_available(), device detection
- [seqeval GitHub](https://github.com/chakki-works/seqeval) — BIO evaluation API, classification_report usage
- [jura_regex GitHub](https://github.com/kiersch/jura_regex) — German legal reference regex patterns (§, Art., Abs., Nr., lit., Satz)
- [german-legal-reference-parser GitHub](https://github.com/lavis-nlp/german-legal-reference-parser) — reference type taxonomy (SimpleLawRef, MultiLawRef, IVM, FileRef)
- [python-dotenv GitHub](https://github.com/theskumar/python-dotenv) — load_dotenv() usage

### Secondary (MEDIUM confidence)
- STACK.md + SUMMARY.md in `.planning/research/` — stack recommendations verified against current docs
- [HuggingFace seqeval evaluate space](https://huggingface.co/spaces/evaluate-metric/seqeval) — confirms entity-level evaluation is the standard

### Tertiary (LOW confidence)
- Exact regex precision/recall on German regulatory text — no public benchmark; will be established empirically in Phase 1

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — OmegaConf, seqeval, torch device detection are all well-documented with current official sources
- Architecture: HIGH — project structure follows the spec exactly; config/src/scripts layout is conventional for ML PoCs
- Regex patterns: MEDIUM-HIGH — jura_regex and german-legal-reference-parser confirm the reference type taxonomy; exact recall on this domain is empirical
- Pitfalls: HIGH — all identified pitfalls are confirmed by official docs (seqeval nested list requirement, OmegaConf CLI syntax, MPS build check)

**Research date:** 2026-03-13
**Valid until:** 2026-06-13 (stable ecosystem; OmegaConf and seqeval are mature)
