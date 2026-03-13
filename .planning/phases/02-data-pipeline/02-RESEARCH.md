# Phase 2: Data Pipeline - Research

**Researched:** 2026-03-13
**Domain:** LLM data generation (OpenRouter/Gemini Flash), BIO token labeling via offset_mapping, PyTorch IterableDataset, disk caching, gold test set construction
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | Training data generated on-the-fly by LLM via OpenRouter API | OpenRouter chat completions API (OpenAI-compatible), httpx async client, tenacity retry pattern documented |
| DATA-02 | LLM generates German regulatory text with `<ref>...</ref>` tagged spans | Prompt engineering pattern with XML-tagged output; Gemini Flash supports structured text generation via OpenRouter |
| DATA-03 | Character-level spans converted to token-level BIO via offset_mapping | BertTokenizerFast offset_mapping pattern documented with code examples; gbert-large supports fast tokenizer |
| DATA-04 | BIO handles BERT subword tokenization (first subtoken gets label, rest configurable) | HuggingFace NER course pattern: first subword gets B-/I- label, continuation subtokens get -100 or I- label |
| DATA-05 | Special tokens ([CLS], [SEP], [PAD]) receive label -100 | offset_mapping returns (0,0) for special tokens; these are masked with -100 for CrossEntropyLoss ignore_index |
| DATA-06 | PyTorch IterableDataset generates batches on-the-fly | IterableDataset pattern with worker_init_fn sharding documented; critical pitfalls identified |
| DATA-07 | Fixed seed per batch for reproducibility | Deterministic seed formula (epoch * 10000 + batch_idx * 100 + offset) already in config; implementation pattern documented |
| DATA-08 | Retry logic with exponential backoff and rate limiting | tenacity + httpx async pattern: retry on 429/408/502, wait_exponential with jitter |
| DATA-09 | Generated data validated -- char offsets verified against text | Round-trip validation: extract text[start:end] for each span, assert match against LLM-tagged content |
| DATA-10 | LLM prompt rotates across regulatory domains | Domain list from config, rotation via seed-based selection per batch |
| DATA-11 | Generated data cached to disk for ensemble resampling | JSONL append-mode caching with one sample per line; reload as standard Dataset |
| GOLD-01 | Gold test set via CLI script (LLM-generated, JSON) | Separate script using same LLM client, higher-quality prompt, fixed seed, JSON output |
| GOLD-02 | Gold samples marked needs_review | Each sample dict gets `"needs_review": true` field |
| GOLD-03 | Gold set contains positive and negative examples | negative_sample_ratio config key (0.4 default); prompt explicitly requests no-reference examples |
</phase_requirements>

---

## Summary

Phase 2 implements the riskiest component of the pipeline: LLM-generated training data that flows through BIO label conversion into a PyTorch-consumable format. The phase has three distinct modules: (1) an async LLM data generator that calls Gemini Flash via OpenRouter with retry logic, (2) a BIO converter that aligns character-level `<ref>` spans to token-level B-REF/I-REF/O labels using the tokenizer's offset_mapping, and (3) a gold test set builder that produces a frozen, reviewable evaluation dataset.

The critical technical risk is the character-span-to-BIO conversion. BERT's WordPiece tokenizer splits German compound words and legal terms (e.g., "Kreditwesengesetz" becomes multiple subtokens), and the offset_mapping must correctly assign B-REF to the first subtoken of each reference span and I-REF to continuations. Special tokens ([CLS], [SEP], [PAD]) must receive label -100 so they are ignored by the loss function. The gbert-large model from deepset uses a WordPiece tokenizer; while it does not ship a `tokenizer.json` file, `AutoTokenizer.from_pretrained("deepset/gbert-large")` will construct a `BertTokenizerFast` from the `vocab.txt` because HuggingFace knows the BERT model type and can build the fast tokenizer automatically. This must be verified at implementation time with an `assert tokenizer.is_fast` guard.

The secondary risk is IterableDataset worker duplication: when `num_workers > 0`, each DataLoader worker gets an independent copy of the dataset, and without proper sharding via `get_worker_info()`, every worker generates identical data. The third risk is LLM API reliability: OpenRouter returns 429 (rate limit), 408 (timeout), and 502 (provider down), all of which need exponential backoff with jitter via tenacity.

**Primary recommendation:** Build the LLM client and BIO converter as independent, testable modules. The IterableDataset is a thin wrapper that composes them. Cache to JSONL (one sample per line) for ensemble resampling. The gold test set builder reuses the LLM client but with a separate high-quality prompt and fixed seed.

---

## Standard Stack

### Core (Phase 2 additions)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| httpx | >= 0.27 | Async HTTP client for OpenRouter API calls | Modern async Python HTTP; superior to aiohttp for typed responses; supports timeouts natively |
| tenacity | >= 8.2 | Retry with exponential backoff + jitter | Standard Python retry library; works with async; used by instructor, langchain, etc. |
| transformers | >= 4.40 | BertTokenizerFast for offset_mapping, tokenization | Required for gbert-large tokenizer; offset_mapping only available on fast tokenizers |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json (stdlib) | - | Parse LLM responses, write gold test set JSON | Always available; no extra dependency |
| asyncio (stdlib) | - | Event loop for async LLM calls | Used by httpx async client |
| pathlib (stdlib) | - | File path handling for cache and gold dirs | Standard over os.path |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| httpx | aiohttp | aiohttp is older, more verbose API; httpx has sync+async in one package, better typing |
| httpx | openai SDK | openai SDK adds heavy dependency; httpx direct calls to OpenRouter's OpenAI-compatible endpoint are simpler |
| tenacity | backoff | backoff is simpler but tenacity has better async support and more retry strategies |
| JSONL cache | SQLite | SQLite adds complexity; JSONL is append-only, human-readable, trivially parseable |
| JSONL cache | pickle/torch.save | Binary formats are fragile across versions; JSONL is portable and inspectable |

**Installation (additions to requirements.txt):**
```bash
pip install httpx tenacity transformers
```

Updated requirements.txt for Phase 2:
```
omegaconf>=2.3.0
torch>=2.0.0
seqeval>=1.2.2
python-dotenv
numpy>=1.24.0
pyyaml>=6.0
regex
pytest
httpx>=0.27.0
tenacity>=8.2.0
transformers>=4.40.0
```

---

## Architecture Patterns

### Recommended Project Structure (Phase 2 additions)
```
src/
├── data/
│   ├── __init__.py
│   ├── llm_client.py         # Async OpenRouter client with retry logic
│   ├── bio_converter.py      # char-span → token-level BIO alignment
│   ├── dataset.py            # IterableDataset wrapping generator + converter
│   └── cache.py              # JSONL disk cache read/write
├── utils/
│   ├── config.py             # (existing)
│   └── device.py             # (existing)
scripts/
├── generate_gold_test.py     # CLI entry point for gold test set creation
data/
├── cache/                    # JSONL training data cache (gitignored)
└── gold_test/                # gold_test_set.json (committed after review)
```

### Pattern 1: Async LLM Client with Tenacity Retry
**What:** Wrap OpenRouter API calls in an async function with tenacity retry decorator
**When to use:** Every LLM call (training data generation and gold test set)
**Example:**
```python
# Source: tenacity docs + OpenRouter API docs
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)

RETRYABLE_STATUS = {429, 408, 502, 503}

class RetryableAPIError(Exception):
    """Raised for HTTP status codes that warrant a retry."""
    pass

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
    retry=retry_if_exception_type(RetryableAPIError),
)
async def call_openrouter(
    client: httpx.AsyncClient,
    model: str,
    messages: list[dict],
    seed: int,
) -> str:
    response = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json={"model": model, "messages": messages, "seed": seed},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    )
    if response.status_code in RETRYABLE_STATUS:
        raise RetryableAPIError(f"Status {response.status_code}")
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
```

### Pattern 2: Character Span to BIO via offset_mapping
**What:** Convert `<ref>...</ref>` character spans to token-level BIO labels
**When to use:** After LLM generates text with tagged spans, before feeding to model
**Example:**
```python
# Source: HuggingFace Course Ch6.3 + NER token classification docs
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-large")
assert tokenizer.is_fast, "offset_mapping requires fast tokenizer"

def char_spans_to_bio(text: str, spans: list[tuple[int, int]], max_length: int = 512):
    """Convert character-level spans to token-level BIO labels.

    Args:
        text: Raw text (without <ref> tags)
        spans: List of (char_start, char_end) tuples for reference spans
        max_length: Max sequence length for tokenizer

    Returns:
        dict with input_ids, attention_mask, labels (BIO as ints)
    """
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors=None,  # plain lists for flexibility
    )
    offsets = encoding["offset_mapping"]
    labels = []

    for idx, (start, end) in enumerate(offsets):
        # Special tokens have offset (0, 0)
        if start == 0 and end == 0:
            labels.append(-100)
            continue

        label = 0  # O by default
        for span_start, span_end in spans:
            if start >= span_start and end <= span_end:
                # Token is inside a reference span
                if start == span_start:
                    label = 1  # B-REF
                else:
                    label = 2  # I-REF
                break
        labels.append(label)

    encoding["labels"] = labels
    del encoding["offset_mapping"]  # not needed for model input
    return encoding
```

### Pattern 3: IterableDataset with Worker Sharding
**What:** PyTorch IterableDataset that generates data on-the-fly, sharded across DataLoader workers
**When to use:** Training loop DataLoader
**Example:**
```python
# Source: PyTorch docs on IterableDataset + get_worker_info
import torch
from torch.utils.data import IterableDataset, get_worker_info

class LLMGeneratedDataset(IterableDataset):
    def __init__(self, config, tokenizer, epoch: int = 0):
        self.config = config
        self.tokenizer = tokenizer
        self.epoch = epoch

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        batch_idx = worker_id  # each worker starts at its own offset
        while True:
            seed = self.epoch * 10000 + batch_idx * 100
            # Generate sample using seed
            sample = self._generate_sample(seed)
            if sample is not None:
                yield sample
            batch_idx += num_workers  # stride by num_workers to avoid overlap
```

### Pattern 4: JSONL Disk Cache
**What:** Append-only JSONL file for caching generated training data
**When to use:** After each batch is generated and converted to BIO labels
**Example:**
```python
import json
from pathlib import Path

def append_to_cache(sample: dict, cache_path: Path) -> None:
    """Append a single sample as one JSON line."""
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def load_cache(cache_path: Path) -> list[dict]:
    """Load all cached samples from JSONL file."""
    samples = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples
```

### Anti-Patterns to Avoid
- **Parsing LLM output with regex only:** LLM output can be malformed. Always validate that extracted `<ref>` spans actually exist at the claimed character positions. Discard samples that fail validation rather than silently corrupting labels.
- **Shared mutable state in IterableDataset:** Each DataLoader worker gets a copy. Do not rely on instance variables being synchronized. Use the seed formula for determinism instead.
- **Blocking sync calls inside async LLM client:** Never mix `requests` with `httpx.AsyncClient`. Use `httpx` throughout for async.
- **Loading entire cache into memory for ensemble:** For large caches, use line-by-line JSONL reading or memory-mapped files. For this PoC scale, full load is fine.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HTTP retry with backoff | Custom retry loops with sleep | tenacity `@retry` decorator | Edge cases: jitter, max attempts, exception filtering, async support |
| Tokenizer offset mapping | Manual character-to-token index math | `tokenizer(return_offsets_mapping=True)` | Off-by-one errors with Unicode, subword boundaries, special chars |
| BIO label scheme | Custom label encoding | Integer mapping {O: 0, B-REF: 1, I-REF: 2, IGNORE: -100} | Must match CrossEntropyLoss ignore_index=-100 default |
| Worker sharding | Manual process management | `get_worker_info()` in IterableDataset.__iter__ | PyTorch handles fork semantics, worker lifecycle |
| JSON serialization | Custom file format | stdlib json + JSONL convention | Human-readable, line-oriented, trivially parseable |

**Key insight:** The BIO conversion is where most NER projects introduce subtle bugs. Using `offset_mapping` from the fast tokenizer eliminates an entire class of alignment errors. Do not attempt manual character counting through subword tokens.

---

## Common Pitfalls

### Pitfall 1: offset_mapping (0,0) Does NOT Mean "First Real Token"
**What goes wrong:** Developers check `start == 0 and end == 0` to detect special tokens, but the first real token of a sentence also starts at character 0 (e.g., offset `(0, 5)` for "Gemae"). The check must be `start == 0 AND end == 0` (both zero), not just `start == 0`.
**Why it happens:** Confusion between "offset is zero" and "offset is the zero-length special token marker."
**How to avoid:** The condition `(start == 0 and end == 0)` is correct and safe -- it uniquely identifies [CLS], [SEP], [PAD]. Just make sure both conditions are AND-ed, not OR-ed.
**Warning signs:** Non-special tokens at position 0 getting label -100.

### Pitfall 2: IterableDataset Worker Duplication
**What goes wrong:** With `num_workers > 0`, every worker generates identical samples because they all start from the same seed without sharding.
**Why it happens:** Each worker gets an independent copy of the dataset object. Without `get_worker_info()`, all copies behave identically.
**How to avoid:** Check `get_worker_info()` in `__iter__()`. Each worker uses `worker_id` as offset and strides by `num_workers`. Seed formula: `epoch * 10000 + batch_idx * 100 + worker_id`.
**Warning signs:** Training loss drops suspiciously fast (model sees same data N times per "epoch").

### Pitfall 3: LLM Returns Malformed XML Tags
**What goes wrong:** The LLM sometimes returns unclosed `<ref>` tags, nested tags, or tags that don't match the actual text positions.
**Why it happens:** LLMs are probabilistic; even with structured prompts, output can be inconsistent.
**How to avoid:** Implement strict validation: (1) parse `<ref>` tags with a simple state machine (not regex for nested cases), (2) verify extracted spans match the cleaned text, (3) discard invalid samples and regenerate. Log discard rate for monitoring.
**Warning signs:** Discard rate above 20% suggests prompt needs refinement.

### Pitfall 4: German Compound Words Spanning Reference Boundaries
**What goes wrong:** German compound words like "Kreditwesengesetz" may be tokenized into 3-4 subtokens. If the reference span ends mid-compound-word, the BIO labels become ambiguous.
**Why it happens:** The LLM-generated `<ref>` tags should wrap complete words, but occasionally cut mid-word.
**How to avoid:** Validation step: assert that every `<ref>` span starts and ends at word boundaries (whitespace or punctuation). If not, expand to nearest word boundary or discard.
**Warning signs:** B-REF label assigned to a `##`-prefixed subtoken.

### Pitfall 5: Padding Tokens Getting Non-(-100) Labels
**What goes wrong:** When padding to max_length, pad tokens receive label 0 (O) instead of -100, causing the model to train on meaningless padding positions.
**Why it happens:** Labels array not padded to same length as input_ids, or pad logic assigns 0 instead of -100.
**How to avoid:** Pad labels to max_length with -100. Verify: `assert all(l == -100 for l, a in zip(labels, attention_mask) if a == 0)`.
**Warning signs:** Model accuracy looks great but actual entity extraction is poor (inflated O-label accuracy from padding).

### Pitfall 6: Rate Limit Cascade in Training Loop
**What goes wrong:** A 429 rate limit during training causes the entire training loop to crash.
**Why it happens:** No retry logic, or retry exhausted without graceful degradation.
**How to avoid:** tenacity retries handle transient failures. For persistent failures (e.g., account out of credits), raise a clear error with instructions rather than hanging indefinitely. Set `stop_after_attempt(5)` to bound retry time.
**Warning signs:** Training hangs for minutes, then crashes with HTTP 429.

---

## Code Examples

### LLM Prompt for Training Data Generation
```python
# Verified pattern: XML-tagged reference generation
DOMAIN_LIST = [
    "BGB", "KWG", "MaRisk", "DORA", "DSGVO", "CRR", "HGB",
    "WpHG", "VAG", "ZAG", "GwG", "SAG", "KAGB",
]

def build_generation_prompt(domain: str, include_references: bool = True) -> str:
    """Build prompt for LLM to generate regulatory text with tagged references."""
    if include_references:
        return (
            f"Generiere einen realistischen deutschen Regulierungstext-Absatz "
            f"aus dem Bereich {domain}. Der Text soll 3-6 Saetze lang sein und "
            f"mehrere Rechtsverweise enthalten (z.B. Paragraphen, Artikel, Absaetze, "
            f"Anhaenge, Verordnungen). Markiere JEDEN Rechtsverweis mit <ref>...</ref> Tags. "
            f"Beispiel: Gemaess <ref>SS 25a Abs. 1 KWG</ref> muessen Institute... "
            f"WICHTIG: Die <ref>-Tags muessen den GESAMTEN Verweis umschliessen, "
            f"inklusive Gesetzeskuerzel. Kein Text ausserhalb der Tags aendern."
        )
    else:
        return (
            f"Generiere einen realistischen deutschen Regulierungstext-Absatz "
            f"aus dem Bereich {domain}. Der Text soll 3-6 Saetze lang sein und "
            f"KEINE Rechtsverweise enthalten (keine Paragraphen, Artikel, etc.)."
        )
```

### Parsing <ref> Tags and Extracting Character Spans
```python
import re

REF_PATTERN = re.compile(r"<ref>(.*?)</ref>", re.DOTALL)

def parse_ref_tags(tagged_text: str) -> tuple[str, list[tuple[int, int]]]:
    """Extract reference spans and return clean text with character offsets.

    Args:
        tagged_text: LLM output with <ref>...</ref> tags

    Returns:
        (clean_text, spans) where spans are (start, end) in clean_text coordinates
    """
    spans = []
    clean_parts = []
    last_end = 0
    offset = 0  # tracks how much we've removed in tags

    for match in REF_PATTERN.finditer(tagged_text):
        # Text before this tag
        clean_parts.append(tagged_text[last_end:match.start()])
        # Character position in clean text
        clean_start = match.start() - offset
        ref_text = match.group(1)
        clean_end = clean_start + len(ref_text)
        spans.append((clean_start, clean_end))
        clean_parts.append(ref_text)
        # Update offset: we removed <ref> (5 chars) and </ref> (6 chars) = 11
        offset += len("<ref>") + len("</ref>")
        last_end = match.end()

    clean_parts.append(tagged_text[last_end:])
    clean_text = "".join(clean_parts)

    # Validation: verify spans match
    for start, end in spans:
        extracted = clean_text[start:end]
        assert len(extracted) > 0, f"Empty span at ({start}, {end})"

    return clean_text, spans
```

### Round-Trip BIO Validation
```python
def validate_bio_roundtrip(
    text: str,
    spans: list[tuple[int, int]],
    encoding: dict,
    offsets: list[tuple[int, int]],
) -> bool:
    """Verify BIO labels can reconstruct the original spans.

    Decodes B-REF/I-REF sequences back to character spans and checks
    they match the original input spans.
    """
    labels = encoding["labels"]
    reconstructed = []
    current_start = None
    current_end = None

    for idx, label in enumerate(labels):
        if label == 1:  # B-REF
            if current_start is not None:
                reconstructed.append((current_start, current_end))
            current_start = offsets[idx][0]
            current_end = offsets[idx][1]
        elif label == 2 and current_start is not None:  # I-REF
            current_end = offsets[idx][1]
        else:
            if current_start is not None:
                reconstructed.append((current_start, current_end))
                current_start = None

    if current_start is not None:
        reconstructed.append((current_start, current_end))

    # Compare: reconstructed spans should match original spans
    return reconstructed == sorted(spans)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual NER annotation | LLM-generated synthetic training data | 2023-2024 | Eliminates annotation cost; requires validation layer |
| word_ids() for label alignment | offset_mapping with char spans | Stable since transformers 4.x | More precise; works at character level not word level |
| requests + manual retry | httpx + tenacity async | 2023+ | Cleaner async, built-in timeout, composable retry |
| torch.save for data cache | JSONL for human-readable cache | Convention | Inspectable, portable, append-friendly |

**Deprecated/outdated:**
- `BertTokenizer` (slow, no offset_mapping) -- always use `BertTokenizerFast` via `AutoTokenizer`
- `word_ids()` approach for char-span alignment -- offset_mapping is more direct when you have character-level annotations
- `aiohttp` for new async HTTP -- httpx is the modern standard with better DX

---

## Open Questions

1. **gbert-large Fast Tokenizer Availability**
   - What we know: The model repo lacks `tokenizer.json`, but `AutoTokenizer` should construct `BertTokenizerFast` from `vocab.txt` for BERT-type models
   - What's unclear: Whether this works reliably for deepset/gbert-large specifically
   - Recommendation: Add `assert tokenizer.is_fast` guard in code. If it fails, manually construct: `BertTokenizerFast.from_pretrained("deepset/gbert-large")`

2. **OpenRouter Gemini Flash Rate Limits**
   - What we know: OpenRouter returns 429 for rate limits; config uses `google/gemini-flash-1.5`
   - What's unclear: Exact rate limits per tier, whether seed parameter is respected by Gemini Flash via OpenRouter
   - Recommendation: Implement retry with backoff. Treat seed as best-effort for reproducibility; disk cache is the true reproducibility mechanism.

3. **Optimal Discard Rate Threshold**
   - What we know: Some LLM outputs will have malformed tags and must be discarded
   - What's unclear: What discard rate is acceptable before prompt tuning is needed
   - Recommendation: Log discard rate. Alert if > 20%. Start with strict validation, relax only if needed.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing from Phase 1) |
| Config file | `pytest.ini` (exists, testpaths = tests) |
| Quick run command | `pytest tests/ -x -q --timeout=10` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | LLM client calls OpenRouter and returns text | unit (mocked httpx) | `pytest tests/test_llm_client.py -x` | Wave 0 |
| DATA-02 | LLM output contains `<ref>` tagged text | unit (mocked) | `pytest tests/test_llm_client.py::test_ref_tag_parsing -x` | Wave 0 |
| DATA-03 | char spans -> token BIO via offset_mapping | unit | `pytest tests/test_bio_converter.py::test_char_to_bio -x` | Wave 0 |
| DATA-04 | Subword tokens get correct BIO labels | unit | `pytest tests/test_bio_converter.py::test_subword_labeling -x` | Wave 0 |
| DATA-05 | Special tokens get -100 | unit | `pytest tests/test_bio_converter.py::test_special_token_masking -x` | Wave 0 |
| DATA-06 | IterableDataset yields tokenized batches | unit (mocked LLM) | `pytest tests/test_dataset.py::test_iterable_yields -x` | Wave 0 |
| DATA-07 | Same seed produces same output | unit | `pytest tests/test_llm_client.py::test_seed_determinism -x` | Wave 0 |
| DATA-08 | Retry on 429/408/502 with backoff | unit (mocked httpx) | `pytest tests/test_llm_client.py::test_retry_on_rate_limit -x` | Wave 0 |
| DATA-09 | Char offsets validated against text | unit | `pytest tests/test_bio_converter.py::test_offset_validation -x` | Wave 0 |
| DATA-10 | Domain rotation across batches | unit | `pytest tests/test_llm_client.py::test_domain_rotation -x` | Wave 0 |
| DATA-11 | JSONL cache write + read roundtrip | unit | `pytest tests/test_cache.py -x` | Wave 0 |
| GOLD-01 | Gold set CLI script produces JSON file | integration | `pytest tests/test_gold_builder.py::test_gold_generation -x` | Wave 0 |
| GOLD-02 | All gold samples have needs_review: true | unit | `pytest tests/test_gold_builder.py::test_needs_review_flag -x` | Wave 0 |
| GOLD-03 | Gold set has positive + negative mix | unit | `pytest tests/test_gold_builder.py::test_positive_negative_mix -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/ -x -q --timeout=10`
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_llm_client.py` -- covers DATA-01, DATA-02, DATA-07, DATA-08, DATA-10
- [ ] `tests/test_bio_converter.py` -- covers DATA-03, DATA-04, DATA-05, DATA-09
- [ ] `tests/test_dataset.py` -- covers DATA-06
- [ ] `tests/test_cache.py` -- covers DATA-11
- [ ] `tests/test_gold_builder.py` -- covers GOLD-01, GOLD-02, GOLD-03
- [ ] `pytest-timeout` added to requirements (optional but recommended)
- [ ] `pytest-asyncio` added to requirements for async LLM client tests

---

## Sources

### Primary (HIGH confidence)
- [HuggingFace Course Ch6.3: Fast Tokenizers' Special Powers](https://huggingface.co/course/chapter6/3) -- offset_mapping pattern, word_ids(), NER post-processing code examples
- [HuggingFace Token Classification Task Guide](https://huggingface.co/docs/transformers/tasks/token_classification) -- BIO label alignment, subword handling
- [HuggingFace Tokenizer API Docs](https://huggingface.co/docs/transformers/main_classes/tokenizer) -- return_offsets_mapping, PreTrainedTokenizerFast requirements
- [PyTorch IterableDataset Docs](https://docs.pytorch.org/docs/stable/data.html) -- get_worker_info, worker sharding pattern, duplicate data prevention
- [Tenacity Documentation](https://tenacity.readthedocs.io/en/stable/) -- retry, wait_exponential_jitter, retry_if_exception_type, async support
- [OpenRouter API Error Reference](https://openrouter.ai/docs/api/reference/errors-and-debugging) -- HTTP 429/408/502 status codes

### Secondary (MEDIUM confidence)
- [deepset/gbert-large Model Card](https://huggingface.co/deepset/gbert-large) -- Model info; tokenizer_config.json has minimal fields (no tokenizer_class specified)
- [deepset/gbert-large File Listing](https://huggingface.co/deepset/gbert-large/tree/main) -- Confirmed: vocab.txt present, tokenizer.json absent
- [OpenRouter Structured Outputs Announcement](https://openrouter.ai/announcements/structured-outputs-and-free-gemini-flash-20) -- Gemini Flash supports structured output via OpenRouter
- [Jannis Vamvas: BERT for NER](https://vamvas.ch/bert-for-ner) -- German BERT NER practical considerations

### Tertiary (LOW confidence)
- [Medium: Streaming DataLoader with PyTorch](https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd) -- IterableDataset patterns (not officially verified)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- httpx, tenacity, transformers are well-established; versions verified against PyPI
- Architecture (BIO converter): HIGH -- offset_mapping pattern is authoritative from HuggingFace course and docs
- Architecture (IterableDataset): HIGH -- PyTorch official docs document worker sharding pattern
- Architecture (LLM client): MEDIUM -- OpenRouter API is OpenAI-compatible but seed behavior with Gemini Flash not fully verified
- Pitfalls: HIGH -- identified from official docs, community issues, and known NER failure modes
- gbert-large fast tokenizer: MEDIUM -- should work (BERT type auto-maps to Fast), but no tokenizer.json shipped; needs runtime assert

**Research date:** 2026-03-13
**Valid until:** 2026-04-13 (stable domain; transformers and httpx APIs unlikely to change)
