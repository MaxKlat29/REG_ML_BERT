"""BIO label converter: character-level spans -> token-level BIO labels.

Uses offset_mapping from fast HuggingFace tokenizers to align character-level
reference spans with subword tokens.

Label constants:
    LABEL_O = 0        Outside any reference
    LABEL_B_REF = 1    Beginning of a reference span
    LABEL_I_REF = 2    Continuation of a reference span
    LABEL_IGNORE = -100  Special tokens, padding (CrossEntropyLoss ignore_index)
"""
from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Label constants
LABEL_O = 0
LABEL_B_REF = 1
LABEL_I_REF = 2
LABEL_IGNORE = -100


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str = "deepset/gbert-large"):
    """Load and cache a fast tokenizer.

    Uses BertTokenizerFast for gbert-large (BERT-based models) to ensure
    offset_mapping support. Falls back to AutoTokenizer for other models.

    Args:
        model_name: HuggingFace model name. Must have a fast tokenizer
                    so that offset_mapping is available.

    Returns:
        PreTrainedTokenizerFast instance.
    """
    from transformers import BertTokenizerFast

    tok = BertTokenizerFast.from_pretrained(model_name)
    assert tok.is_fast, (
        f"Tokenizer for '{model_name}' is not a fast tokenizer. "
        "offset_mapping requires a fast tokenizer."
    )
    return tok


def char_spans_to_bio(
    text: str,
    spans: list[tuple[int, int]],
    tokenizer,
    max_length: int = 512,
) -> dict:
    """Convert character-level reference spans to token-level BIO labels.

    Args:
        text: Raw text to tokenize.
        spans: List of (start, end) char offsets for reference spans.
               start is inclusive, end is exclusive.
        tokenizer: Fast HuggingFace tokenizer (must support offset_mapping).
        max_length: Maximum sequence length (padding + truncation target).

    Returns:
        dict with keys:
            input_ids: list[int]
            attention_mask: list[int]
            labels: list[int]  -- BIO labels, -100 for special/padding tokens
    """
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    offset_mapping: list[tuple[int, int]] = encoding["offset_mapping"]
    attention_mask: list[int] = encoding["attention_mask"]
    labels: list[int] = []

    for idx, (tok_start, tok_end) in enumerate(offset_mapping):
        # Special tokens have (0, 0) for BOTH start and end.
        # CRITICAL: first real token can have start=0, end>0 — check both!
        if tok_start == 0 and tok_end == 0:
            labels.append(LABEL_IGNORE)
            continue

        # Padding: attention_mask == 0
        if attention_mask[idx] == 0:
            labels.append(LABEL_IGNORE)
            continue

        # Determine label by checking overlap with each span
        assigned = LABEL_O
        for span_start, span_end in spans:
            # Token is fully inside span
            if tok_start >= span_start and tok_end <= span_end:
                if tok_start == span_start:
                    assigned = LABEL_B_REF
                else:
                    assigned = LABEL_I_REF
                break  # first matching span wins

        labels.append(assigned)

    # Safety pass: ensure every padding position has LABEL_IGNORE
    for idx, mask in enumerate(attention_mask):
        if mask == 0 and labels[idx] != LABEL_IGNORE:
            logger.warning("Padding position %d had non-ignore label; correcting.", idx)
            labels[idx] = LABEL_IGNORE

    return {
        "input_ids": list(encoding["input_ids"]),
        "attention_mask": list(attention_mask),
        "labels": labels,
    }


def validate_bio_roundtrip(
    text: str,
    spans: list[tuple[int, int]],
    encoding: dict,
    offsets: list[tuple[int, int]],
) -> bool:
    """Reconstruct spans from BIO labels + offset mapping and compare to original.

    Args:
        text: Original text (unused for reconstruction but kept for signature).
        spans: Expected (ground-truth) char spans.
        encoding: dict returned by char_spans_to_bio (must contain 'labels').
        offsets: offset_mapping from the tokenizer encoding.

    Returns:
        True if reconstructed spans match original spans exactly.
    """
    labels = encoding["labels"]

    reconstructed: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int | None = None

    for (tok_start, tok_end), label in zip(offsets, labels):
        if label == LABEL_B_REF:
            # Emit previous span if any
            if current_start is not None:
                reconstructed.append((current_start, current_end))
            current_start = tok_start
            current_end = tok_end
        elif label == LABEL_I_REF:
            if current_start is not None:
                current_end = tok_end
            # else: I-REF without preceding B-REF — malformed, ignore
        else:
            # End of span (O, IGNORE, etc.)
            if current_start is not None:
                reconstructed.append((current_start, current_end))
                current_start = None
                current_end = None

    # Don't forget last open span
    if current_start is not None:
        reconstructed.append((current_start, current_end))

    return reconstructed == list(spans)
