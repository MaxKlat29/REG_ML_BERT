"""Evaluation metrics: span-to-BIO conversion and seqeval entity-level metrics."""

import regex
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# Label integer constants (mirror bio_converter.py)
_LABEL_O = 0
_LABEL_B_REF = 1
_LABEL_I_REF = 2
_LABEL_IGNORE = -100

# Per-type classification patterns — checked in priority order.
# Each maps a reference type name to a compiled regex that matches
# the *start* or body of a span text.
_TYPE_PATTERNS: list[tuple[str, object]] = [
    ("PARAGRAPH",   regex.compile(r"§")),
    ("ARTIKEL",     regex.compile(r"(?:Art\.|Artikel)")),
    ("ABSATZ",      regex.compile(r"(?:Abs\.|Absatz)")),
    ("NUMMER",      regex.compile(r"(?:Nr\.|Nummer)")),
    ("LITERAL",     regex.compile(r"lit\.")),
    ("SATZ",        regex.compile(r"(?:Satz|S\.)\s*\d")),
    ("TEILZIFFER",  regex.compile(r"(?:Tz\.|Teilziffer)")),
    ("ANHANG",      regex.compile(r"Anhang")),
    ("VERORDNUNG",  regex.compile(r"(?:EU-)?Verordnung")),
]


def spans_to_bio(
    text: str,
    char_spans: list[tuple[int, int]],
) -> tuple[list[str], list[str]]:
    """Convert character-level spans to BIO token labels.

    Whitespace-tokenizes text, computes each token's character range,
    and maps to B-REF/I-REF/O based on overlap with spans.

    Args:
        text: Raw input text.
        char_spans: List of (start, end) character offset spans.

    Returns:
        Tuple of (tokens, bio_labels).
    """
    tokens: list[str] = []
    token_ranges: list[tuple[int, int]] = []

    # Whitespace tokenization with character offset tracking
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        start = i
        while i < len(text) and not text[i].isspace():
            i += 1
        tokens.append(text[start:i])
        token_ranges.append((start, i))

    labels: list[str] = []
    for tok_start, tok_end in token_ranges:
        label = "O"
        for span_start, span_end in char_spans:
            # Token overlaps with span if they intersect
            if tok_start < span_end and tok_end > span_start:
                # B-REF if this is the first token in the span
                if tok_start <= span_start:
                    label = "B-REF"
                else:
                    label = "I-REF"
                break
        labels.append(label)

    return tokens, labels


def compute_entity_metrics(
    y_true: list[list[str]],
    y_pred: list[list[str]],
) -> dict:
    """Compute entity-level Precision/Recall/F1 using seqeval.

    Args:
        y_true: List of BIO label sequences (ground truth), one per sample.
        y_pred: List of BIO label sequences (predicted), one per sample.

    Returns:
        Dict with keys: precision, recall, f1, report.
    """
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "report": classification_report(y_true, y_pred),
    }


def span_iou(pred: tuple[int, int], gold: tuple[int, int]) -> float:
    """Compute intersection over union for two character-offset spans.

    Args:
        pred: Predicted (start, end) span (end is exclusive).
        gold: Gold (start, end) span (end is exclusive).

    Returns:
        IoU as float in [0.0, 1.0]. Returns 1.0 for two zero-length spans
        at the same position (degenerate case where union is 0).
    """
    p_start, p_end = pred
    g_start, g_end = gold

    intersection = max(0, min(p_end, g_end) - max(p_start, g_start))
    union = max(p_end, g_end) - min(p_start, g_start)

    if union <= 0:
        # Both spans are zero-length at the same position — treat as identical
        return 1.0 if intersection == 0 and p_start == g_start else 0.0

    return intersection / union


def classify_span_type(span_text: str, full_text: str = "") -> str:
    """Classify a reference span into a German legal reference type.

    Checks span_text against type-specific patterns in priority order.
    Returns the first match or "REF" as fallback.

    Args:
        span_text: The text content of the span.
        full_text: Unused — kept for API compatibility.

    Returns:
        One of: PARAGRAPH, ARTIKEL, ABSATZ, NUMMER, LITERAL, SATZ,
        TEILZIFFER, ANHANG, VERORDNUNG, or REF.
    """
    for type_name, pattern in _TYPE_PATTERNS:
        if pattern.search(span_text):
            return type_name
    return "REF"


def decode_bio_to_char_spans(
    token_labels: list[int],
    offset_mapping: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Reconstruct character-level spans from integer BIO labels + offset mapping.

    Skips special tokens identified by (start==0, end==0) offsets.
    Handles B-REF starting a new span, I-REF continuing, and O/other closing.

    Args:
        token_labels: Integer BIO labels per token (0=O, 1=B-REF, 2=I-REF, -100=IGNORE).
        offset_mapping: List of (char_start, char_end) per token from tokenizer.

    Returns:
        List of (start, end) character offset spans, in order.
    """
    spans: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int | None = None

    for (tok_start, tok_end), label in zip(offset_mapping, token_labels):
        # Skip special tokens (CLS, SEP, PAD) identified by (0, 0)
        if tok_start == 0 and tok_end == 0:
            # Close any open span
            if current_start is not None:
                spans.append((current_start, current_end))
                current_start = None
                current_end = None
            continue

        if label == _LABEL_B_REF:
            # Close any previously open span
            if current_start is not None:
                spans.append((current_start, current_end))
            current_start = tok_start
            current_end = tok_end

        elif label == _LABEL_I_REF:
            if current_start is not None:
                current_end = tok_end
            # I-REF without B-REF (malformed) — ignore

        else:
            # O, IGNORE, or anything else — close current span if any
            if current_start is not None:
                spans.append((current_start, current_end))
                current_start = None
                current_end = None

    # Don't forget last open span
    if current_start is not None:
        spans.append((current_start, current_end))

    return spans


def compute_partial_match_metrics(
    gold_spans: list[tuple[int, int]],
    pred_spans: list[tuple[int, int]],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute precision/recall/F1 at span level using IoU matching.

    A predicted span is a true positive if it matches any unmatched gold span
    with IoU > iou_threshold (greedy 1:1 matching, first come first matched).

    Edge cases:
        - Empty pred_spans and non-empty gold: precision=1.0, recall=0.0, f1=0.0
        - Non-empty pred and empty gold: precision=0.0, recall=1.0, f1=0.0
        - Both empty: precision=1.0, recall=1.0, f1=1.0

    Args:
        gold_spans: Gold (start, end) spans.
        pred_spans: Predicted (start, end) spans.
        iou_threshold: IoU threshold above which a match is a TP.

    Returns:
        Dict with keys: precision, recall, f1.
    """
    if not gold_spans and not pred_spans:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not pred_spans:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

    if not gold_spans:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    matched_gold: set[int] = set()
    tp = 0

    for pred in pred_spans:
        best_iou = 0.0
        best_idx = -1
        for g_idx, gold in enumerate(gold_spans):
            if g_idx in matched_gold:
                continue
            iou = span_iou(pred, gold)
            if iou > best_iou:
                best_iou = iou
                best_idx = g_idx
        if best_iou > iou_threshold and best_idx >= 0:
            tp += 1
            matched_gold.add(best_idx)

    precision = tp / len(pred_spans) if pred_spans else 1.0
    recall = tp / len(gold_spans) if gold_spans else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}
