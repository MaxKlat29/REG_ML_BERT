"""Evaluation metrics: span-to-BIO conversion and seqeval entity-level metrics."""

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


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
