"""Tests for BIO converter: char spans -> token-level BIO labels.

All tests use a module-scoped tokenizer fixture to avoid repeated loads.
"""
import pytest

from src.data.bio_converter import (
    LABEL_B_REF,
    LABEL_I_REF,
    LABEL_IGNORE,
    LABEL_O,
    char_spans_to_bio,
    validate_bio_roundtrip,
)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import BertTokenizerFast
    tok = BertTokenizerFast.from_pretrained("deepset/gbert-large")
    assert tok.is_fast, "offset_mapping requires fast tokenizer"
    return tok


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_real_labels(encoding: dict) -> list[int]:
    """Return labels for positions where attention_mask == 1."""
    return [
        label
        for label, mask in zip(encoding["labels"], encoding["attention_mask"])
        if mask == 1
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_char_to_bio_simple(tokenizer):
    """Span 'SS 25a KWG' in 'Gemaess SS 25a KWG gilt' produces B-REF on first
    subtoken, I-REF on rest of span."""
    text = "Gemaess SS 25a KWG gilt"
    span_text = "SS 25a KWG"
    start = text.index(span_text)
    end = start + len(span_text)
    encoding = char_spans_to_bio(text, [(start, end)], tokenizer)

    labels = encoding["labels"]
    # Exactly one B-REF
    assert labels.count(LABEL_B_REF) == 1
    # At least one I-REF (span has multiple subtokens)
    assert labels.count(LABEL_I_REF) >= 1
    # No O labels inside span region: B-REF followed immediately by I-REF(s)
    b_idx = labels.index(LABEL_B_REF)
    for i in range(b_idx + 1, b_idx + labels.count(LABEL_B_REF) + labels.count(LABEL_I_REF)):
        assert labels[i] in (LABEL_I_REF, LABEL_B_REF, LABEL_IGNORE)


def test_char_to_bio_multiple_spans(tokenizer):
    """Two non-overlapping spans produce two B-REF starts."""
    text = "Gemaess SS 25a KWG und Art. 6 DSGVO gilt."
    span1_text = "SS 25a KWG"
    span2_text = "Art. 6 DSGVO"
    s1 = text.index(span1_text)
    e1 = s1 + len(span1_text)
    s2 = text.index(span2_text)
    e2 = s2 + len(span2_text)
    encoding = char_spans_to_bio(text, [(s1, e1), (s2, e2)], tokenizer)

    labels = encoding["labels"]
    assert labels.count(LABEL_B_REF) == 2


def test_special_token_masking(tokenizer):
    """[CLS] at index 0 and [SEP] at last real-token position get LABEL_IGNORE."""
    text = "Gemaess SS 25a KWG gilt"
    span_text = "SS 25a KWG"
    start = text.index(span_text)
    end = start + len(span_text)
    encoding = char_spans_to_bio(text, [(start, end)], tokenizer)

    labels = encoding["labels"]
    input_ids = encoding["input_ids"]

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    cls_positions = [i for i, t in enumerate(input_ids) if t == cls_id]
    sep_positions = [i for i, t in enumerate(input_ids) if t == sep_id]

    assert all(labels[i] == LABEL_IGNORE for i in cls_positions), "CLS must be -100"
    assert all(labels[i] == LABEL_IGNORE for i in sep_positions), "SEP must be -100"


def test_padding_gets_ignore_label(tokenizer):
    """For short text padded to max_length, all padding positions have LABEL_IGNORE."""
    text = "Kurzer Text."
    encoding = char_spans_to_bio(text, [], tokenizer, max_length=512)

    labels = encoding["labels"]
    attention_mask = encoding["attention_mask"]

    pad_labels = [labels[i] for i, m in enumerate(attention_mask) if m == 0]
    assert all(lbl == LABEL_IGNORE for lbl in pad_labels), (
        "All padded positions must have label -100"
    )
    # Short text must actually have padding
    assert any(m == 0 for m in attention_mask), "Expected padding for short text"


def test_subword_labeling(tokenizer):
    """German compound 'Kreditwesengesetz' inside a span: first subtoken B-REF,
    continuations I-REF."""
    text = "Gemaess Kreditwesengesetz gilt"
    span_text = "Kreditwesengesetz"
    start = text.index(span_text)
    end = start + len(span_text)
    encoding = char_spans_to_bio(text, [(start, end)], tokenizer)

    labels = encoding["labels"]
    # Compound splits into subtokens -> B-REF once, possibly I-REF
    assert labels.count(LABEL_B_REF) == 1
    b_idx = labels.index(LABEL_B_REF)
    # Everything right after B-REF (until next O or special) should be I-REF or special
    after = labels[b_idx + 1 :]
    # First non-ignore after B-REF should be I-REF if there are continuation subtokens
    real_after = [l for l in after if l != LABEL_IGNORE]
    if real_after and real_after[0] != LABEL_O:
        assert real_after[0] == LABEL_I_REF


def test_offset_validation_passes(tokenizer):
    """Valid spans return True from validate_bio_roundtrip."""
    text = "Gemaess SS 25a KWG gilt"
    span_text = "SS 25a KWG"
    start = text.index(span_text)
    end = start + len(span_text)
    encoding = char_spans_to_bio(text, [(start, end)], tokenizer)
    enc_obj = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    result = validate_bio_roundtrip(text, [(start, end)], encoding, enc_obj["offset_mapping"])
    assert result is True


def test_offset_validation_detects_mismatch(tokenizer):
    """Deliberately wrong spans produce False from validate_bio_roundtrip."""
    text = "Gemaess SS 25a KWG gilt"
    span_text = "SS 25a KWG"
    start = text.index(span_text)
    end = start + len(span_text)
    encoding = char_spans_to_bio(text, [(start, end)], tokenizer)
    enc_obj = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    # Provide wrong spans -> mismatch
    wrong_spans = [(0, 3)]  # "Gem" — not the labelled span
    result = validate_bio_roundtrip(text, wrong_spans, encoding, enc_obj["offset_mapping"])
    assert result is False


def test_no_refs_all_O(tokenizer):
    """Text with no reference spans produces only O and -100 labels (no B-REF/I-REF)."""
    text = "Dies ist ein normaler Satz ohne Referenz."
    encoding = char_spans_to_bio(text, [], tokenizer)

    labels = encoding["labels"]
    for lbl in labels:
        assert lbl in (LABEL_O, LABEL_IGNORE), f"Unexpected label {lbl} in no-ref text"
