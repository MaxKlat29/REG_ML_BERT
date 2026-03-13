"""
Tests for RegulatoryNERModel.

Uses a tiny randomly initialized BertConfig (hidden_size=64, 1 layer) to avoid
downloading gbert-large.  BertModel.from_pretrained and
BertForTokenClassification.from_pretrained are patched to return the tiny models.
"""

import types
import pytest
import torch
from types import SimpleNamespace
from unittest.mock import patch

from transformers import BertConfig, BertModel, BertForTokenClassification


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TINY_CFG = BertConfig(
    hidden_size=64,
    num_hidden_layers=1,
    num_attention_heads=1,
    intermediate_size=128,
    max_position_embeddings=64,
    vocab_size=100,
)


def tiny_bert_model(*args, **kwargs):
    """Return a tiny BertModel with random weights (no download)."""
    return BertModel(TINY_CFG)


def tiny_bert_for_tc(*args, **kwargs):
    """Return a tiny BertForTokenClassification with num_labels=3."""
    return BertForTokenClassification(BertConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=128,
        max_position_embeddings=64,
        vocab_size=100,
        num_labels=3,
    ))


def make_config(
    use_crf=False,
    freeze_backbone=False,
    use_lora=False,
    lora_rank=4,
    name="deepset/gbert-large",
):
    """Build a SimpleNamespace that mirrors the real OmegaConf config schema."""
    model_ns = SimpleNamespace(
        name=name,
        use_crf=use_crf,
        freeze_backbone=freeze_backbone,
        use_lora=use_lora,
        lora_rank=lora_rank,
    )
    return SimpleNamespace(model=model_ns)


def make_batch(batch_size=2, seq_len=16, num_labels=3, with_labels=True):
    """Create a small random batch of token IDs + attention mask (+ labels).

    Note on CRF masking: pytorch-crf requires mask[:,0] to be True (batch_first).
    So we use -100 only for the last token ([SEP]), NOT for position 0 ([CLS]).
    In the CRF forward, position 0 will have label 0 (O) which is valid.
    """
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    if not with_labels:
        return input_ids, attention_mask
    # labels: mostly 0 (O), with -100 for last position only (SEP token)
    labels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    labels[:, -1] = -100  # [SEP]
    return input_ids, attention_mask, labels


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNonCRFForward:
    """test_forward_shape: Non-CRF model returns loss + logits (B, S, 3)."""

    def test_forward_shape(self):
        cfg = make_config(use_crf=False)
        with (
            patch("transformers.BertForTokenClassification.from_pretrained", side_effect=tiny_bert_for_tc),
        ):
            from src.model.ner_model import RegulatoryNERModel
            model = RegulatoryNERModel(cfg)

        input_ids, attention_mask, labels = make_batch()
        output = model(input_ids, attention_mask, labels=labels)

        # BertForTokenClassification returns TokenClassifierOutput
        loss = output.loss
        logits = output.logits

        assert loss is not None, "loss should not be None when labels provided"
        assert loss.shape == (), f"loss must be scalar, got shape {loss.shape}"
        assert logits.shape == (2, 16, 3), f"logits shape wrong: {logits.shape}"


class TestCRFToggle:
    """test_crf_toggle: CRF model returns (loss, emissions); decode returns tag lists."""

    def test_crf_toggle_with_labels(self):
        cfg = make_config(use_crf=True)
        with (
            patch("transformers.BertModel.from_pretrained", side_effect=tiny_bert_model),
        ):
            from src.model.ner_model import RegulatoryNERModel
            model = RegulatoryNERModel(cfg)

        input_ids, attention_mask, labels = make_batch()
        loss, emissions = model(input_ids, attention_mask, labels=labels)

        assert loss.shape == (), f"CRF loss must be scalar, got {loss.shape}"
        assert torch.isfinite(loss), "CRF loss must be finite"
        assert emissions.shape == (2, 16, 3), f"emissions shape wrong: {emissions.shape}"

    def test_crf_toggle_decode(self):
        cfg = make_config(use_crf=True)
        with (
            patch("transformers.BertModel.from_pretrained", side_effect=tiny_bert_model),
        ):
            from src.model.ner_model import RegulatoryNERModel
            model = RegulatoryNERModel(cfg)

        input_ids, attention_mask = make_batch(with_labels=False)
        result = model(input_ids, attention_mask, labels=None)

        # decode() returns list of lists of tag indices
        assert isinstance(result, list), "decode must return list"
        assert len(result) == 2, "one tag list per batch item"
        for tag_seq in result:
            assert isinstance(tag_seq, list)
            assert all(isinstance(t, int) for t in tag_seq)


class TestCRFHandlesMinus100:
    """test_crf_handles_minus100: CRF path masks -100 labels; no IndexError, loss finite.

    pytorch-crf requires mask[:,0] (first timestep) to be True.
    So we set position 0 to label 0 (valid), and use -100 only for non-first positions.
    """

    def test_crf_handles_minus100(self):
        cfg = make_config(use_crf=True)
        with (
            patch("transformers.BertModel.from_pretrained", side_effect=tiny_bert_model),
        ):
            from src.model.ner_model import RegulatoryNERModel
            model = RegulatoryNERModel(cfg)

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        # Heavy -100 masking: only positions 0 and 2..13 are real tokens
        # Position 0 must be non-(-100) for pytorch-crf constraint (mask[:,0] must be True)
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        labels[:, 0] = 0    # first position must be valid (CRF constraint)
        labels[:, 2:14] = 0  # label O for positions 2..13

        # Must not raise IndexError / AssertionError / ValueError
        loss, emissions = model(input_ids, attention_mask, labels=labels)

        assert torch.isfinite(loss), f"loss must be finite with -100 labels, got {loss.item()}"


class TestFreezeToggle:
    """test_freeze_toggle: freeze_backbone=True makes all BERT params non-trainable."""

    def test_freeze_toggle(self):
        cfg = make_config(use_crf=False, freeze_backbone=True)
        with (
            patch("transformers.BertForTokenClassification.from_pretrained", side_effect=tiny_bert_for_tc),
        ):
            from src.model.ner_model import RegulatoryNERModel
            model = RegulatoryNERModel(cfg)

        # All BERT encoder params should be frozen
        bert_params = list(model.bert_tc.bert.parameters())
        assert bert_params, "BERT encoder should have parameters"
        for p in bert_params:
            assert not p.requires_grad, "BERT encoder param should be frozen"

        # Classifier head should still be trainable
        head_params = list(model.bert_tc.classifier.parameters())
        assert head_params, "classifier head should have parameters"
        for p in head_params:
            assert p.requires_grad, "classifier head param should be trainable"

    def test_freeze_toggle_crf_path(self):
        cfg = make_config(use_crf=True, freeze_backbone=True)
        with (
            patch("transformers.BertModel.from_pretrained", side_effect=tiny_bert_model),
        ):
            from src.model.ner_model import RegulatoryNERModel
            model = RegulatoryNERModel(cfg)

        bert_params = list(model.bert.parameters())
        for p in bert_params:
            assert not p.requires_grad, "CRF path: BERT encoder param should be frozen"

        # classifier linear layer should still be trainable
        for p in model.classifier.parameters():
            assert p.requires_grad, "CRF path: classifier should still be trainable"


class TestLoRAToggle:
    """test_lora_toggle: use_lora=True results in only LoRA adapter + head params trainable."""

    def test_lora_toggle(self):
        cfg = make_config(use_crf=False, use_lora=True, lora_rank=4)
        with (
            patch("transformers.BertForTokenClassification.from_pretrained", side_effect=tiny_bert_for_tc),
        ):
            from src.model.ner_model import RegulatoryNERModel
            model = RegulatoryNERModel(cfg)

        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        total = list(model.parameters())

        assert len(trainable) > 0, "LoRA model must have some trainable params"
        # trainable count must be strictly less than total param count (most are frozen)
        total_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
        assert trainable_count < total_count, (
            f"LoRA should freeze most params; trainable={trainable_count}, total={total_count}"
        )


class TestCPUForward:
    """test_cpu_forward: Model instantiates and runs forward on CPU for both CRF and non-CRF."""

    def test_cpu_forward_non_crf(self):
        cfg = make_config(use_crf=False)
        with (
            patch("transformers.BertForTokenClassification.from_pretrained", side_effect=tiny_bert_for_tc),
        ):
            from src.model.ner_model import RegulatoryNERModel
            model = RegulatoryNERModel(cfg)

        assert next(model.parameters()).device.type == "cpu"
        input_ids, attention_mask, labels = make_batch()
        output = model(input_ids, attention_mask, labels=labels)
        assert output.loss is not None

    def test_cpu_forward_crf(self):
        cfg = make_config(use_crf=True)
        with (
            patch("transformers.BertModel.from_pretrained", side_effect=tiny_bert_model),
        ):
            from src.model.ner_model import RegulatoryNERModel
            model = RegulatoryNERModel(cfg)

        assert next(model.parameters()).device.type == "cpu"
        input_ids, attention_mask, labels = make_batch()
        loss, emissions = model(input_ids, attention_mask, labels=labels)
        assert torch.isfinite(loss)
