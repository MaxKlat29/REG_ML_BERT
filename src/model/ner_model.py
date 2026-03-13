"""
RegulatoryNERModel — BERT-based token classifier for German legal references.

Labels: O=0, B-REF=1, I-REF=2

Config toggles (via config.model.*):
  use_crf        — CRF layer for valid BIO transition enforcement
  freeze_backbone — freeze all BERT encoder parameters
  use_lora       — LoRA adapters on attention query/value matrices (PEFT)
  lora_rank      — LoRA rank r (default 16)
"""

from __future__ import annotations

import logging
import warnings

import torch
import torch.nn as nn
from transformers import BertModel, BertForTokenClassification

logger = logging.getLogger(__name__)


class RegulatoryNERModel(nn.Module):
    """Token classifier built on BERT with optional CRF, LoRA, and backbone freeze."""

    NUM_LABELS = 3
    O = 0
    B_REF = 1
    I_REF = 2

    def __init__(self, config) -> None:
        super().__init__()
        model_name: str = config.model.name

        if config.model.use_crf:
            self._build_crf_path(model_name)
        else:
            self._build_non_crf_path(model_name)

        # Freeze backbone BEFORE applying LoRA (LoRA overrides adapter params)
        if config.model.freeze_backbone:
            self._freeze_backbone()

        if config.model.use_lora:
            self._apply_lora(config.model.lora_rank)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_crf_path(self, model_name: str) -> None:
        from torchcrf import CRF

        self.bert = BertModel.from_pretrained(model_name)
        hidden_size: int = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, self.NUM_LABELS)
        self.crf = CRF(self.NUM_LABELS, batch_first=True)
        self._use_crf = True

    def _build_non_crf_path(self, model_name: str) -> None:
        self.bert_tc = BertForTokenClassification.from_pretrained(
            model_name, num_labels=self.NUM_LABELS
        )
        self._use_crf = False

    def _freeze_backbone(self) -> None:
        """Set requires_grad=False on all BERT encoder parameters."""
        if self._use_crf:
            for p in self.bert.parameters():
                p.requires_grad = False
        else:
            for p in self.bert_tc.bert.parameters():
                p.requires_grad = False

    def _apply_lora(self, lora_rank: int) -> None:
        """Apply PEFT LoRA adapters to the BERT encoder."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType
        except ImportError:
            logger.warning("peft not installed — skipping LoRA")
            return

        target_modules = self._resolve_lora_target_modules()

        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
        )

        if self._use_crf:
            self.bert = get_peft_model(self.bert, lora_config)
        else:
            self.bert_tc = get_peft_model(self.bert_tc, lora_config)

    def _resolve_lora_target_modules(self) -> list[str]:
        """
        Determine which attention module names to target for LoRA.

        gbert-large uses standard BERT naming (query / value).  If we discover
        the model uses different names we fall back gracefully.
        """
        bert_module = self.bert if self._use_crf else self.bert_tc
        module_names = [n for n, _ in bert_module.named_modules()]

        # Check standard BERT naming first
        has_query = any("query" in n for n in module_names)
        has_value = any("value" in n for n in module_names)

        if has_query and has_value:
            return ["query", "value"]

        # Fallback: GPT-style naming
        has_q_proj = any("q_proj" in n for n in module_names)
        has_v_proj = any("v_proj" in n for n in module_names)
        if has_q_proj and has_v_proj:
            logger.warning("LoRA: using q_proj/v_proj target modules instead of query/value")
            return ["q_proj", "v_proj"]

        logger.warning(
            "LoRA: could not find matching attention module names — LoRA will target no modules"
        )
        return []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """
        Non-CRF path: delegates to BertForTokenClassification and returns
        a TokenClassifierOutput (with .loss and .logits).

        CRF path with labels: returns (loss: scalar, emissions: B x S x 3).
        CRF path without labels: returns list[list[int]] from Viterbi decode.
        """
        if self._use_crf:
            return self._forward_crf(input_ids, attention_mask, labels)
        return self.bert_tc(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def _forward_crf(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        emissions: torch.Tensor = self.classifier(outputs.last_hidden_state)  # B x S x 3

        if labels is not None:
            # Build mask from -100 positions BEFORE modifying labels
            mask = (labels != -100).bool()
            clean_labels = labels.clone()
            clean_labels[clean_labels == -100] = 0

            loss = -self.crf(emissions, clean_labels, mask=mask, reduction="mean")
            return loss, emissions

        # Decode without labels
        return self.crf.decode(emissions, mask=attention_mask.bool())

    # ------------------------------------------------------------------
    # Properties & parameter helpers
    # ------------------------------------------------------------------

    @property
    def use_crf(self) -> bool:
        """Whether the CRF layer is active."""
        return self._use_crf

    def get_bert_parameters(self):
        """Return BERT encoder parameters (for differential LR in optimizer)."""
        if self._use_crf:
            return list(self.bert.parameters())
        return list(self.bert_tc.bert.parameters())

    def get_head_parameters(self):
        """Return classification head parameters (for differential LR in optimizer)."""
        if self._use_crf:
            return list(self.classifier.parameters()) + list(self.crf.parameters())
        return list(self.bert_tc.classifier.parameters())
