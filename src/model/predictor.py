"""Predictor for RegulatoryNERModel — single-text and batch inference.

Provides:
  - PredictedSpan: dataclass for character-offset prediction results
  - Predictor: loads a checkpoint and runs inference, returning PredictedSpan lists
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from src.data.bio_converter import LABEL_B_REF, LABEL_I_REF, LABEL_O
from src.evaluation.metrics import decode_bio_to_char_spans
from src.model.ner_model import RegulatoryNERModel
from src.model.trainer import CHECKPOINT_BASE, load_checkpoint
from transformers import BertTokenizerFast

# Global inference settings
REF_THRESHOLD = 0.2
EXPAND_WORDS = 1

logger = logging.getLogger(__name__)


@dataclass
class PredictedSpan:
    """A single predicted reference span with character offsets and confidence.

    Attributes:
        start: Inclusive character offset into the original text.
        end: Exclusive character offset into the original text.
        text: Substring of original text from start to end.
        confidence: Mean softmax probability of the predicted label for tokens
            in this span. Always 1.0 for CRF models (Viterbi has no marginals).
    """

    start: int
    end: int
    text: str
    confidence: float


class Predictor:
    """Loads a trained RegulatoryNERModel checkpoint and runs inference.

    Supports single-text and batch prediction returning character-offset
    PredictedSpan objects.

    Args:
        checkpoint_path: Path to a .pt checkpoint file produced by save_checkpoint().
        config: OmegaConf / SimpleNamespace config with model and data sub-configs.
        device: torch.device on which inference is executed.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config,
        device: torch.device,
    ) -> None:
        self._device = device
        self._max_length: int = int(config.data.max_seq_length)

        # Build model and load weights
        model = RegulatoryNERModel(config)
        load_checkpoint(Path(checkpoint_path), model)
        model.eval()
        model.to(device)
        self._model = model

        # Tokenizer — BertTokenizerFast per STATE.md decision (AutoTokenizer fails on gbert-large)
        self._tokenizer = BertTokenizerFast.from_pretrained(config.model.name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, text: str) -> list[PredictedSpan]:
        """Run inference on a single text and return predicted reference spans.

        Uses soft thresholding: if P(B-REF)+P(I-REF) > ref_threshold, prefer
        REF label over O. After decoding, expands each span by expand_words
        words in each direction.

        Args:
            text: Raw input text to extract legal references from.

        Returns:
            List of PredictedSpan objects sorted by start offset.
        """
        encoding = self._tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            padding="max_length",
            max_length=self._max_length,
        )

        input_ids = torch.tensor([encoding["input_ids"]], dtype=torch.long).to(self._device)
        attention_mask = torch.tensor([encoding["attention_mask"]], dtype=torch.long).to(self._device)
        offset_mapping: list[tuple[int, int]] = list(encoding["offset_mapping"])

        with torch.no_grad():
            output = self._model(input_ids, attention_mask)

        use_crf = getattr(self._model, "_use_crf", False)

        if use_crf:
            pred_ids: list[int] = output[0]
            char_spans = decode_bio_to_char_spans(pred_ids, offset_mapping)
            spans = self._expand_and_build_spans(text, char_spans, probs=None, pred_ids=pred_ids, offset_mapping=offset_mapping)
            return spans

        # Non-CRF: soft threshold instead of argmax
        logits: torch.Tensor = output.logits  # (1, seq_len, 3)
        probs = torch.softmax(logits, dim=-1)[0]  # (seq_len, 3)
        pred_ids = self._apply_threshold(probs)

        char_spans = decode_bio_to_char_spans(pred_ids, offset_mapping)

        return self._expand_and_build_spans(text, char_spans, probs=probs, pred_ids=pred_ids, offset_mapping=offset_mapping)

    def _apply_threshold(self, probs: torch.Tensor) -> list[int]:
        """Apply soft threshold: prefer B/I-REF over O when ref prob > threshold.

        Also enforces valid BIO transitions: I-REF can only follow B-REF or I-REF.
        """
        pred_ids = []
        prev_label = LABEL_O
        for i in range(probs.shape[0]):
            p_o, p_b, p_i = probs[i].tolist()
            p_ref = p_b + p_i

            if p_ref > REF_THRESHOLD:
                # Prefer ref label — pick B vs I
                if prev_label in (LABEL_B_REF, LABEL_I_REF) and p_i > p_b:
                    label = LABEL_I_REF
                else:
                    label = LABEL_B_REF
            else:
                label = probs[i].argmax().item()

            # Enforce valid BIO: I-REF can't follow O
            if label == LABEL_I_REF and prev_label == LABEL_O:
                label = LABEL_B_REF

            pred_ids.append(label)
            prev_label = label
        return pred_ids

    def _expand_and_build_spans(
        self,
        text: str,
        char_spans: list[tuple[int, int]],
        probs: torch.Tensor | None,
        pred_ids: list[int],
        offset_mapping: list[tuple[int, int]],
    ) -> list[PredictedSpan]:
        """Expand spans by N words in each direction, compute confidence, merge overlaps."""
        # Find word boundaries in the text
        word_bounds = [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]

        expanded = []
        for span_start, span_end in char_spans:
            new_start, new_end = self._expand_span(span_start, span_end, word_bounds)

            # Confidence from token probs
            confidence = 1.0
            if probs is not None:
                span_probs: list[float] = []
                for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
                    if tok_start == 0 and tok_end == 0:
                        continue
                    if tok_start < span_end and tok_end > span_start:
                        label = pred_ids[tok_idx]
                        if label in (LABEL_B_REF, LABEL_I_REF):
                            span_probs.append(probs[tok_idx, label].item())
                confidence = sum(span_probs) / len(span_probs) if span_probs else 0.0

            expanded.append((new_start, new_end, confidence))

        # Merge overlapping/adjacent spans
        merged = self._merge_overlapping(expanded)

        return [
            PredictedSpan(start=s, end=e, text=text[s:e], confidence=c)
            for s, e, c in merged
        ]

    def _expand_span(self, start: int, end: int, word_bounds: list[tuple[int, int]]) -> tuple[int, int]:
        """Expand a char span by EXPAND_WORDS words before and after."""
        if not word_bounds or EXPAND_WORDS <= 0:
            return start, end

        # Find first word overlapping span start
        first_word_idx = 0
        for i, (ws, we) in enumerate(word_bounds):
            if we > start:
                first_word_idx = i
                break

        # Find last word overlapping span end
        last_word_idx = len(word_bounds) - 1
        for i, (ws, we) in enumerate(word_bounds):
            if ws >= end:
                last_word_idx = i - 1
                break

        # Expand
        new_first = max(0, first_word_idx - EXPAND_WORDS)
        new_last = min(len(word_bounds) - 1, last_word_idx + EXPAND_WORDS)

        return word_bounds[new_first][0], word_bounds[new_last][1]

    @staticmethod
    def _merge_overlapping(spans: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
        """Merge overlapping or adjacent spans, averaging confidence."""
        if not spans:
            return []
        sorted_spans = sorted(spans, key=lambda x: x[0])
        merged = [sorted_spans[0]]
        for s, e, c in sorted_spans[1:]:
            prev_s, prev_e, prev_c = merged[-1]
            if s <= prev_e:  # overlapping or adjacent
                merged[-1] = (prev_s, max(prev_e, e), (prev_c + c) / 2)
            else:
                merged.append((s, e, c))
        return merged

    def predict_batch(self, texts: list[str]) -> list[list[PredictedSpan]]:
        """Run inference on multiple texts.

        Calls predict() for each text. Batch tokenization with padding is
        possible but a loop is simpler and sufficient for PoC-scale inference.

        Args:
            texts: List of raw input texts.

        Returns:
            List of lists of PredictedSpan objects, one inner list per input text.
        """
        return [self.predict(text) for text in texts]

    @classmethod
    def find_latest_checkpoint(cls, base_dir: Path | None = None) -> Path:
        """Find the most recently modified .pt file under base_dir.

        Searches recursively using rglob("*.pt").

        Args:
            base_dir: Root directory to search. Defaults to CHECKPOINT_BASE
                from trainer.py.

        Returns:
            Path to the most recently modified .pt file.

        Raises:
            FileNotFoundError: If no .pt files are found under base_dir.
        """
        search_root = Path(base_dir) if base_dir is not None else CHECKPOINT_BASE
        candidates = list(search_root.rglob("*.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"No .pt checkpoint files found under '{search_root}'. "
                "Run 'python run.py train' first."
            )
        return max(candidates, key=lambda p: p.stat().st_mtime)
