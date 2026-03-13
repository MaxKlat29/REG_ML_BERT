"""Predictor for RegulatoryNERModel — single-text and batch inference.

Provides:
  - PredictedSpan: dataclass for character-offset prediction results
  - Predictor: loads a checkpoint and runs inference, returning PredictedSpan lists
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from src.data.bio_converter import LABEL_B_REF, LABEL_I_REF, LABEL_O
from src.evaluation.metrics import decode_bio_to_char_spans
from src.model.ner_model import RegulatoryNERModel
from src.model.trainer import CHECKPOINT_BASE, load_checkpoint
from transformers import BertTokenizerFast

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

        Tokenizes text, runs model forward under torch.no_grad(), decodes
        BIO labels back to character spans, and computes confidence scores.

        For CRF models the Viterbi decoder does not produce marginals, so
        confidence is always set to 1.0.

        Args:
            text: Raw input text to extract legal references from.

        Returns:
            List of PredictedSpan objects sorted by start offset. Empty list
            if no references are found.
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
            # CRF decode: output is list[list[int]] from Viterbi
            pred_ids: list[int] = output[0]
            char_spans = decode_bio_to_char_spans(pred_ids, offset_mapping)
            return [
                PredictedSpan(
                    start=start,
                    end=end,
                    text=text[start:end],
                    confidence=1.0,
                )
                for start, end in char_spans
            ]

        # Non-CRF: output has .logits (B x S x C)
        logits: torch.Tensor = output.logits  # shape (1, seq_len, 3)
        probs = torch.softmax(logits, dim=-1)[0]  # (seq_len, 3)
        pred_ids_tensor = probs.argmax(dim=-1)     # (seq_len,)
        pred_ids = pred_ids_tensor.tolist()

        char_spans = decode_bio_to_char_spans(pred_ids, offset_mapping)

        # Build a map from char-span -> token indices for confidence calculation
        # We need to find which token positions correspond to each char span
        spans_with_confidence = []
        for span_start, span_end in char_spans:
            # Find token indices whose offsets are within [span_start, span_end)
            span_probs: list[float] = []
            for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
                # Skip special/padding tokens
                if tok_start == 0 and tok_end == 0:
                    continue
                # Token is within the span
                if tok_start >= span_start and tok_end <= span_end:
                    label = pred_ids[tok_idx]
                    if label in (LABEL_B_REF, LABEL_I_REF):
                        span_probs.append(probs[tok_idx, label].item())

            confidence = sum(span_probs) / len(span_probs) if span_probs else 0.0

            spans_with_confidence.append(
                PredictedSpan(
                    start=span_start,
                    end=span_end,
                    text=text[span_start:span_end],
                    confidence=float(confidence),
                )
            )

        return spans_with_confidence

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
