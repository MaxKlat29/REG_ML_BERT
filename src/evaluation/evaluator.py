"""Evaluator that runs the regex baseline and computes entity-level metrics."""

from omegaconf import DictConfig

from src.evaluation.regex_baseline import RegexBaseline
from src.evaluation.metrics import spans_to_bio, compute_entity_metrics


class Evaluator:
    """Runs baseline extraction and computes metrics against gold spans."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.baseline = RegexBaseline()

    def evaluate_baseline(self, samples: list[dict]) -> dict:
        """Evaluate the regex baseline against gold-annotated samples.

        Args:
            samples: List of dicts with keys:
                - "text": str — the input sentence
                - "spans": list[tuple[int, int]] — gold character-offset spans

        Returns:
            Dict with precision, recall, f1, report keys.
        """
        all_true: list[list[str]] = []
        all_pred: list[list[str]] = []

        for sample in samples:
            text = sample["text"]
            gold_spans = sample["spans"]
            pred_spans = self.baseline.extract(text)

            _, gold_labels = spans_to_bio(text, gold_spans)
            _, pred_labels = spans_to_bio(text, pred_spans)

            all_true.append(gold_labels)
            all_pred.append(pred_labels)

        return compute_entity_metrics(all_true, all_pred)

    def format_report(self, metrics: dict) -> str:
        """Pretty-print Precision/Recall/F1 table.

        Args:
            metrics: Dict returned by evaluate_baseline().

        Returns:
            Formatted string with evaluation results.
        """
        lines = [
            "=" * 50,
            "  Regex Baseline Evaluation Report",
            "=" * 50,
            "",
            f"  Precision:  {metrics['precision']:.4f}",
            f"  Recall:     {metrics['recall']:.4f}",
            f"  F1-Score:   {metrics['f1']:.4f}",
            "",
            "  Detailed Report:",
            "-" * 50,
            metrics["report"],
            "=" * 50,
        ]
        return "\n".join(lines)
