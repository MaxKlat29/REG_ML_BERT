"""Tests for the seqeval metrics wrapper and BIO conversion."""

from src.evaluation.metrics import spans_to_bio, compute_entity_metrics


class TestSpansToBio:
    """Test BIO label generation from character spans."""

    def test_spans_to_bio_basic(self):
        """spans_to_bio on known text/spans produces correct B-REF/I-REF/O sequence."""
        text = "Gemäß § 25a KWG gilt"
        # "§ 25a KWG" starts at index 6, ends at index 15
        ref_text = "§ 25a KWG"
        start = text.index(ref_text)
        end = start + len(ref_text)
        spans = [(start, end)]

        tokens, labels = spans_to_bio(text, spans)

        assert tokens == ["Gemäß", "§", "25a", "KWG", "gilt"]
        assert labels == ["O", "B-REF", "I-REF", "I-REF", "O"]


class TestEntityMetrics:
    """Test seqeval metrics wrapper."""

    def test_entity_metrics_perfect(self):
        """Identical y_true/y_pred returns P=R=F1=1.0."""
        y_true = [["O", "B-REF", "I-REF", "O"]]
        y_pred = [["O", "B-REF", "I-REF", "O"]]

        metrics = compute_entity_metrics(y_true, y_pred)

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert "report" in metrics

    def test_entity_metrics_partial(self):
        """One missed entity returns recall=0.5."""
        y_true = [["O", "B-REF", "I-REF", "O", "B-REF", "O"]]
        y_pred = [["O", "B-REF", "I-REF", "O", "O",     "O"]]

        metrics = compute_entity_metrics(y_true, y_pred)

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 0.5
        assert "report" in metrics
