"""Tests for the regex baseline extractor covering all 10 German legal reference types."""

from src.evaluation.regex_baseline import RegexBaseline


class TestRegexBaseline:
    """Test that RegexBaseline.extract() finds all reference types."""

    def setup_method(self):
        self.baseline = RegexBaseline()

    def test_paragraph_ref(self, sample_paragraph_ref):
        """extract('Gemäß § 25a KWG gilt') returns span covering '§ 25a KWG'."""
        text = sample_paragraph_ref["text"]
        spans = self.baseline.extract(text)
        assert len(spans) >= 1
        matched = text[spans[0][0]:spans[0][1]]
        assert "§ 25a KWG" in matched

    def test_paragraph_with_absatz(self, sample_paragraph_with_absatz):
        """extract('§ 25a Abs. 1 KWG') returns span covering full reference."""
        text = sample_paragraph_with_absatz["text"]
        spans = self.baseline.extract(text)
        assert len(spans) >= 1
        matched = text[spans[0][0]:spans[0][1]]
        assert "Abs." in matched
        assert "KWG" in matched

    def test_artikel(self, sample_artikel):
        """extract('Art. 6 DSGVO regelt') returns span for 'Art. 6 DSGVO'."""
        text = sample_artikel["text"]
        spans = self.baseline.extract(text)
        assert len(spans) >= 1
        matched = text[spans[0][0]:spans[0][1]]
        assert "Art. 6 DSGVO" in matched

    def test_anhang(self, sample_anhang):
        """extract('Anhang II CRR enthält') returns span for 'Anhang II CRR'."""
        text = sample_anhang["text"]
        spans = self.baseline.extract(text)
        assert len(spans) >= 1
        matched = text[spans[0][0]:spans[0][1]]
        assert "Anhang II" in matched

    def test_verordnung(self, sample_verordnung):
        """extract('EU-Verordnung 648/2012') returns span for the full reference."""
        text = sample_verordnung["text"]
        spans = self.baseline.extract(text)
        assert len(spans) >= 1
        matched = text[spans[0][0]:spans[0][1]]
        assert "Verordnung" in matched
        assert "648/2012" in matched

    def test_multi_paragraph(self, sample_multi_paragraph):
        """extract('§§ 3, 4 UWG') returns at least one span."""
        text = sample_multi_paragraph["text"]
        spans = self.baseline.extract(text)
        assert len(spans) >= 1

    def test_satz(self, sample_satz):
        """extract('§ 5 Abs. 2 S. 1 BGB') returns span including Satz."""
        text = sample_satz["text"]
        spans = self.baseline.extract(text)
        assert len(spans) >= 1
        matched = text[spans[0][0]:spans[0][1]]
        assert "S." in matched

    def test_nr(self, sample_nr):
        """extract('§ 1 Nr. 3 KWG') returns span including Nr."""
        text = sample_nr["text"]
        spans = self.baseline.extract(text)
        assert len(spans) >= 1
        matched = text[spans[0][0]:spans[0][1]]
        assert "Nr." in matched

    def test_lit(self, sample_lit):
        """extract('§ 2 Abs. 1 lit. a BGB') returns span including lit."""
        text = sample_lit["text"]
        spans = self.baseline.extract(text)
        assert len(spans) >= 1
        matched = text[spans[0][0]:spans[0][1]]
        assert "lit." in matched

    def test_tz(self, sample_tz):
        """extract('Tz. 4 MaRisk') returns span for Teilziffer reference."""
        text = sample_tz["text"]
        spans = self.baseline.extract(text)
        assert len(spans) >= 1
        matched = text[spans[0][0]:spans[0][1]]
        assert "Tz." in matched

    def test_no_reference(self, sample_no_reference):
        """extract('Dies ist ein normaler Satz.') returns empty list."""
        text = sample_no_reference["text"]
        spans = self.baseline.extract(text)
        assert spans == []
