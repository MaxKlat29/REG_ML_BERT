"""Regex baseline for German legal reference extraction.

Uses the `regex` PyPI package (not stdlib `re`) for Unicode support
and overlapped matching of multi-section references (§§).
"""

import regex

# Pattern covers all 10 German legal reference types:
# §, §§, Art./Artikel, Abs./Absatz, Nr./Nummer, lit., Satz/S.,
# Tz./Teilziffer, Anhang, Verordnung/EU-Verordnung
GERMAN_LEGAL_REF_PATTERN = regex.compile(
    r"""
    (?:
        # Tz. / Teilziffer references (must come before § to avoid partial matches)
        (?:Tz\.|Teilziffer)\s*\d+
        (?:\s+[A-Z][A-Za-zÄÖÜäöü]*[A-Z][A-Za-zÄÖÜäöü]*)?
        |
        # § and §§ patterns (single and multi-section)
        §§?\s*\d+(?:\w*\b)?
        (?:
            (?:\s*,\s*\d+)*
        )?
        (?:
            (?:\s*(?:Abs\.|Absatz)\s*\d+(?:\w*\b)?)?
            (?:\s*(?:S\.|Satz)\s*\d+)?
            (?:\s*(?:Nr\.|Nummer)\s*\d+(?:\w*\b)?)?
            (?:\s*lit\.\s*[a-z])?
            (?:\s*(?:Tz\.|Teilziffer)\s*\d+)?
        )*
        (?:\s+[A-Z][A-Za-zÄÖÜäöü]*[A-Z][A-Za-zÄÖÜäöü]*)?
        |
        # Art. / Artikel patterns
        (?:Art\.|Artikel)\s*\d+(?:\w*\b)?
        (?:
            (?:\s*(?:Abs\.|Absatz)\s*\d+(?:\w*\b)?)?
            (?:\s*(?:S\.|Satz)\s*\d+)?
            (?:\s*(?:Nr\.|Nummer)\s*\d+(?:\w*\b)?)?
            (?:\s*lit\.\s*[a-z])?
            (?:\s*(?:Tz\.|Teilziffer)\s*\d+)?
        )*
        (?:\s+[A-Z][A-Za-zÄÖÜäöü]*[A-Z][A-Za-zÄÖÜäöü]*)?
        |
        # Anhang (Appendix references)
        Anhang\s+(?:[IVX]+|\d+)
        (?:\s+[A-Z][A-Za-zÄÖÜäöü]*[A-Z][A-Za-zÄÖÜäöü]*)?
        |
        # Verordnung (Regulation references)
        (?:EU-)?Verordnung\s*(?:Nr\.\s*)?\d+/\d+(?:/[A-Z]+)?
    )
    """,
    regex.VERBOSE | regex.UNICODE,
)


class RegexBaseline:
    """Extracts German legal references from text using regex patterns.

    Returns character-offset spans for all detected references.
    """

    def __init__(self):
        self.pattern = GERMAN_LEGAL_REF_PATTERN

    def extract(self, text: str) -> list[tuple[int, int]]:
        """Extract legal reference spans from text.

        Args:
            text: Input text to search for legal references.

        Returns:
            List of (start, end) character offset tuples.
        """
        return [(m.start(), m.end()) for m in self.pattern.finditer(text)]
