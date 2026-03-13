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

    # Per-type compiled patterns for typed extraction.
    # Each entry maps a reference type name to a compiled regex pattern.
    # Ordered so more specific (multi-token) patterns come before generic ones.
    TYPED_PATTERNS: dict[str, object] = {
        "TEILZIFFER": regex.compile(
            r"(?:Tz\.|Teilziffer)\s*\d+(?:\s+[A-Z][A-Za-zÄÖÜäöü]*[A-Z][A-Za-zÄÖÜäöü]*)?",
            regex.UNICODE,
        ),
        "PARAGRAPH": regex.compile(
            r"""
            §§?\s*\d+(?:\w*\b)?
            (?:(?:\s*,\s*\d+)*)?
            (?:
                (?:\s*(?:Abs\.|Absatz)\s*\d+(?:\w*\b)?)?
                (?:\s*(?:S\.|Satz)\s*\d+)?
                (?:\s*(?:Nr\.|Nummer)\s*\d+(?:\w*\b)?)?
                (?:\s*lit\.\s*[a-z])?
                (?:\s*(?:Tz\.|Teilziffer)\s*\d+)?
            )*
            (?:\s+[A-Z][A-Za-zÄÖÜäöü]*[A-Z][A-Za-zÄÖÜäöü]*)?
            """,
            regex.VERBOSE | regex.UNICODE,
        ),
        "ARTIKEL": regex.compile(
            r"""
            (?:Art\.|Artikel)\s*\d+(?:\w*\b)?
            (?:
                (?:\s*(?:Abs\.|Absatz)\s*\d+(?:\w*\b)?)?
                (?:\s*(?:S\.|Satz)\s*\d+)?
                (?:\s*(?:Nr\.|Nummer)\s*\d+(?:\w*\b)?)?
                (?:\s*lit\.\s*[a-z])?
                (?:\s*(?:Tz\.|Teilziffer)\s*\d+)?
            )*
            (?:\s+[A-Z][A-Za-zÄÖÜäöü]*[A-Z][A-Za-zÄÖÜäöü]*)?
            """,
            regex.VERBOSE | regex.UNICODE,
        ),
        "ANHANG": regex.compile(
            r"Anhang\s+(?:[IVX]+|\d+)(?:\s+[A-Z][A-Za-zÄÖÜäöü]*[A-Z][A-Za-zÄÖÜäöü]*)?",
            regex.UNICODE,
        ),
        "VERORDNUNG": regex.compile(
            r"(?:EU-)?Verordnung\s*(?:Nr\.\s*)?\d+/\d+(?:/[A-Z]+)?",
            regex.UNICODE,
        ),
    }

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

    def extract_typed(self, text: str) -> list[tuple[int, int, str]]:
        """Extract legal reference spans with their type labels.

        Iterates all TYPED_PATTERNS, collects matches with type name, sorts
        by start position, and deduplicates overlapping spans (keeping the
        longest span at each position).

        Args:
            text: Input text to search for legal references.

        Returns:
            Sorted list of (start, end, type_name) triples.
        """
        candidates: list[tuple[int, int, str]] = []
        for type_name, pattern in self.TYPED_PATTERNS.items():
            for m in pattern.finditer(text):
                candidates.append((m.start(), m.end(), type_name))

        # Sort by start position, then by span length descending (longest first)
        candidates.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        # Deduplicate: remove spans overlapping with already-accepted spans
        result: list[tuple[int, int, str]] = []
        for start, end, typ in candidates:
            # Check overlap with any already accepted span
            overlaps = False
            for a_start, a_end, _ in result:
                intersection = max(0, min(end, a_end) - max(start, a_start))
                if intersection > 0:
                    overlaps = True
                    break
            if not overlaps:
                result.append((start, end, typ))

        return result
