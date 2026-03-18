"""
Async Ollama LLM client with retry logic, ref-tag parser, and prompt builder.

Generates diverse German documents (regulatory, contracts, policies, SLAs, etc.)
with cross-reference annotations for NER training.

Exports:
    call_ollama           - Async HTTP call to local Ollama chat endpoint
    parse_ref_tags        - Extract clean text + character-offset spans from <ref>...</ref>
    build_generation_prompt - Build German document generation prompt with cross-references
    get_context_for_seed  - Select document type + scenario deterministically from seed
    DOCUMENT_CONTEXTS     - List of document type / scenario pairs
    RetryableAPIError     - Exception raised for retryable HTTP status codes
"""
from __future__ import annotations

import os
import re
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Document contexts: (document_type, scenario_hint)
# Each produces distinct cross-reference patterns the model must learn.
DOCUMENT_CONTEXTS: list[tuple[str, str]] = [
    # --- Regulatory / Legal ---
    ("Regulatorischer Fachtext", "BGB — Bürgerliches Gesetzbuch"),
    ("Regulatorischer Fachtext", "KWG — Kreditwesengesetz"),
    ("Regulatorischer Fachtext", "MaRisk — Mindestanforderungen Risikomanagement"),
    ("Regulatorischer Fachtext", "DORA — Digital Operational Resilience Act"),
    ("Regulatorischer Fachtext", "DSGVO — Datenschutz-Grundverordnung"),
    ("Regulatorischer Fachtext", "CRR — Capital Requirements Regulation"),
    ("Regulatorischer Fachtext", "HGB — Handelsgesetzbuch"),
    ("Regulatorischer Fachtext", "WpHG — Wertpapierhandelsgesetz"),
    ("Regulatorischer Fachtext", "VAG — Versicherungsaufsichtsgesetz"),
    ("Regulatorischer Fachtext", "ZAG — Zahlungsdiensteaufsichtsgesetz"),
    ("Regulatorischer Fachtext", "GwG — Geldwäschegesetz"),
    ("Regulatorischer Fachtext", "SAG — Sanierungs- und Abwicklungsgesetz"),
    ("Regulatorischer Fachtext", "KAGB — Kapitalanlagegesetzbuch"),
    ("Regulatorischer Fachtext", "AO — Abgabenordnung"),
    ("Regulatorischer Fachtext", "EStG — Einkommensteuergesetz"),
    ("Regulatorischer Fachtext", "UStG — Umsatzsteuergesetz"),
    ("Regulatorischer Fachtext", "GmbHG — GmbH-Gesetz"),
    ("Regulatorischer Fachtext", "AktG — Aktiengesetz"),
    ("Regulatorischer Fachtext", "InsO — Insolvenzordnung"),
    ("Regulatorischer Fachtext", "ZPO — Zivilprozessordnung"),
    # --- Contracts ---
    ("Dienstleistungsvertrag", "IT-Outsourcing mit SLA-Anhängen"),
    ("Dienstleistungsvertrag", "Beratungsvertrag mit Leistungsbeschreibung"),
    ("Kaufvertrag", "Unternehmenskauf (Share Deal) mit Garantien"),
    ("Kaufvertrag", "Asset Deal mit Anlageverzeichnis"),
    ("Mietvertrag", "Gewerbemietvertrag mit Nebenkostenregelung"),
    ("Lizenzvertrag", "Softwarelizenz mit Nutzungsbedingungen"),
    ("Rahmenvertrag", "IT-Rahmenvertrag mit Einzelaufträgen und Anlagen"),
    ("Gesellschaftsvertrag", "GmbH-Gesellschaftsvertrag mit Gesellschafterliste"),
    ("Kooperationsvertrag", "Joint-Venture-Vereinbarung mit Gewinnverteilung"),
    ("Arbeitsvertrag", "Anstellungsvertrag mit Verweis auf Betriebsvereinbarungen"),
    ("Darlehensvertrag", "Kreditvertrag mit Sicherheitenverzeichnis"),
    ("Geheimhaltungsvereinbarung", "NDA mit Definition vertraulicher Informationen"),
    # --- SLAs & Service Documents ---
    ("Service Level Agreement", "Cloud-Hosting SLA mit Verfügbarkeitsgarantien"),
    ("Service Level Agreement", "Managed-Services SLA mit Reaktionszeiten"),
    ("Service Level Agreement", "IT-Support SLA mit Eskalationsstufen"),
    ("Leistungsbeschreibung", "Pflichtenheft Software-Entwicklung"),
    ("Betriebshandbuch", "IT-Betriebshandbuch mit Prozessreferenzen"),
    # --- Corporate Policies & Internal Documents ---
    ("Unternehmensrichtlinie", "Datenschutzrichtlinie mit Gesetzesverweisen"),
    ("Unternehmensrichtlinie", "Compliance-Richtlinie mit internen Verweisen"),
    ("Unternehmensrichtlinie", "IT-Sicherheitsrichtlinie nach ISO 27001"),
    ("Unternehmensrichtlinie", "Anti-Geldwäsche-Richtlinie"),
    ("Arbeitsanweisung", "Prozessbeschreibung mit Querverweisen auf andere Dokumente"),
    ("Organisationshandbuch", "Aufbauorganisation mit Zuständigkeitsverweisen"),
    # --- Audit & Compliance Reports ---
    ("Prüfungsbericht", "Jahresabschlussprüfung mit Normverweisen"),
    ("Prüfungsbericht", "IT-Audit nach IDW PS 330"),
    ("Compliance-Bericht", "Geldwäsche-Prüfung mit Feststellungen"),
    ("Risikobericht", "Risikoinventur mit Maßnahmenverweisen"),
    # --- Terms & Conditions ---
    ("AGB", "Allgemeine Geschäftsbedingungen Online-Shop"),
    ("AGB", "Allgemeine Einkaufsbedingungen B2B"),
    ("Nutzungsbedingungen", "SaaS-Plattform Nutzungsbedingungen"),
    ("Datenschutzerklärung", "Website-Datenschutzerklärung nach DSGVO"),
    # --- Board & Corporate Governance ---
    ("Vorstandsbeschluss", "Beschluss zur Kreditvergabe mit Satzungsverweisen"),
    ("Aufsichtsratsbericht", "Bericht mit Verweisen auf Geschäftsordnung"),
    ("Geschäftsordnung", "Geschäftsordnung Vorstand mit Kompetenzregeln"),
    ("Satzung", "Vereinssatzung mit internen Querverweisen"),
    # --- Technical & Project Documentation ---
    ("Technische Dokumentation", "Systemarchitektur mit Schnittstellenverweisen"),
    ("Projektdokumentation", "Projektplan mit Meilenstein-Querverweisen"),
    ("Ausschreibung", "Vergabeunterlagen nach VOB/VOL"),
    ("Gutachten", "Sachverständigengutachten mit Normbezügen"),
]

RETRYABLE_STATUS: frozenset[int] = frozenset({408, 429, 502, 503})

OLLAMA_DEFAULT_ENDPOINT = "http://localhost:11434"

_REF_PATTERN = re.compile(r"<ref>(.*?)</ref>", re.DOTALL)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class RetryableAPIError(Exception):
    """Raised when Ollama returns a retryable HTTP status code."""

    def __init__(self, status_code: int, message: str = "") -> None:
        self.status_code = status_code
        super().__init__(message or f"Retryable API error: HTTP {status_code}")


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
    retry=retry_if_exception_type(RetryableAPIError),
    reraise=True,
)
async def call_ollama(
    client: httpx.AsyncClient,
    model: str,
    messages: list[dict[str, str]],
    seed: int,
    endpoint: str | None = None,
) -> str:
    """
    Call the Ollama chat API and return the assistant's text.

    Args:
        client:   An httpx.AsyncClient instance (caller manages lifecycle).
        model:    Ollama model name, e.g. "qwen2.5:14b".
        messages: List of chat message dicts with "role" and "content" keys.
        seed:     Integer seed passed to the model for reproducibility.
        endpoint: Ollama base URL. Falls back to OLLAMA_ENDPOINT env var
                  or localhost:11434.

    Returns:
        The assistant message content string.

    Raises:
        RetryableAPIError: If a retryable status code is returned after all attempts.
    """
    base_url = (
        endpoint
        or os.environ.get("OLLAMA_ENDPOINT", "")
        or OLLAMA_DEFAULT_ENDPOINT
    )
    url = f"{base_url}/api/chat"

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "seed": seed,
            "temperature": 0.8,
            "top_p": 0.9,
        },
    }

    response = await client.post(url, json=payload, timeout=120.0)

    if response.status_code in RETRYABLE_STATUS:
        raise RetryableAPIError(response.status_code)

    if response.status_code >= 400:
        response.raise_for_status()

    return response.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Ref-tag parser
# ---------------------------------------------------------------------------

def parse_ref_tags(tagged_text: str) -> tuple[str, list[tuple[int, int]]]:
    """
    Parse <ref>...</ref> tags from tagged_text and return clean text with spans.

    Args:
        tagged_text: Text that may contain one or more <ref>...</ref> tags.

    Returns:
        A tuple of:
          - clean_text: The original text with all <ref>...</ref> tags removed.
          - spans: List of (start, end) character-offset tuples into clean_text,
                   one per ref tag, in document order.

    Each span satisfies:
        clean_text[start:end] == content inside the corresponding <ref> tag.
    """
    spans: list[tuple[int, int]] = []
    clean_parts: list[str] = []
    cursor = 0          # position in tagged_text
    clean_offset = 0    # running offset in clean_text being built

    for match in _REF_PATTERN.finditer(tagged_text):
        tag_start = match.start()
        tag_end = match.end()
        ref_content = match.group(1).strip()

        # Skip empty or whitespace-only refs (LLM artefact)
        if not ref_content:
            # Still need to consume the tag from the input
            before = tagged_text[cursor:tag_start]
            clean_parts.append(before)
            clean_offset += len(before)
            cursor = tag_end
            continue

        # Append text before this tag to clean output
        before = tagged_text[cursor:tag_start]
        clean_parts.append(before)
        clean_offset += len(before)

        # Record span for ref content
        span_start = clean_offset
        span_end = clean_offset + len(ref_content)
        spans.append((span_start, span_end))

        clean_parts.append(ref_content)
        clean_offset += len(ref_content)
        cursor = tag_end

    # Append any trailing text after last tag
    clean_parts.append(tagged_text[cursor:])
    clean_text = "".join(clean_parts)

    # Validate spans
    for start, end in spans:
        assert end > start, f"Empty span [{start}:{end}]"
        assert "<ref>" not in clean_text[start:end], "Span contains leftover tag"

    return clean_text, spans


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_generation_prompt(doc_type: str, scenario: str, include_references: bool = True) -> str:
    """
    Build a German prompt instructing the LLM to generate a document excerpt
    with annotated cross-references.

    The prompt covers ALL reference types the model should learn to detect:
    - Gesetzesverweise (§, Art., Abs., Nr., lit., Satz, Tz.)
    - Vertragsklauseln (Ziffer, Punkt, Klausel, Abschnitt)
    - Anhänge/Anlagen (Anhang, Anlage, Exhibit, Schedule, Appendix)
    - Interne Dokumentverweise (siehe Kapitel, vgl. Abschnitt)
    - SLA-/Leistungskennzahlen (SLA Ziffer, Service Level)
    - Normen/Standards (ISO, DIN, IDW)

    Args:
        doc_type:           Document type, e.g. "Dienstleistungsvertrag".
        scenario:           Scenario hint, e.g. "IT-Outsourcing mit SLA-Anhängen".
        include_references: If True, ask for <ref>-tagged references.
                            If False, ask for text with NO references (negative sample).

    Returns:
        A German prompt string ready to use as a user message.
    """
    if include_references:
        return (
            f"Schreiben Sie einen realistischen deutschen Textauszug aus einem Dokument "
            f"vom Typ '{doc_type}' zum Thema: {scenario}.\n\n"
            f"Der Text soll 1-3 Absätze (80-200 Wörter) umfassen und wie ein echtes "
            f"Dokument klingen — professionell, sachlich, detailliert.\n\n"
            f"WICHTIG: Markieren Sie JEDEN Querverweis mit <ref>...</ref> XML-Tags. "
            f"Querverweise sind ALLE Stellen, die auf andere Textstellen, Dokumente, "
            f"Normen oder Abschnitte verweisen. Dazu gehören:\n"
            f"- Gesetzesverweise: § 25a KWG, Art. 5 DSGVO, §§ 305-310 BGB\n"
            f"- Vertragsklauseln: Ziffer 3.1, Punkt 4.2, Klausel 7\n"
            f"- Anhänge/Anlagen: Anhang A, Anlage 3, Anlage 'Leistungsbeschreibung'\n"
            f"- Abschnitte im Dokument: Abschnitt 4.2, Kapitel 3, Teil B\n"
            f"- SLA-Verweise: SLA Ziffer 2.1, Service Level gemäß Anlage 2\n"
            f"- Normen/Standards: ISO 27001, DIN EN 62305, IDW PS 330\n"
            f"- Absatz/Nummer-Verweise: Abs. 1, Nr. 3, lit. a, Satz 2\n\n"
            f"Beispiel:\n"
            f"'Der Auftragnehmer verpflichtet sich gemäß <ref>Ziffer 5.1</ref> dieses "
            f"Vertrages, die in <ref>Anlage 2 (Leistungsbeschreibung)</ref> definierten "
            f"Services zu erbringen. Die Verfügbarkeit richtet sich nach "
            f"<ref>SLA Ziffer 3.2</ref>. Bei Verstößen gegen <ref>§ 280 Abs. 1 BGB</ref> "
            f"gelten die Regelungen in <ref>Abschnitt 8 (Haftung)</ref>.'\n\n"
            f"Verwenden Sie möglichst viele verschiedene, realistische Querverweise. "
            f"Jeder Verweis muss vollständig im <ref>-Tag stehen."
        )
    else:
        return (
            f"Schreiben Sie einen realistischen deutschen Textauszug aus einem Dokument "
            f"vom Typ '{doc_type}' zum Thema: {scenario}.\n\n"
            f"Der Text soll 1-3 Absätze (80-200 Wörter) umfassen.\n\n"
            f"WICHTIG: Verwenden Sie KEINE Querverweise jeglicher Art. Keine §-Zeichen, "
            f"keine Art./Artikel, keine 'Ziffer', 'Punkt', 'Abschnitt', 'Kapitel', "
            f"'Anhang', 'Anlage', 'siehe', 'vgl.', keine Normverweise (ISO, DIN), "
            f"keine Klauselreferenzen. "
            f"Der Text soll sachlich und fachlich korrekt sein, aber rein beschreibend "
            f"ohne jegliche Verweise auf andere Textstellen oder Dokumente."
        )


# ---------------------------------------------------------------------------
# Context rotation (replaces domain rotation)
# ---------------------------------------------------------------------------

def get_context_for_seed(seed: int) -> tuple[str, str]:
    """
    Select a document context deterministically based on seed.

    Args:
        seed: Integer seed value.

    Returns:
        A (doc_type, scenario) tuple from DOCUMENT_CONTEXTS.
    """
    return DOCUMENT_CONTEXTS[seed % len(DOCUMENT_CONTEXTS)]


# --- Backwards compatibility aliases ---
DOMAIN_LIST = [ctx[1].split(" — ")[0] if " — " in ctx[1] else ctx[1] for ctx in DOCUMENT_CONTEXTS]

def get_domain_for_seed(seed: int) -> str:
    """Backwards-compatible alias returning a domain string for the given seed.

    Args:
        seed: Integer seed value.

    Returns:
        A string combining document type and scenario.
    """
    doc_type, scenario = get_context_for_seed(seed)
    return f"{doc_type}: {scenario}"
