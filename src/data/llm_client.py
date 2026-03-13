"""
Async OpenRouter LLM client with retry logic, ref-tag parser, and prompt builder.

Exports:
    call_openrouter       - Async HTTP call to OpenRouter chat completions endpoint
    parse_ref_tags        - Extract clean text + character-offset spans from <ref>...</ref>
    build_generation_prompt - Build German regulatory text generation prompt
    get_domain_for_seed   - Select regulatory domain deterministically from seed
    DOMAIN_LIST           - List of German regulatory domain abbreviations
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

DOMAIN_LIST: list[str] = [
    "BGB",
    "KWG",
    "MaRisk",
    "DORA",
    "DSGVO",
    "CRR",
    "HGB",
    "WpHG",
    "VAG",
    "ZAG",
    "GwG",
    "SAG",
    "KAGB",
]

RETRYABLE_STATUS: frozenset[int] = frozenset({408, 429, 502, 503})

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

_REF_PATTERN = re.compile(r"<ref>(.*?)</ref>", re.DOTALL)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class RetryableAPIError(Exception):
    """Raised when OpenRouter returns a retryable HTTP status code."""

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
async def call_openrouter(
    client: httpx.AsyncClient,
    model: str,
    messages: list[dict[str, str]],
    seed: int,
    api_key: str | None = None,
    max_retries: int = 5,  # kept for API compatibility; actual value controlled by decorator
) -> str:
    """
    Call the OpenRouter chat completions API and return the assistant's text.

    Args:
        client:     An httpx.AsyncClient instance (caller manages lifecycle).
        model:      Model identifier, e.g. "google/gemini-flash-1.5".
        messages:   List of chat message dicts with "role" and "content" keys.
        seed:       Integer seed passed to the model for reproducibility.
        api_key:    OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
        max_retries: Kept for signature compatibility (decorator uses fixed 5).

    Returns:
        The assistant message content string.

    Raises:
        RetryableAPIError: If a retryable status code is returned after all attempts.
    """
    resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "seed": seed,
    }

    response = await client.post(
        OPENROUTER_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=30.0,
    )

    if response.status_code in RETRYABLE_STATUS:
        raise RetryableAPIError(response.status_code)

    if response.status_code >= 400:
        response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


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
        ref_content = match.group(1)

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

def build_generation_prompt(domain: str, include_references: bool = True) -> str:
    """
    Build a German prompt instructing the LLM to generate regulatory text.

    Args:
        domain:             German regulatory domain abbreviation, e.g. "KWG".
        include_references: If True, ask for <ref>-tagged references.
                            If False, ask for text with NO references.

    Returns:
        A German prompt string ready to use as a user message.
    """
    if include_references:
        return (
            f"Schreiben Sie einen deutschen regulatorischen Absatz zum Thema {domain}. "
            f"Markieren Sie jeden Rechtsverweis (z.B. §, Art., Anhang) mit XML-Tags "
            f"im Format <ref>§ 25a {domain}</ref>. "
            f"Beispiel: 'Gemäß <ref>§ 25a {domain}</ref> sind Kreditinstitute verpflichtet...' "
            f"Der Text soll praxisnahe Verweise auf reale Normen des {domain} enthalten."
        )
    else:
        return (
            f"Schreiben Sie einen deutschen regulatorischen Absatz zum Thema {domain}. "
            f"Verwenden Sie KEINE Rechtsverweise (keine §, Art., Anhang-Verweise). "
            f"Der Text soll sachlich und fachlich korrekt sein, "
            f"aber ausschließlich erklärenden Charakter ohne Normzitate haben."
        )


# ---------------------------------------------------------------------------
# Domain rotation
# ---------------------------------------------------------------------------

def get_domain_for_seed(seed: int) -> str:
    """
    Select a regulatory domain deterministically based on seed.

    Args:
        seed: Integer seed value (e.g. from config data.llm_seed).

    Returns:
        A domain abbreviation string from DOMAIN_LIST.
    """
    return DOMAIN_LIST[seed % len(DOMAIN_LIST)]
