"""
Unit tests for src.data.llm_client.
All HTTP calls are mocked — no live API calls are made.
"""
import pytest
import httpx
from unittest.mock import AsyncMock, patch
from tenacity import wait_none

import src.data.llm_client as llm_mod
from src.data.llm_client import (
    call_ollama,
    parse_ref_tags,
    build_generation_prompt,
    get_domain_for_seed,
    get_context_for_seed,
    DOCUMENT_CONTEXTS,
    DOMAIN_LIST,
    RetryableAPIError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status_code: int, content: str = "generated text") -> httpx.Response:
    """Build a fake httpx.Response for mocking."""
    if status_code == 200:
        json_body = {"message": {"role": "assistant", "content": content}}
        return httpx.Response(status_code, json=json_body)
    return httpx.Response(status_code)


def _no_wait(retry_state):
    """Wait function that returns 0 — used to speed up retry tests."""
    return 0


# ---------------------------------------------------------------------------
# call_ollama tests
# ---------------------------------------------------------------------------

async def test_call_ollama_success():
    """Mocked httpx returns 200; call_ollama returns content string."""
    mock_response = _make_response(200, "Gemäß § 25a KWG gilt folgendes.")
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    result = await call_ollama(
        client=mock_client,
        model="qwen2.5:14b",
        messages=[{"role": "user", "content": "Generate text"}],
        seed=42,
    )
    assert result == "Gemäß § 25a KWG gilt folgendes."
    mock_client.post.assert_called_once()


async def test_call_ollama_retry_on_429():
    """Mocked httpx returns 429 twice then 200; succeeds after retries."""
    responses = [
        _make_response(429),
        _make_response(429),
        _make_response(200, "success after retries"),
    ]
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=responses)

    call_ollama.retry.wait = wait_none()
    try:
        result = await call_ollama(
            client=mock_client,
            model="qwen2.5:14b",
            messages=[{"role": "user", "content": "test"}],
            seed=1,
        )
    finally:
        from tenacity import wait_exponential_jitter
        call_ollama.retry.wait = wait_exponential_jitter(initial=1, max=60, jitter=5)

    assert result == "success after retries"
    assert mock_client.post.call_count == 3


async def test_call_ollama_retry_on_502():
    """Mocked httpx returns 502 then 200; succeeds after retry."""
    responses = [
        _make_response(502),
        _make_response(200, "recovered"),
    ]
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=responses)

    call_ollama.retry.wait = wait_none()
    try:
        result = await call_ollama(
            client=mock_client,
            model="qwen2.5:14b",
            messages=[{"role": "user", "content": "test"}],
            seed=2,
        )
    finally:
        from tenacity import wait_exponential_jitter
        call_ollama.retry.wait = wait_exponential_jitter(initial=1, max=60, jitter=5)

    assert result == "recovered"
    assert mock_client.post.call_count == 2


async def test_call_ollama_gives_up_after_max_retries():
    """Mocked httpx always returns 429; raises RetryableAPIError after max attempts."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_make_response(429))

    call_ollama.retry.wait = wait_none()
    try:
        with pytest.raises(Exception):
            await call_ollama(
                client=mock_client,
                model="qwen2.5:14b",
                messages=[{"role": "user", "content": "test"}],
                seed=3,
            )
    finally:
        from tenacity import wait_exponential_jitter
        call_ollama.retry.wait = wait_exponential_jitter(initial=1, max=60, jitter=5)

    assert mock_client.post.call_count >= 5


# ---------------------------------------------------------------------------
# parse_ref_tags tests
# ---------------------------------------------------------------------------

def test_parse_ref_tags_single():
    """Single ref tag: produces clean text + one span."""
    tagged = "Gemäß <ref>SS 25a KWG</ref> gilt."
    clean_text, spans = parse_ref_tags(tagged)
    assert "<ref>" not in clean_text
    assert "</ref>" not in clean_text
    assert len(spans) == 1
    start, end = spans[0]
    assert clean_text[start:end] == "SS 25a KWG"


def test_parse_ref_tags_multiple():
    """Three ref tags produce 3 spans with correct character offsets."""
    tagged = (
        "Siehe <ref>§ 1 KWG</ref> sowie <ref>Art. 6 DSGVO</ref>"
        " und <ref>§ 25a Abs. 1 KWG</ref>."
    )
    clean_text, spans = parse_ref_tags(tagged)
    assert len(spans) == 3
    expected_contents = ["§ 1 KWG", "Art. 6 DSGVO", "§ 25a Abs. 1 KWG"]
    for (start, end), expected in zip(spans, expected_contents):
        assert clean_text[start:end] == expected


def test_parse_ref_tags_no_refs():
    """No ref tags: returns original text and empty span list."""
    text = "Ein normaler Satz ohne Verweise."
    clean_text, spans = parse_ref_tags(text)
    assert clean_text == text
    assert spans == []


def test_parse_ref_tags_validates_spans():
    """Each extracted span text matches clean_text[start:end]."""
    tagged = (
        "<ref>BGB § 307</ref> ist relevant für <ref>HGB § 238 Abs. 1</ref>."
    )
    clean_text, spans = parse_ref_tags(tagged)
    assert len(spans) == 2
    for start, end in spans:
        assert end > start
        span_text = clean_text[start:end]
        assert len(span_text) > 0
        assert "<ref>" not in span_text
        assert "</ref>" not in span_text


def test_parse_ref_tags_contract_references():
    """Contract-style references (Ziffer, Anlage, Abschnitt) are parsed correctly."""
    tagged = (
        "Gemäß <ref>Ziffer 5.1</ref> und <ref>Anlage 2 (Leistungsbeschreibung)</ref> "
        "sowie <ref>Abschnitt 8 (Haftung)</ref>."
    )
    clean_text, spans = parse_ref_tags(tagged)
    assert len(spans) == 3
    expected = ["Ziffer 5.1", "Anlage 2 (Leistungsbeschreibung)", "Abschnitt 8 (Haftung)"]
    for (start, end), exp in zip(spans, expected):
        assert clean_text[start:end] == exp


# ---------------------------------------------------------------------------
# build_generation_prompt tests
# ---------------------------------------------------------------------------

def test_build_prompt_with_references():
    """With include_references=True, prompt contains doc type and <ref> example."""
    prompt = build_generation_prompt("Dienstleistungsvertrag", "IT-Outsourcing", include_references=True)
    assert "Dienstleistungsvertrag" in prompt
    assert "<ref>" in prompt
    assert "Querverweis" in prompt


def test_build_prompt_negative():
    """With include_references=False, prompt contains doc type and no-reference instruction."""
    prompt = build_generation_prompt("Kaufvertrag", "Asset Deal", include_references=False)
    assert "Kaufvertrag" in prompt
    assert "KEINE" in prompt


def test_build_prompt_covers_reference_types():
    """Positive prompt mentions diverse reference types (§, Ziffer, Anlage, ISO, etc.)."""
    prompt = build_generation_prompt("Rahmenvertrag", "IT-Rahmenvertrag", include_references=True)
    for keyword in ["§", "Ziffer", "Anlage", "Abschnitt", "ISO", "Anhang"]:
        assert keyword in prompt, f"Prompt missing reference type: {keyword}"


# ---------------------------------------------------------------------------
# Context & seed rotation tests
# ---------------------------------------------------------------------------

def test_seed_determinism():
    """Same seed produces same context."""
    ctx1 = get_context_for_seed(42)
    ctx2 = get_context_for_seed(42)
    assert ctx1 == ctx2


def test_context_rotation():
    """Different seed values select different contexts; selection is deterministic."""
    assert len(DOCUMENT_CONTEXTS) >= 40, "Should have diverse document contexts"
    n = len(DOCUMENT_CONTEXTS)
    selected = [get_context_for_seed(i) for i in range(n)]
    assert len(set(selected)) == n, "All contexts should be unique for seeds 0..n-1"


def test_domain_list_backwards_compat():
    """DOMAIN_LIST backwards compatibility alias exists and has entries."""
    assert len(DOMAIN_LIST) == len(DOCUMENT_CONTEXTS)


def test_get_domain_for_seed_returns_string():
    """get_domain_for_seed returns a non-empty string."""
    result = get_domain_for_seed(0)
    assert isinstance(result, str)
    assert len(result) > 0
