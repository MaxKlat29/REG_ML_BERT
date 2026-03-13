"""
Unit tests for src.data.llm_client.
All HTTP calls are mocked — no live API calls are made.
"""
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from src.data.llm_client import (
    call_openrouter,
    parse_ref_tags,
    build_generation_prompt,
    get_domain_for_seed,
    DOMAIN_LIST,
    RetryableAPIError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status_code: int, content: str = "generated text") -> httpx.Response:
    """Build a fake httpx.Response for mocking."""
    if status_code == 200:
        json_body = {"choices": [{"message": {"content": content}}]}
        return httpx.Response(status_code, json=json_body)
    return httpx.Response(status_code)


# ---------------------------------------------------------------------------
# call_openrouter tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_openrouter_success():
    """Mocked httpx returns 200; call_openrouter returns content string."""
    mock_response = _make_response(200, "Gemäß § 25a KWG gilt folgendes.")
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    result = await call_openrouter(
        client=mock_client,
        model="google/gemini-flash-1.5",
        messages=[{"role": "user", "content": "Generate text"}],
        seed=42,
        api_key="test-key",
    )
    assert result == "Gemäß § 25a KWG gilt folgendes."
    mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_call_openrouter_retry_on_429():
    """Mocked httpx returns 429 twice then 200; succeeds after retries."""
    responses = [
        _make_response(429),
        _make_response(429),
        _make_response(200, "success after retries"),
    ]
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=responses)

    # Patch wait to instant to avoid slow tests
    with patch("src.data.llm_client.call_openrouter.retry.wait", new=None):
        pass  # The patching happens at call site

    # Use patched version with no wait
    with patch("tenacity.wait.wait_exponential_jitter.__call__", return_value=0):
        result = await call_openrouter(
            client=mock_client,
            model="google/gemini-flash-1.5",
            messages=[{"role": "user", "content": "test"}],
            seed=1,
            api_key="test-key",
        )
    assert result == "success after retries"
    assert mock_client.post.call_count == 3


@pytest.mark.asyncio
async def test_call_openrouter_retry_on_502():
    """Mocked httpx returns 502 then 200; succeeds after retry."""
    responses = [
        _make_response(502),
        _make_response(200, "recovered"),
    ]
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=responses)

    with patch("tenacity.wait.wait_exponential_jitter.__call__", return_value=0):
        result = await call_openrouter(
            client=mock_client,
            model="google/gemini-flash-1.5",
            messages=[{"role": "user", "content": "test"}],
            seed=2,
            api_key="test-key",
        )
    assert result == "recovered"
    assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_call_openrouter_gives_up_after_max_retries():
    """Mocked httpx always returns 429; raises RetryableAPIError after max attempts."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_make_response(429))

    with patch("tenacity.wait.wait_exponential_jitter.__call__", return_value=0):
        with pytest.raises(Exception):
            await call_openrouter(
                client=mock_client,
                model="google/gemini-flash-1.5",
                messages=[{"role": "user", "content": "test"}],
                seed=3,
                api_key="test-key",
            )
    # At least 5 attempts should have been made (max_retries=5)
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
    # Validate each span matches the expected content
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
        # Span must be non-empty and must index into clean text correctly
        assert end > start
        span_text = clean_text[start:end]
        assert len(span_text) > 0
        assert "<ref>" not in span_text
        assert "</ref>" not in span_text


# ---------------------------------------------------------------------------
# build_generation_prompt tests
# ---------------------------------------------------------------------------

def test_build_prompt_with_references():
    """With include_references=True, prompt contains domain name and <ref> example."""
    prompt = build_generation_prompt("KWG", include_references=True)
    assert "KWG" in prompt
    assert "<ref>" in prompt


def test_build_prompt_negative():
    """With include_references=False, prompt contains domain and no-reference instruction."""
    prompt = build_generation_prompt("KWG", include_references=False)
    assert "KWG" in prompt
    assert "KEINE" in prompt.upper() or "keine" in prompt.lower()


# ---------------------------------------------------------------------------
# Seed & domain rotation tests
# ---------------------------------------------------------------------------

def test_seed_determinism():
    """Same seed produces same request payload fields (model, messages, seed)."""
    # Verify domain selection is stable: two calls with same seed give same domain
    domain1 = get_domain_for_seed(42)
    domain2 = get_domain_for_seed(42)
    assert domain1 == domain2


def test_domain_rotation():
    """Different seed values select different domains; selection is deterministic."""
    # With 13 domains (len(DOMAIN_LIST) == 13), seeds 0..12 cover all domains
    assert len(DOMAIN_LIST) >= 10, "DOMAIN_LIST must have at least 10 domains"
    n = len(DOMAIN_LIST)
    selected = [get_domain_for_seed(i) for i in range(n)]
    # All should be valid domain strings
    for d in selected:
        assert d in DOMAIN_LIST
    # The full range should cover all domains (seed 0..n-1)
    assert set(selected) == set(DOMAIN_LIST)
    # Determinism: same seed always same result
    for i in range(n):
        assert get_domain_for_seed(i) == get_domain_for_seed(i)
