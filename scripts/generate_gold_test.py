"""Gold test set generator.

Generates a frozen, reviewable evaluation dataset of German regulatory text
with both positive (with references) and negative (no references) examples.

The gold set must be generated and reviewed BEFORE any model training begins.
It serves as ground truth for evaluating the ML model vs the regex baseline.

Usage (from project root):
    PYTHONPATH=. OPENROUTER_API_KEY=sk-... python scripts/generate_gold_test.py
    PYTHONPATH=. OPENROUTER_API_KEY=sk-... python scripts/generate_gold_test.py data.gold_test_dir=data/gold_test

Output:
    data/gold_test/gold_test_set.json  (or configured path)

Every sample in the output has needs_review: true so human reviewers know
to inspect each entry before using it for evaluation.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx

from src.data.bio_converter import char_spans_to_bio, get_tokenizer
from src.data.llm_client import (
    call_openrouter,
    get_domain_for_seed,
    parse_ref_tags,
)


# ---------------------------------------------------------------------------
# Prompt builder — higher quality than training prompts
# ---------------------------------------------------------------------------

def build_gold_prompt(domain: str, include_references: bool = True) -> str:
    """Build a high-quality German prompt for gold test set generation.

    The prompt requests realistic, diverse regulatory text. For positive
    examples it asks for precise <ref> tag placement around all legal
    citations. For negative examples it strictly excludes all references.

    Args:
        domain:             German regulatory domain abbreviation, e.g. "KWG".
        include_references: True for positive samples, False for negatives.

    Returns:
        A German prompt string suitable as a chat user message.
    """
    if include_references:
        return (
            f"Schreiben Sie einen realistischen deutschen Regulierungstext zum Thema {domain}. "
            f"Markieren Sie JEDEN Rechtsverweis (§, Art., Abs., Anhang, Nr., Tz.) präzise "
            f"mit XML-Tags: <ref>§ 25a {domain}</ref>. "
            f"Der Tag beginnt genau am ersten Zeichen des Verweises und endet nach dem letzten. "
            f"Verwenden Sie reale Normen des {domain}. "
            f"Der Absatz soll 2-4 Sätze umfassen und mindestens 2 Rechtsverweise enthalten. "
            f"Beispiel: 'Gemäß <ref>§ 25a Abs. 1 {domain}</ref> sind Institute verpflichtet, "
            f"nach <ref>§ 10 {domain}</ref> ausreichend Eigenkapital vorzuhalten.'"
        )
    else:
        return (
            f"Schreiben Sie einen sachlichen deutschen Regulierungstext zum Thema {domain}. "
            f"Verwenden Sie KEINE Rechtsverweise — KEINE §, Art., Abs., Anhang, Tz. oder "
            f"ähnlichen Normzitate. "
            f"Der Text soll fachlich korrekte Erklärungen zu regulatorischen Konzepten "
            f"enthalten, aber ausschließlich beschreibend ohne Normbezüge sein. "
            f"Der Absatz soll 2-4 Sätze umfassen."
        )


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

async def _generate_single_sample(
    client: httpx.AsyncClient,
    config,
    sample_index: int,
    num_samples: int,
    tokenizer,
) -> dict:
    """Generate one gold sample asynchronously.

    Args:
        client:       Shared httpx.AsyncClient.
        config:       OmegaConf config object.
        sample_index: Zero-based index of this sample (0 .. num_samples-1).
        num_samples:  Total number of samples being generated.
        tokenizer:    Fast HuggingFace tokenizer for BIO conversion.

    Returns:
        Sample dict with all required fields.
    """
    # Deterministic positive/negative split:
    # first (ratio * num_samples) samples are negative, the rest are positive.
    num_negative = int(num_samples * config.data.negative_sample_ratio)
    include_refs = sample_index >= num_negative

    sample_seed = config.data.llm_seed + sample_index
    domain = get_domain_for_seed(sample_seed)

    prompt_text = build_gold_prompt(domain, include_references=include_refs)
    messages = [{"role": "user", "content": prompt_text}]

    tagged_text = await call_openrouter(
        client=client,
        model=config.data.llm_model,
        messages=messages,
        seed=sample_seed,
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    )

    clean_text, spans = parse_ref_tags(tagged_text)
    bio_encoding = char_spans_to_bio(clean_text, spans, tokenizer)

    return {
        "text": clean_text,
        "spans": list(spans),
        "bio_labels": bio_encoding,
        "needs_review": True,
        "domain": domain,
        "seed": sample_seed,
        "has_references": include_refs,
    }


def generate_gold_set(
    config,
    num_samples: int = 50,
    output_path: Path | None = None,
) -> list[dict]:
    """Generate the gold test set and persist it as JSON.

    Uses a fixed seed from config for full reproducibility. The positive/negative
    split is determined by config.data.negative_sample_ratio. Every sample has
    needs_review: True.

    Args:
        config:       OmegaConf DictConfig with data.* keys.
        num_samples:  Number of gold samples to generate (default 50).
        output_path:  Where to write the JSON file. If None, derived from
                      config.data.gold_test_dir / "gold_test_set.json".

    Returns:
        List of sample dicts.
    """
    if output_path is None:
        output_path = Path(config.data.gold_test_dir) / "gold_test_set.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer(config.model.name if hasattr(config, "model") else "deepset/gbert-large")

    async def _run_all() -> list[dict]:
        async with httpx.AsyncClient() as client:
            tasks = [
                _generate_single_sample(client, config, i, num_samples, tokenizer)
                for i in range(num_samples)
            ]
            return await asyncio.gather(*tasks)

    samples = asyncio.run(_run_all())

    save_gold_set(samples, output_path)
    return samples


def save_gold_set(samples: list[dict], output_path: Path) -> None:
    """Persist gold set samples to a JSON file.

    Args:
        samples:     List of sample dicts to serialize.
        output_path: Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(samples, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Load config and generate the gold test set.

    Reads config from config/default.yaml (with optional CLI overrides),
    generates num_samples samples, and writes them to the configured path.
    Prints a summary after completion.
    """
    from src.utils.config import load_config

    config = load_config()
    output_path = Path(config.data.gold_test_dir) / "gold_test_set.json"

    print(f"Generating gold test set ({50} samples) ...")
    print(f"  Model     : {config.data.llm_model}")
    print(f"  Seed      : {config.data.llm_seed}")
    print(f"  Neg ratio : {config.data.negative_sample_ratio}")
    print(f"  Output    : {output_path}")

    samples = generate_gold_set(config, num_samples=50, output_path=output_path)

    num_positive = sum(1 for s in samples if s["has_references"])
    num_negative = sum(1 for s in samples if not s["has_references"])
    print(f"\nDone. {len(samples)} samples written to {output_path}")
    print(f"  Positive (with refs) : {num_positive}")
    print(f"  Negative (no refs)   : {num_negative}")
    print("  All samples have needs_review: true — please review before use.")


if __name__ == "__main__":
    main()
