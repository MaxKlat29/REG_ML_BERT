"""Gold test set generator.

Generates a frozen, reviewable evaluation dataset of German documents
with both positive (with cross-references) and negative (no references) examples.

The gold set must be generated and reviewed BEFORE any model training begins.
It serves as ground truth for evaluating the ML model vs the regex baseline.

Usage (from project root, on GPU workstation):
    PYTHONPATH=. python scripts/generate_gold_test.py
    PYTHONPATH=. python scripts/generate_gold_test.py data.gold_test_dir=data/gold_test

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
    call_ollama,
    get_context_for_seed,
    parse_ref_tags,
)


# ---------------------------------------------------------------------------
# Prompt builder — higher quality than training prompts
# ---------------------------------------------------------------------------

def build_gold_prompt(doc_type: str, scenario: str, include_references: bool = True) -> str:
    """Build a high-quality German prompt for gold test set generation.

    Args:
        doc_type:           Document type, e.g. "Dienstleistungsvertrag".
        scenario:           Scenario hint, e.g. "IT-Outsourcing mit SLA-Anhängen".
        include_references: True for positive samples, False for negatives.

    Returns:
        A German prompt string suitable as a chat user message.
    """
    if include_references:
        return (
            f"Schreiben Sie einen realistischen deutschen Textauszug aus einem "
            f"'{doc_type}' zum Thema: {scenario}.\n\n"
            f"Markieren Sie JEDEN Querverweis präzise mit XML-Tags: <ref>...</ref>.\n"
            f"Der Tag beginnt genau am ersten Zeichen des Verweises und endet nach dem letzten.\n\n"
            f"Querverweise umfassen: Gesetzesverweise (§, Art.), Vertragsklauseln (Ziffer, Punkt), "
            f"Anhänge/Anlagen, Abschnitte/Kapitel, SLA-Verweise, Normen (ISO, DIN).\n\n"
            f"Der Absatz soll 2-4 Sätze umfassen und mindestens 3 verschiedene Querverweise enthalten.\n\n"
            f"Beispiel: 'Gemäß <ref>Ziffer 5.1</ref> dieses Vertrages sind die in "
            f"<ref>Anlage 2</ref> definierten Leistungen nach <ref>§ 280 Abs. 1 BGB</ref> "
            f"geschuldet.'"
        )
    else:
        return (
            f"Schreiben Sie einen sachlichen deutschen Textauszug aus einem "
            f"'{doc_type}' zum Thema: {scenario}.\n\n"
            f"Verwenden Sie KEINE Querverweise jeglicher Art — KEINE §, Art., Abs., "
            f"Anhang, Anlage, Ziffer, Punkt, Abschnitt, Kapitel, 'siehe', 'vgl.', "
            f"ISO, DIN oder ähnliche Verweise.\n\n"
            f"Der Text soll fachlich korrekte Erklärungen enthalten, aber ausschließlich "
            f"beschreibend ohne Verweise auf andere Textstellen sein.\n"
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
    """Generate one gold sample asynchronously."""
    num_negative = int(num_samples * config.data.negative_sample_ratio)
    include_refs = sample_index >= num_negative

    sample_seed = config.data.llm_seed + sample_index
    doc_type, scenario = get_context_for_seed(sample_seed)

    prompt_text = build_gold_prompt(doc_type, scenario, include_references=include_refs)
    messages = [{"role": "user", "content": prompt_text}]

    endpoint = getattr(config.data, "ollama_endpoint", "") or os.environ.get("OLLAMA_ENDPOINT", "") or "http://localhost:11434"

    tagged_text = await call_ollama(
        client=client,
        model=config.data.llm_model,
        messages=messages,
        seed=sample_seed,
        endpoint=endpoint,
    )

    clean_text, spans = parse_ref_tags(tagged_text)
    bio_encoding = char_spans_to_bio(clean_text, spans, tokenizer)

    return {
        "text": clean_text,
        "spans": list(spans),
        "bio_labels": bio_encoding,
        "needs_review": True,
        "domain": f"{doc_type}: {scenario}",
        "seed": sample_seed,
        "has_references": include_refs,
    }


def generate_gold_set(
    config,
    num_samples: int = 50,
    output_path: Path | None = None,
) -> list[dict]:
    """Generate the gold test set and persist it as JSON."""
    if output_path is None:
        output_path = Path(config.data.gold_test_dir) / "gold_test_set.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer(config.model.name if hasattr(config, "model") else "deepset/gbert-large")

    async def _run_all() -> list[dict]:
        async with httpx.AsyncClient(timeout=180.0) as client:
            tasks = [
                _generate_single_sample(client, config, i, num_samples, tokenizer)
                for i in range(num_samples)
            ]
            return await asyncio.gather(*tasks)

    samples = asyncio.run(_run_all())

    save_gold_set(samples, output_path)
    return samples


def save_gold_set(samples: list[dict], output_path: Path) -> None:
    """Persist gold set samples to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(samples, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Load config and generate the gold test set."""
    from src.utils.config import load_config

    config = load_config()
    output_path = Path(config.data.gold_test_dir) / "gold_test_set.json"

    print(f"Generating gold test set ({50} samples) ...")
    print(f"  Model     : {config.data.llm_model}")
    print(f"  Endpoint  : {getattr(config.data, 'ollama_endpoint', 'http://localhost:11434')}")
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
