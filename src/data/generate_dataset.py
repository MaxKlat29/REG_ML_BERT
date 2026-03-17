"""Parallel dataset generator for cross-reference NER training.

Generates all training samples upfront via async Ollama LLM calls, converts to
BIO-labeled encodings, and exports to JSON. Designed to run as a
preprocessing step before training on the GPU workstation.

Usage:
    python run.py generate --config config/gpu.yaml
    python run.py generate -c config/gpu.yaml -o data/training_dataset.json
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import httpx

from src.data.bio_converter import (
    LABEL_B_REF,
    LABEL_I_REF,
    LABEL_O,
    LABEL_IGNORE,
    char_spans_to_bio,
)
from src.data.llm_client import (
    build_generation_prompt,
    call_ollama,
    get_context_for_seed,
    get_domain_for_seed,
    parse_ref_tags,
)

logger = logging.getLogger(__name__)

# Ollama runs locally — keep concurrency modest to avoid OOM on GPU
DEFAULT_CONCURRENCY = 4


async def _generate_one(
    client: httpx.AsyncClient,
    model: str,
    endpoint: str,
    seed: int,
    tokenizer,
    max_seq_length: int,
    negative_ratio: float,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Generate a single sample via Ollama and convert to BIO encoding."""
    async with semaphore:
        try:
            doc_type, scenario = get_context_for_seed(seed)
            include_refs = (seed % 100) >= (negative_ratio * 100)
            prompt = build_generation_prompt(doc_type, scenario, include_references=include_refs)
            messages = [{"role": "user", "content": prompt}]

            tagged_text = await call_ollama(
                client, model, messages, seed, endpoint=endpoint
            )

            text, spans = parse_ref_tags(tagged_text)
            encoding = char_spans_to_bio(
                text, spans, tokenizer, max_length=max_seq_length
            )
            encoding["_meta"] = {
                "seed": seed,
                "domain": f"{doc_type}: {scenario}",
                "has_refs": include_refs,
                "text": text,
                "spans": spans,
            }
            return encoding
        except Exception as exc:
            logger.warning("Skipping sample (seed=%d): %s", seed, exc)
            return None


async def generate_all(
    config,
    tokenizer,
    total_samples: int,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[dict]:
    """Generate all training samples in parallel.

    Args:
        config: OmegaConf config with data.llm_model, data.max_seq_length, etc.
        tokenizer: Fast HuggingFace tokenizer.
        total_samples: Total number of samples to generate.
        concurrency: Max parallel Ollama requests.

    Returns:
        List of encoding dicts with _meta field.
    """
    model = config.data.llm_model
    endpoint = getattr(config.data, "ollama_endpoint", "") or os.environ.get("OLLAMA_ENDPOINT", "") or "http://localhost:11434"
    max_seq_length = config.data.max_seq_length
    negative_ratio = getattr(config.data, "negative_sample_ratio", 0.4)
    base_seed = getattr(config.data, "llm_seed", 1337)

    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    generated = 0
    skipped = 0

    print(
        f"  [generate] Starting {total_samples} Ollama calls "
        f"(concurrency={concurrency}, model={model}, endpoint={endpoint})",
        flush=True,
    )

    start = time.time()

    async with httpx.AsyncClient(timeout=180.0) as client:
        tasks = [
            _generate_one(
                client,
                model,
                endpoint,
                base_seed + i,
                tokenizer,
                max_seq_length,
                negative_ratio,
                semaphore,
            )
            for i in range(total_samples)
        ]

        # Process with progress reporting
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            if result is not None:
                results.append(result)
                generated += 1
            else:
                skipped += 1

            done = generated + skipped
            if done % 50 == 0 or done == total_samples:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                pct = done / total_samples * 100
                bar_len = 40
                filled = int(bar_len * done // total_samples)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(
                    f"\r  [{bar}] {done}/{total_samples} ({pct:.0f}%) | "
                    f"{generated} ok, {skipped} skipped | "
                    f"{rate:.1f} samples/s",
                    end="",
                    flush=True,
                )

    elapsed = time.time() - start
    print(flush=True)
    print(
        f"  [generate] Done: {generated} samples in {elapsed:.1f}s "
        f"({generated/elapsed:.1f} samples/s, {skipped} skipped)",
        flush=True,
    )
    return results


def export_dataset_json(
    samples: list[dict],
    output_path: Path,
    config,
) -> Path:
    """Export generated samples to a clean JSON file.

    Args:
        samples: List of encoding dicts (with _meta field).
        output_path: Where to write the JSON.
        config: Config for metadata.

    Returns:
        The output path.
    """
    label_map = {
        LABEL_O: "O",
        LABEL_B_REF: "B-REF",
        LABEL_I_REF: "I-REF",
        LABEL_IGNORE: "IGNORE",
    }

    export = {
        "metadata": {
            "total_samples": len(samples),
            "max_seq_length": config.data.max_seq_length,
            "model": config.data.llm_model,
            "label_schema": {"O": 0, "B-REF": 1, "I-REF": 2, "IGNORE": -100},
        },
        "samples": [],
    }

    for idx, sample in enumerate(samples):
        meta = sample.get("_meta", {})
        labels_numeric = sample["labels"]
        labels_str = [label_map.get(l, str(l)) for l in labels_numeric]

        export["samples"].append({
            "index": idx,
            "text": meta.get("text", ""),
            "domain": meta.get("domain", ""),
            "has_refs": meta.get("has_refs", True),
            "spans": meta.get("spans", []),
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "labels_numeric": labels_numeric,
            "labels": labels_str,
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  [export] {len(samples)} samples → {output_path} ({size_mb:.1f} MB)", flush=True)
    return output_path


def run_generate(config, tokenizer, output_path: Path, concurrency: int = DEFAULT_CONCURRENCY) -> Path:
    """Main entry point: generate dataset and export to JSON.

    Args:
        config: Full config.
        tokenizer: Fast tokenizer.
        output_path: Output JSON path.
        concurrency: Max parallel requests.

    Returns:
        Path to the exported JSON.
    """
    total = config.data.total_samples
    print(f"\n{'━'*60}", flush=True)
    print(f" Dataset Generation: {total} samples", flush=True)
    print(f"{'━'*60}", flush=True)

    samples = asyncio.run(
        generate_all(config, tokenizer, total, concurrency=concurrency)
    )

    print(f"\n{'━'*60}", flush=True)
    print(f" Exporting to JSON", flush=True)
    print(f"{'━'*60}", flush=True)

    return export_dataset_json(samples, output_path, config)
