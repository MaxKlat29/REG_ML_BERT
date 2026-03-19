"""Parallel dataset generator for cross-reference NER training.

Generates all training samples upfront via async Ollama LLM calls, converts to
BIO-labeled encodings, and exports to JSON. Samples are saved incrementally
to a JSONL checkpoint file so that interrupted runs are recoverable.

Usage:
    python run.py generate --config config/gpu.yaml
    python run.py generate -c config/gpu.yaml -o data/training_dataset.json
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import signal
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
    checkpoint_path: Path | None = None,
) -> list[dict]:
    """Generate all training samples in parallel with incremental checkpointing.

    Samples are appended to a JSONL checkpoint file as they complete,
    so interrupted runs can be recovered.

    Args:
        config: OmegaConf config with data.llm_model, data.max_seq_length, etc.
        tokenizer: Fast HuggingFace tokenizer.
        total_samples: Total number of samples to generate.
        concurrency: Max parallel Ollama requests.
        checkpoint_path: Path for the JSONL checkpoint file.

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
    shutting_down = False

    # Resume from checkpoint if it exists
    existing_seeds: set[int] = set()
    if checkpoint_path and checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                results.append(sample)
                existing_seeds.add(sample.get("_meta", {}).get("seed", -1))
        generated = len(results)
        print(
            f"  [checkpoint] Resumed {generated} samples from {checkpoint_path}",
            flush=True,
        )

    # Figure out which seeds still need generating
    seeds_todo = [
        base_seed + i
        for i in range(total_samples)
        if (base_seed + i) not in existing_seeds
    ]
    remaining = len(seeds_todo)

    if remaining == 0:
        print("  [generate] All samples already in checkpoint, nothing to do.", flush=True)
        return results

    print(
        f"  [generate] Starting {remaining} Ollama calls "
        f"(concurrency={concurrency}, model={model}, endpoint={endpoint})",
        flush=True,
    )

    start = time.time()

    # Open checkpoint for appending
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_file = checkpoint_path.open("a", encoding="utf-8") if checkpoint_path else None

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            tasks = [
                _generate_one(
                    client, model, endpoint, seed,
                    tokenizer, max_seq_length, negative_ratio, semaphore,
                )
                for seed in seeds_todo
            ]

            for coro in asyncio.as_completed(tasks):
                if shutting_down:
                    break
                try:
                    result = await coro
                except asyncio.CancelledError:
                    break

                if result is not None:
                    results.append(result)
                    generated += 1
                    if ckpt_file:
                        ckpt_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                        ckpt_file.flush()
                else:
                    skipped += 1

                done = generated + skipped
                if done % 50 == 0 or done == generated + skipped:
                    total_todo = remaining + len(existing_seeds)
                    elapsed = time.time() - start
                    rate = (done - len(existing_seeds)) / elapsed if elapsed > 0 else 0
                    pct = done / total_todo * 100
                    bar_len = 40
                    filled = int(bar_len * done // total_todo)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(
                        f"\r  [{bar}] {done}/{total_todo} ({pct:.0f}%) | "
                        f"{generated} ok, {skipped} skipped | "
                        f"{rate:.1f} samples/s",
                        end="",
                        flush=True,
                    )
    except (KeyboardInterrupt, asyncio.CancelledError):
        print(f"\n  [generate] Interrupted — {generated} samples saved to checkpoint.", flush=True)
    finally:
        if ckpt_file:
            ckpt_file.close()

    elapsed = time.time() - start
    rate = (generated - len(existing_seeds)) / elapsed if elapsed > 0 else 0
    print(flush=True)
    print(
        f"  [generate] Done: {generated} samples in {elapsed:.1f}s "
        f"({rate:.1f} samples/s, {skipped} skipped)",
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
    checkpoint_path = Path(output_path).with_suffix(".checkpoint.jsonl")

    print(f"\n{'━'*60}", flush=True)
    print(f" Dataset Generation: {total} samples", flush=True)
    print(f" Checkpoint: {checkpoint_path}", flush=True)
    print(f"{'━'*60}", flush=True)

    samples = asyncio.run(
        generate_all(config, tokenizer, total, concurrency=concurrency,
                     checkpoint_path=checkpoint_path)
    )

    print(f"\n{'━'*60}", flush=True)
    print(f" Exporting to JSON", flush=True)
    print(f"{'━'*60}", flush=True)

    result = export_dataset_json(samples, output_path, config)

    # Clean up checkpoint after successful export
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  [checkpoint] Removed {checkpoint_path}", flush=True)

    return result


def merge_datasets(
    output_path: Path,
    source_paths: list[Path],
    subsample_per_source: int | None = None,
    per_source_limits: dict[str, int] | None = None,
    seed: int = 42,
) -> Path:
    """Merge and optionally subsample multiple dataset JSON files.

    Args:
        output_path: Where to write the merged JSON.
        source_paths: List of existing dataset JSON files.
        subsample_per_source: Default max samples per source (fallback).
        per_source_limits: Dict mapping source path string to per-file limit
            (overrides subsample_per_source for that file).
        seed: Random seed for reproducible subsampling.

    Returns:
        The output path.
    """
    rng = random.Random(seed)
    per_source_limits = per_source_limits or {}
    all_samples = []

    for src in source_paths:
        with src.open("r", encoding="utf-8") as f:
            data = json.load(f)
        samples = data["samples"]
        total = len(samples)
        model = data.get("metadata", {}).get("model", "unknown")

        # Per-source limit takes priority, then global default
        limit = per_source_limits.get(str(src), subsample_per_source)
        if limit and len(samples) > limit:
            samples = rng.sample(samples, limit)

        # Tag source
        for s in samples:
            s["_source"] = str(src.name)

        print(f"  [merge] {src.name}: {len(samples)}/{total} samples (model={model})", flush=True)
        all_samples.extend(samples)

    rng.shuffle(all_samples)

    # Re-index
    for idx, s in enumerate(all_samples):
        s["index"] = idx

    # Build merged metadata
    merged = {
        "metadata": {
            "total_samples": len(all_samples),
            "sources": [str(p) for p in source_paths],
            "subsample_per_source": subsample_per_source,
        },
        "samples": all_samples,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  [merge] {len(all_samples)} total samples → {output_path} ({size_mb:.1f} MB)", flush=True)
    return output_path
