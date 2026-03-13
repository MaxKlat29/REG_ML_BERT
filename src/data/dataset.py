"""LLMGeneratedDataset: PyTorch IterableDataset for on-the-fly training data.

Composes:
  - LLM generation (OpenRouter API via llm_client)
  - BIO label conversion (char spans -> token labels via offset_mapping)
  - JSONL disk cache (optional read-from-cache mode for ensemble resampling)

Worker sharding: Uses torch.utils.data.get_worker_info() to offset seeds
per worker so each worker generates distinct samples.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Iterator

from torch.utils.data import IterableDataset, get_worker_info

from src.data.bio_converter import char_spans_to_bio
from src.data.cache import append_to_cache, load_cache
from src.data.llm_client import (
    build_generation_prompt,
    call_openrouter,
    get_domain_for_seed,
    parse_ref_tags,
)

logger = logging.getLogger(__name__)


class LLMGeneratedDataset(IterableDataset):
    """Iterable dataset that generates training samples via LLM on-the-fly.

    In live mode (_iter_from_llm): calls OpenRouter for each sample, parses
    <ref>...</ref> tags, converts char spans to BIO labels.

    In cache mode (_iter_from_cache): reads pre-generated JSONL cache
    (no LLM calls). Useful for ensemble resampling or offline testing.

    Args:
        config: OmegaConf/SimpleNamespace config object with data sub-config.
        tokenizer: Fast HuggingFace tokenizer (offset_mapping support required).
        epoch: Current training epoch (used in seed computation for diversity).
        cache_path: Optional path to JSONL cache file. If set and file exists,
                    reads from cache instead of calling LLM.
    """

    def __init__(self, config, tokenizer, epoch: int = 0, cache_path=None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.cache_path = cache_path

    def __iter__(self) -> Iterator[dict]:
        if self.cache_path is not None and Path(self.cache_path).exists():
            yield from self._iter_from_cache()
        else:
            yield from self._iter_from_llm()

    def _iter_from_cache(self) -> Iterator[dict]:
        """Yield all samples from the JSONL cache file."""
        samples = load_cache(Path(self.cache_path))
        for sample in samples:
            yield sample

    def _iter_from_llm(self) -> Iterator[dict]:
        """Generate samples via LLM, apply BIO conversion, yield encodings."""
        import sys

        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        batch_idx = worker_id
        total = self.config.data.samples_per_batch

        generated = 0
        skipped = 0
        for i in range(total):
            seed = self.epoch * 10000 + batch_idx * 100 + worker_id
            domain = get_domain_for_seed(seed)
            # Progress bar
            pct = (i + 1) / total * 100
            bar_len = 30
            filled = int(bar_len * (i + 1) // total)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(
                f"\r  [{bar}] {i+1}/{total} ({pct:.0f}%) | "
                f"Epoch {self.epoch} | seed={seed} | domain={domain}",
                end="", flush=True,
            )
            sample = self._generate_sample(seed)
            if sample is not None:
                generated += 1
                # Show the generated text preview
                print(flush=True)  # newline after progress bar
                self._print_sample_preview(sample, seed, domain, generated)
                yield sample
            else:
                skipped += 1
            batch_idx += num_workers

        print(flush=True)  # final newline
        print(
            f"  ✓ Epoch {self.epoch} done: {generated} samples generated, "
            f"{skipped} skipped"
        )

    def _print_sample_preview(self, sample: dict, seed: int, domain: str, idx: int):
        """Print a compact live preview of the generated sample."""
        from src.data.bio_converter import LABEL_B_REF, LABEL_I_REF

        labels = sample["labels"]
        input_ids = sample["input_ids"]

        # Count ref tokens
        b_count = labels.count(LABEL_B_REF) if isinstance(labels, list) else sum(1 for l in labels if l == LABEL_B_REF)
        i_count = labels.count(LABEL_I_REF) if isinstance(labels, list) else sum(1 for l in labels if l == LABEL_I_REF)
        total_ref = b_count + i_count
        num_entities = b_count  # each B-REF starts an entity

        # Decode tokens to show a text snippet
        text_tokens = self.tokenizer.decode(
            [t for t, m in zip(input_ids, sample["attention_mask"]) if m == 1],
            skip_special_tokens=True,
        )
        preview = text_tokens[:120].replace("\n", " ")
        if len(text_tokens) > 120:
            preview += "..."

        # Color coding
        if num_entities > 0:
            tag = f"✓ {num_entities} ref(s), {total_ref} tokens"
        else:
            tag = "○ negative sample (no refs)"

        print(f"    #{idx} [{domain}] {tag}")
        print(f"       \033[2m{preview}\033[0m")

    def _generate_sample(self, seed: int) -> dict | None:
        """Generate a single training sample for the given seed.

        Returns:
            Encoding dict (input_ids, attention_mask, labels) or None on error.
        """
        try:
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            domain = get_domain_for_seed(seed)
            prompt = build_generation_prompt(domain)
            model = self.config.data.llm_model

            # Async -> sync: run the coroutine in a new event loop
            messages = [{"role": "user", "content": prompt}]

            import httpx

            async def _run():
                async with httpx.AsyncClient() as client:
                    return await call_openrouter(
                        client, model, messages, seed, api_key
                    )

            tagged_text = asyncio.run(_run())
            text, spans = parse_ref_tags(tagged_text)
            encoding = char_spans_to_bio(
                text, spans, self.tokenizer,
                max_length=self.config.data.max_seq_length,
            )
            return encoding

        except Exception as exc:
            logger.warning("Skipping sample (seed=%d): %s", seed, exc)
            return None
