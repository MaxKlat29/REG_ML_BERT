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
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        batch_idx = worker_id
        samples_per_worker = self.config.data.samples_per_batch

        for _ in range(samples_per_worker):
            seed = self.epoch * 10000 + batch_idx * 100 + worker_id
            sample = self._generate_sample(seed)
            if sample is not None:
                yield sample
            batch_idx += num_workers

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
