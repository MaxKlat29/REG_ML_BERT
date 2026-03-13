"""
CLI entry point for the Regulatory NER pipeline.

Subcommands:
  train    — train RegulatoryNERModel (or ensemble) and save checkpoints
  evaluate — (Phase 4) not yet implemented
  predict  — (Phase 4) not yet implemented

Usage:
  python run.py train [config overrides...]
  python run.py train --help
"""
from __future__ import annotations

import argparse
import sys

import logging

from dotenv import load_dotenv
load_dotenv()  # Load .env before any module reads OPENROUTER_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def main(argv: list[str] | None = None) -> None:
    """Main entry point.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).
    """
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Regulatory NER training and inference CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train subcommand ---
    train_parser = subparsers.add_parser(
        "train",
        help="Train RegulatoryNERModel and save checkpoints",
    )
    train_parser.add_argument(
        "--config", "-c",
        default="config/default.yaml",
        help="Path to YAML config file (default: config/default.yaml)",
    )
    train_parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf-style config overrides (e.g. training.num_epochs=5)",
    )

    # --- evaluate subcommand (stub) ---
    subparsers.add_parser(
        "evaluate",
        help="Evaluate model on gold test set (Phase 4)",
    )

    # --- predict subcommand (stub) ---
    subparsers.add_parser(
        "predict",
        help="Run inference on input text (Phase 4)",
    )

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.command == "train":
        _run_train(args)
    elif args.command in ("evaluate", "predict"):
        print(f"Not yet implemented — see Phase 4")
        sys.exit(0)


def _run_train(args) -> None:
    """Execute the training workflow."""
    import time

    from accelerate import Accelerator
    from transformers import BertTokenizerFast

    from src.utils.config import load_config
    from src.utils.device import get_device, set_seed
    from src.model.trainer import (
        Trainer,
        resolve_mixed_precision,
        train_ensemble,
    )
    from src.model.ner_model import RegulatoryNERModel

    total_start = time.time()

    # Load config with optional CLI overrides
    config_path = args.config if hasattr(args, "config") else "config/default.yaml"
    overrides = args.overrides if hasattr(args, "overrides") else []
    print(f"[init] Config: {config_path} | overrides: {overrides}", flush=True)
    config = load_config(config_path=config_path, overrides=overrides)
    print(f"[init] Config loaded: {config.training.num_epochs} epochs, {config.data.samples_per_batch} samples/epoch", flush=True)

    # Seed and device setup
    set_seed(config.project.seed)
    device = get_device()
    print(f"[init] Seed: {config.project.seed} | Device: {device}", flush=True)

    # Mixed precision
    mixed_precision = resolve_mixed_precision(config, device)
    print(f"[init] Mixed precision: {mixed_precision}", flush=True)

    # Accelerator
    print(f"[init] Creating Accelerator...", flush=True)
    t0 = time.time()
    accelerator = Accelerator(mixed_precision=mixed_precision)
    print(f"[init] Accelerator ready in {time.time()-t0:.1f}s | device={accelerator.device}", flush=True)

    # Tokenizer — BertTokenizerFast (not AutoTokenizer; see decision in STATE.md)
    print(f"[init] Loading tokenizer: {config.model.name}...", flush=True)
    t0 = time.time()
    tokenizer = BertTokenizerFast.from_pretrained(config.model.name)
    print(f"[init] Tokenizer loaded in {time.time()-t0:.1f}s | vocab={tokenizer.vocab_size}", flush=True)

    if config.ensemble.enabled:
        print(f"[init] Ensemble mode: {config.ensemble.n_estimators} models", flush=True)
        checkpoint_paths = train_ensemble(config, tokenizer, accelerator)
        for path in checkpoint_paths:
            print(f"Ensemble checkpoint: {path}", flush=True)
    else:
        print(f"[init] Single model training — CRF: {config.model.use_crf}, LoRA: {config.model.use_lora}", flush=True)
        print(f"[init] Loading model: {config.model.name}...", flush=True)
        t0 = time.time()
        model = RegulatoryNERModel(config)
        print(f"[init] Model loaded in {time.time()-t0:.1f}s", flush=True)

        print(f"[init] Setup complete in {time.time()-total_start:.1f}s — starting training...\n", flush=True)
        trainer = Trainer(config, model, tokenizer, accelerator)
        checkpoint_path = trainer.train()
        print(f"\nCheckpoint: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
