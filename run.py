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

    # Load config with optional CLI overrides
    cli_args = args.overrides if hasattr(args, "overrides") else []
    config = load_config(cli_args=cli_args)

    # Seed and device setup
    set_seed(config.project.seed)
    device = get_device()

    # Mixed precision
    mixed_precision = resolve_mixed_precision(config, device)

    # Accelerator
    accelerator = Accelerator(mixed_precision=mixed_precision)

    # Tokenizer — BertTokenizerFast (not AutoTokenizer; see decision in STATE.md)
    tokenizer = BertTokenizerFast.from_pretrained(config.model.name)

    if config.ensemble.enabled:
        checkpoint_paths = train_ensemble(config, tokenizer, accelerator)
        for path in checkpoint_paths:
            print(f"Ensemble checkpoint: {path}")
    else:
        model = RegulatoryNERModel(config)
        trainer = Trainer(config, model, tokenizer, accelerator)
        checkpoint_path = trainer.train()
        print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
