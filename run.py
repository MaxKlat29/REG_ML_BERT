"""
CLI entry point for the Regulatory NER pipeline.

Subcommands:
  train    — train RegulatoryNERModel (or ensemble) and save checkpoints
  evaluate — evaluate model vs regex baseline on gold test set
  predict  — run inference on a text string or file of texts

Usage:
  python run.py train [config overrides...]
  python run.py train --help
  python run.py evaluate --checkpoint checkpoints/run/epoch_0.pt
  python run.py predict --text "Gemaess § 25a KWG"
  python run.py predict --file inputs.txt
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

    # --- generate subcommand ---
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate training dataset via parallel LLM calls and export to JSON",
    )
    gen_parser.add_argument(
        "--config", "-c",
        default="config/default.yaml",
        help="Path to YAML config file (default: config/default.yaml)",
    )
    gen_parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON path (default: from config data.dataset_path)",
    )
    gen_parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Max parallel LLM requests (default: 50)",
    )
    gen_parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf-style config overrides (e.g. data.total_samples=5120)",
    )

    # --- evaluate subcommand ---
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model on gold test set and compare with regex baseline",
    )
    evaluate_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to .pt checkpoint file (defaults to most recent in checkpoints/)",
    )
    evaluate_parser.add_argument(
        "--gold-set",
        default=None,
        help="Path to gold JSON file (defaults to data/gold_test/gold_test_set.json)",
    )
    evaluate_parser.add_argument(
        "--config", "-c",
        default="config/default.yaml",
        help="Path to YAML config file",
    )
    evaluate_parser.add_argument(
        "--output-dir",
        default="eval_output",
        help="Directory for error dump output (default: eval_output)",
    )
    evaluate_parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf-style config overrides",
    )

    # --- predict subcommand ---
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run inference on input text and print detected reference spans",
    )
    predict_parser.add_argument(
        "--text",
        default=None,
        help="Single text string to run inference on",
    )
    predict_parser.add_argument(
        "--file",
        default=None,
        help="Path to a text file with one input per line (batch prediction)",
    )
    predict_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to .pt checkpoint file (defaults to most recent in checkpoints/)",
    )
    predict_parser.add_argument(
        "--config", "-c",
        default="config/default.yaml",
        help="Path to YAML config file",
    )
    predict_parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf-style config overrides",
    )

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.command == "train":
        _run_train(args)
    elif args.command == "generate":
        _run_generate(args)
    elif args.command == "evaluate":
        _run_evaluate(args)
    elif args.command == "predict":
        _run_predict(args)


def _run_generate(args) -> None:
    """Execute the dataset generation workflow.

    Args:
        args: Parsed argparse namespace with config, output, concurrency, overrides.
    """
    from pathlib import Path

    from transformers import BertTokenizerFast

    from src.utils.config import load_config
    from src.data.generate_dataset import run_generate

    config_path = getattr(args, "config", "config/default.yaml")
    overrides = getattr(args, "overrides", [])
    config = load_config(config_path=config_path, overrides=overrides)

    print(f"[generate] Config: {config_path}", flush=True)
    print(f"[generate] Total samples: {config.data.total_samples}", flush=True)
    print(f"[generate] Model: {config.data.llm_model}", flush=True)

    tokenizer = BertTokenizerFast.from_pretrained(config.model.name)

    output_path = Path(args.output) if args.output else Path(config.data.dataset_path or "data/training_dataset.json")
    concurrency = args.concurrency

    result_path = run_generate(config, tokenizer, output_path, concurrency=concurrency)
    print(f"\n[generate] Dataset ready: {result_path}", flush=True)


def _run_train(args) -> None:
    """Execute the training workflow.

    Args:
        args: Parsed argparse namespace with config, overrides attributes.
    """
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


def _run_evaluate(args) -> None:
    """Execute the evaluate workflow: ML model vs regex baseline comparison.

    Loads the specified (or latest) checkpoint, runs model inference on the
    gold test set, compares against the regex baseline, prints a formatted
    comparison table, and writes FP/FN error records to output_dir/errors.json.

    Args:
        args: Parsed argparse namespace with checkpoint, gold_set, config,
            output_dir, and overrides attributes.
    """
    from pathlib import Path

    from transformers import BertTokenizerFast

    from src.utils.config import load_config
    from src.utils.device import get_device, set_seed
    from src.model.ner_model import RegulatoryNERModel
    from src.model.trainer import load_checkpoint
    from src.model.predictor import Predictor
    from src.evaluation.evaluator import Evaluator

    config_path = getattr(args, "config", "config/default.yaml")
    overrides = getattr(args, "overrides", [])
    config = load_config(config_path=config_path, overrides=overrides)

    set_seed(config.project.seed)
    device = get_device()

    # Resolve checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = Predictor.find_latest_checkpoint()
    print(f"[evaluate] Checkpoint: {checkpoint_path}", flush=True)

    # Load model and tokenizer
    model = RegulatoryNERModel(config)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    model.to(device)

    tokenizer = BertTokenizerFast.from_pretrained(config.model.name)

    # Load gold set and evaluate
    evaluator = Evaluator(config)
    samples = evaluator.load_gold_set(path=args.gold_set)
    print(f"[evaluate] Gold samples loaded: {len(samples)}", flush=True)

    comparison = evaluator.evaluate_comparison(model, tokenizer, samples, device)

    # Print formatted comparison table
    report = evaluator.format_comparison_report(comparison)
    print(report)

    # Dump FP/FN errors
    output_dir = Path(getattr(args, "output_dir", "eval_output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    errors_path = output_dir / "errors.json"

    # Re-run model to collect per-sample predicted spans for error dump
    pred_spans_per_sample = []
    max_length = int(config.get("model", {}).get("max_length", 512))
    model.eval()
    import torch
    with torch.no_grad():
        for sample in samples:
            text = sample["text"]
            bio_labels_dict = sample.get("bio_labels", {})
            input_ids = torch.tensor(
                [bio_labels_dict["input_ids"]], dtype=torch.long
            ).to(device)
            attention_mask_t = torch.tensor(
                [bio_labels_dict["attention_mask"]], dtype=torch.long
            ).to(device)
            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            output = model(input_ids, attention_mask_t)
            use_crf = getattr(model, "_use_crf", False)
            if use_crf:
                pred_int_labels: list[int] = output[0]
            else:
                pred_int_labels = output.logits.argmax(dim=-1)[0].tolist()

            from src.evaluation.metrics import decode_bio_to_char_spans
            pred_spans = decode_bio_to_char_spans(
                pred_int_labels, list(enc["offset_mapping"])
            )
            pred_spans_per_sample.append(pred_spans)

    evaluator.dump_errors(samples, pred_spans_per_sample, errors_path)
    print(f"[evaluate] Error dump written to: {errors_path}", flush=True)


def _run_predict(args) -> None:
    """Execute the predict workflow: inference on text or file.

    Creates a Predictor from the specified (or latest) checkpoint and runs
    inference. With --text prints spans for a single input. With --file reads
    one text per line and runs batch prediction.

    Args:
        args: Parsed argparse namespace with text, file, checkpoint, config,
            and overrides attributes.
    """
    from pathlib import Path

    from src.utils.config import load_config
    from src.utils.device import get_device, set_seed
    from src.model.predictor import Predictor

    if not args.text and not args.file:
        print("Error: provide --text or --file", file=sys.stderr)
        sys.exit(1)

    config_path = getattr(args, "config", "config/default.yaml")
    overrides = getattr(args, "overrides", [])
    config = load_config(config_path=config_path, overrides=overrides)

    set_seed(config.project.seed)
    device = get_device()

    # Resolve checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = Predictor.find_latest_checkpoint()
    print(f"[predict] Checkpoint: {checkpoint_path}", flush=True)

    predictor = Predictor(checkpoint_path=checkpoint_path, config=config, device=device)

    if args.text:
        spans = predictor.predict(args.text)
        print(f"\nInput: {args.text!r}")
        if spans:
            print(f"Found {len(spans)} span(s):")
            for span in spans:
                print(f"  [{span.start}:{span.end}] {span.text!r} (confidence={span.confidence:.4f})")
        else:
            print("  (no references found)")

    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: file not found: {file_path}", file=sys.stderr)
            sys.exit(1)

        texts = [line.rstrip("\n") for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"[predict] Processing {len(texts)} lines from {file_path}", flush=True)

        results = predictor.predict_batch(texts)
        for idx, (text, spans) in enumerate(zip(texts, results)):
            print(f"\n[{idx}] {text!r}")
            if spans:
                for span in spans:
                    print(f"  [{span.start}:{span.end}] {span.text!r} (confidence={span.confidence:.4f})")
            else:
                print("  (no references found)")


if __name__ == "__main__":
    main()
