"""
Trainer for RegulatoryNERModel.

Provides:
  - resolve_mixed_precision: selects correct Accelerate precision per device
  - build_optimizer: AdamW with differential LR (backbone vs head)
  - build_scheduler: linear warmup + linear decay via transformers scheduler
  - save_checkpoint / load_checkpoint: .pt checkpoint I/O
  - Trainer: main training loop (one dataset per epoch, accelerate-wrapped)
  - majority_vote: per-position mode over N model prediction lists
  - train_ensemble: bagging driver; model 0 writes cache, rest read from it
"""
from __future__ import annotations

import collections
import copy
import logging
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data.dataset import LLMGeneratedDataset
from src.model.ner_model import RegulatoryNERModel

logger = logging.getLogger(__name__)

# Patchable constant for tests to override the checkpoint base directory.
CHECKPOINT_BASE = Path("checkpoints")


# ---------------------------------------------------------------------------
# Mixed precision resolution
# ---------------------------------------------------------------------------

def resolve_mixed_precision(config, device) -> str:
    """Return the Accelerate mixed_precision string for the given device.

    CUDA — honour config.training.mixed_precision (user-set bf16/fp16/no).
    MPS  — bf16 requires torch >= 2.6; fall back to "no" on older builds.
    CPU  — always "no" (no hardware support).

    Args:
        config: OmegaConf / SimpleNamespace config with training.mixed_precision.
        device: torch.device (or mock with .type attribute).

    Returns:
        One of "fp16", "bf16", or "no".
    """
    device_type: str = device.type

    if device_type == "cuda":
        return config.training.mixed_precision

    if device_type == "mps":
        import torch as _torch  # import here so patch("torch.__version__") works
        version_str = _torch.__version__
        # Parse major.minor from e.g. "2.6.0" or "2.6.0+cu118"
        try:
            parts = version_str.split("+")[0].split(".")
            major, minor = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            return "no"

        if (major, minor) >= (2, 6):
            # bf16 supported on MPS from torch 2.6
            return "bf16"
        return "no"

    # CPU (and anything else)
    return "no"


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(model: RegulatoryNERModel, config) -> AdamW:
    """Build AdamW with two parameter groups for differential LR.

    Group 0 (backbone): model.get_bert_parameters(), lr=learning_rate_backbone
    Group 1 (head):     model.get_head_parameters(),  lr=learning_rate_head

    Only parameters with requires_grad=True are included in each group.

    Args:
        model: RegulatoryNERModel instance.
        config: Config with training.learning_rate_backbone, training.learning_rate_head.

    Returns:
        AdamW optimizer with two param groups.
    """
    backbone_params = [p for p in model.get_bert_parameters() if p.requires_grad]
    head_params = [p for p in model.get_head_parameters() if p.requires_grad]

    param_groups = [
        {"params": backbone_params, "lr": config.training.learning_rate_backbone},
        {"params": head_params, "lr": config.training.learning_rate_head},
    ]

    return AdamW(param_groups, weight_decay=0.01)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def build_scheduler(optimizer: AdamW, config, steps_per_epoch: int):
    """Build a linear warmup + linear decay LambdaLR scheduler.

    Args:
        optimizer: The AdamW optimizer (must already have initial LR set).
        config: Config with training.num_epochs and training.warmup_steps.
        steps_per_epoch: Number of gradient steps per epoch (approximate for
            IterableDataset).

    Returns:
        LambdaLR scheduler from transformers.
    """
    num_training_steps = config.training.num_epochs * steps_per_epoch
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=num_training_steps,
    )


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    config,
    accelerator,
    run_id: str = "",
) -> Path:
    """Save model/optimizer/scheduler state to a .pt file.

    Path: <CHECKPOINT_BASE>/<run_id>/epoch_<epoch>.pt

    Args:
        model: Model (will be unwrapped via accelerator.unwrap_model).
        optimizer: AdamW optimizer.
        scheduler: LambdaLR scheduler.
        epoch: Current epoch index (0-based is fine; saved as-is).
        config: Config (not saved, available for future use).
        accelerator: Accelerate Accelerator instance.
        run_id: Optional run identifier string for directory naming.

    Returns:
        Path to the saved checkpoint file.
    """
    unwrapped = accelerator.unwrap_model(model)
    ckpt_dir = CHECKPOINT_BASE / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": unwrapped.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        ckpt_path,
    )
    logger.info("Checkpoint saved: %s", ckpt_path)
    return ckpt_path


def load_checkpoint(
    path: Path | str,
    model,
    optimizer=None,
    scheduler=None,
) -> int:
    """Load model (and optionally optimizer/scheduler) state from a checkpoint.

    Args:
        path: Path to the .pt checkpoint file.
        model: Model to restore weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.

    Returns:
        Epoch number stored in the checkpoint.
    """
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class Trainer:
    """Training loop for RegulatoryNERModel.

    Wraps model, optimizer, scheduler, and dataloader via Accelerate for
    device-agnostic mixed-precision training with gradient clipping.

    Each epoch recreates LLMGeneratedDataset to get fresh seeds (avoids
    stale IterableDataset after exhaustion — Pitfall 5 from research).

    Args:
        config: Full OmegaConf / SimpleNamespace config.
        model: RegulatoryNERModel instance.
        tokenizer: BertTokenizerFast (or mock in tests).
        accelerator: accelerate.Accelerator instance.
        run_id: Optional string identifier for checkpoint subdirectory.
        cache_path: Optional path to JSONL cache; passed to LLMGeneratedDataset.
    """

    def __init__(self, config, model, tokenizer, accelerator, run_id: str = "", cache_path=None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.run_id = run_id
        self.cache_path = cache_path

    def train(self) -> Path:
        """Run the full training loop.

        Returns:
            Path to the final epoch checkpoint.
        """
        config = self.config

        # Build optimizer BEFORE accelerator.prepare
        optimizer = build_optimizer(self.model, config)

        # Approximate steps per epoch (IterableDataset has no __len__)
        steps_per_epoch = config.data.samples_per_batch

        scheduler = build_scheduler(optimizer, config, steps_per_epoch)

        # Prepare model, optimizer, scheduler together
        self.model, optimizer, scheduler = self.accelerator.prepare(
            self.model, optimizer, scheduler
        )

        final_ckpt_path: Path | None = None

        for epoch in range(config.training.num_epochs):
            # Recreate dataset each epoch for fresh seeds (Pitfall 5)
            dataset = LLMGeneratedDataset(
                config,
                self.tokenizer,
                epoch=epoch,
                cache_path=self.cache_path,
            )
            dataloader = DataLoader(dataset, batch_size=config.training.batch_size)
            dataloader = self.accelerator.prepare(dataloader)

            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                with self.accelerator.autocast():
                    output = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    # Non-CRF: TokenClassifierOutput with .loss
                    # CRF: tuple (loss, emissions)
                    if isinstance(output, tuple):
                        loss = output[0]
                    else:
                        loss = output.loss

                self.accelerator.backward(loss)
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    config.training.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info("Epoch %d/%d — avg loss: %.4f", epoch + 1, config.training.num_epochs, avg_loss)

            # Save checkpoint after each epoch
            final_ckpt_path = save_checkpoint(
                self.model,
                optimizer,
                scheduler,
                epoch=epoch,
                config=config,
                accelerator=self.accelerator,
                run_id=self.run_id,
            )

        return final_ckpt_path


# ---------------------------------------------------------------------------
# Majority vote (ensemble aggregation)
# ---------------------------------------------------------------------------

def majority_vote(predictions: list[list[int]]) -> list[int]:
    """Aggregate N model predictions by per-position majority vote.

    Ties are broken by taking the label that appears first in Counter's
    most_common() output (deterministic for same inputs).

    Args:
        predictions: List of N prediction sequences; each is a list of
            integer BIO labels (same length).

    Returns:
        List of integer labels, one per position.
    """
    if not predictions:
        return []
    seq_len = len(predictions[0])
    result = []
    for pos in range(seq_len):
        labels_at_pos = [pred[pos] for pred in predictions]
        majority_label = collections.Counter(labels_at_pos).most_common(1)[0][0]
        result.append(majority_label)
    return result


# ---------------------------------------------------------------------------
# Ensemble driver
# ---------------------------------------------------------------------------

def train_ensemble(config, tokenizer, accelerator) -> list[Path]:
    """Train N models with bagging (cache handoff pattern).

    Model 0 — generates training data live, caches to ensemble_base.jsonl.
    Models 1..N-1 — read from the base cache (no LLM calls).

    Each estimator gets a seed offset: config.project.seed + i * 1000.

    Args:
        config: Full config with ensemble.n_estimators and data.cache_dir.
        tokenizer: BertTokenizerFast (or mock).
        accelerator: accelerate.Accelerator instance.

    Returns:
        List of checkpoint Paths, one per estimator.
    """
    n = config.ensemble.n_estimators
    use_gradient_boost = getattr(config.ensemble, "use_gradient_boost", False)

    if use_gradient_boost:
        logger.warning(
            "gradient-boost ensemble not yet implemented, falling back to uniform resampling"
        )

    base_cache_path = Path(config.data.cache_dir) / "ensemble_base.jsonl"
    checkpoint_paths: list[Path] = []

    for i in range(n):
        # Offset seed for diversity
        seed_cfg = copy.deepcopy(config)
        seed_cfg.project = copy.copy(config.project)
        seed_cfg.project.seed = config.project.seed + i * 1000

        if i == 0:
            # First model: generate live, no cache
            estimator_cfg = seed_cfg
            cache_path_for_trainer = None
        else:
            # Subsequent models: read from base cache
            estimator_cfg = copy.deepcopy(seed_cfg)
            estimator_cfg.data = copy.copy(seed_cfg.data)
            estimator_cfg.data.cache_dir = str(base_cache_path)
            estimator_cfg.data.cache_path = base_cache_path
            cache_path_for_trainer = base_cache_path

        model = RegulatoryNERModel(estimator_cfg)
        trainer = Trainer(
            estimator_cfg, model, tokenizer, accelerator,
            run_id=f"ensemble_{i}",
            cache_path=cache_path_for_trainer,
        )
        ckpt = trainer.train()
        checkpoint_paths.append(ckpt)

    return checkpoint_paths
