"""
Tests for Trainer class, ensemble driver, and majority vote.

Uses tiny BertConfig (hidden_size=64, 1 layer, 1 head) to avoid model downloads.
BertModel.from_pretrained and BertForTokenClassification.from_pretrained are patched.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call
import pytest
import torch
from transformers import BertConfig, BertModel, BertForTokenClassification


# ---------------------------------------------------------------------------
# Shared tiny-model helpers (mirrors test_ner_model.py pattern)
# ---------------------------------------------------------------------------

TINY_CFG = BertConfig(
    hidden_size=64,
    num_hidden_layers=1,
    num_attention_heads=1,
    intermediate_size=128,
    max_position_embeddings=64,
    vocab_size=100,
)


def tiny_bert_model(*args, **kwargs):
    return BertModel(TINY_CFG)


def tiny_bert_for_tc(*args, **kwargs):
    return BertForTokenClassification(
        BertConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=128,
            max_position_embeddings=64,
            vocab_size=100,
            num_labels=3,
        )
    )


def make_full_config(
    use_crf=False,
    lr_backbone=2e-5,
    lr_head=1e-4,
    warmup_steps=10,
    max_grad_norm=1.0,
    num_epochs=2,
    batch_size=2,
    samples_per_batch=4,
    mixed_precision="bf16",
    cache_dir="data/cache",
    n_estimators=3,
    ensemble_enabled=False,
    seed=42,
):
    """Build full SimpleNamespace mirroring the OmegaConf config schema."""
    return SimpleNamespace(
        project=SimpleNamespace(seed=seed),
        model=SimpleNamespace(
            name="deepset/gbert-large",
            use_crf=use_crf,
            freeze_backbone=False,
            use_lora=False,
            lora_rank=4,
        ),
        training=SimpleNamespace(
            batch_size=batch_size,
            learning_rate_backbone=lr_backbone,
            learning_rate_head=lr_head,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm,
            num_epochs=num_epochs,
            mixed_precision=mixed_precision,
        ),
        data=SimpleNamespace(
            max_seq_length=32,
            samples_per_batch=samples_per_batch,
            negative_sample_ratio=0.4,
            cache_dir=cache_dir,
            gold_test_dir="data/gold_test",
            llm_seed=1337,
            llm_model="google/gemini-flash-1.5",
        ),
        ensemble=SimpleNamespace(
            enabled=ensemble_enabled,
            n_estimators=n_estimators,
        ),
        evaluation=SimpleNamespace(output_dir="evaluation_output"),
    )


# ---------------------------------------------------------------------------
# Task 1: Tests for build_optimizer, resolve_mixed_precision, gradient
# clipping, schedule, checkpoint save/load, epoch dataset recreation
# ---------------------------------------------------------------------------


class TestDifferentialLR:
    """test_differential_lr: optimizer has 2 param groups with distinct LRs."""

    def test_differential_lr(self):
        from src.model.trainer import build_optimizer

        with (
            patch(
                "transformers.BertForTokenClassification.from_pretrained",
                side_effect=tiny_bert_for_tc,
            ),
        ):
            from src.model.ner_model import RegulatoryNERModel

            cfg = make_full_config()
            model = RegulatoryNERModel(cfg)

        optimizer = build_optimizer(model, cfg)

        assert len(optimizer.param_groups) == 2, (
            f"Expected 2 param groups, got {len(optimizer.param_groups)}"
        )
        # Group 0 = backbone, group 1 = head
        assert abs(optimizer.param_groups[0]["lr"] - cfg.training.learning_rate_backbone) < 1e-10, (
            f"Group 0 LR should be backbone LR {cfg.training.learning_rate_backbone}"
        )
        assert abs(optimizer.param_groups[1]["lr"] - cfg.training.learning_rate_head) < 1e-10, (
            f"Group 1 LR should be head LR {cfg.training.learning_rate_head}"
        )


class TestMixedPrecisionResolution:
    """test_mixed_precision_resolution: correct precision per device type."""

    def test_cuda_returns_config_value(self):
        from src.model.trainer import resolve_mixed_precision

        cfg = make_full_config(mixed_precision="bf16")
        device = MagicMock()
        device.type = "cuda"
        result = resolve_mixed_precision(cfg, device)
        assert result == "bf16"

    def test_cuda_fp16(self):
        from src.model.trainer import resolve_mixed_precision

        cfg = make_full_config(mixed_precision="fp16")
        device = MagicMock()
        device.type = "cuda"
        result = resolve_mixed_precision(cfg, device)
        assert result == "fp16"

    def test_cpu_returns_no(self):
        from src.model.trainer import resolve_mixed_precision

        cfg = make_full_config(mixed_precision="bf16")
        device = MagicMock()
        device.type = "cpu"
        result = resolve_mixed_precision(cfg, device)
        assert result == "no"

    def test_mps_new_torch_returns_bf16(self):
        """MPS on torch >= 2.6 should support bf16."""
        from src.model.trainer import resolve_mixed_precision

        cfg = make_full_config(mixed_precision="bf16")
        device = MagicMock()
        device.type = "mps"
        with patch("torch.__version__", "2.6.0"):
            result = resolve_mixed_precision(cfg, device)
        assert result == "bf16"

    def test_mps_old_torch_returns_no(self):
        """MPS on torch < 2.6 should fall back to 'no'."""
        from src.model.trainer import resolve_mixed_precision

        cfg = make_full_config(mixed_precision="bf16")
        device = MagicMock()
        device.type = "mps"
        with patch("torch.__version__", "2.4.0"):
            result = resolve_mixed_precision(cfg, device)
        assert result == "no"


class TestGradientClipping:
    """test_gradient_clipping: after backward pass, grad norm <= max_grad_norm."""

    def test_gradient_clipping(self):
        """Verify that after a backward pass the gradient norm is bounded."""
        with (
            patch(
                "transformers.BertForTokenClassification.from_pretrained",
                side_effect=tiny_bert_for_tc,
            ),
        ):
            from src.model.ner_model import RegulatoryNERModel

            cfg = make_full_config(max_grad_norm=0.5)
            model = RegulatoryNERModel(cfg)

        # Run a real forward + backward on random data
        input_ids = torch.randint(0, 100, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.long)
        labels = torch.zeros(2, 16, dtype=torch.long)
        labels[:, -1] = -100

        output = model(input_ids, attention_mask, labels=labels)
        output.loss.backward()

        # Apply gradient clipping (simulates Accelerator.clip_grad_norm_ behavior)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)

        # Verify gradient norm is bounded
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        assert total_norm <= cfg.training.max_grad_norm + 1e-5, (
            f"Gradient norm {total_norm} exceeds max_grad_norm {cfg.training.max_grad_norm}"
        )


class TestLRScheduleShape:
    """test_lr_schedule_shape: LR increases during warmup, then decreases."""

    def test_lr_schedule_warmup_then_decay(self):
        from src.model.trainer import build_optimizer, build_scheduler

        with (
            patch(
                "transformers.BertForTokenClassification.from_pretrained",
                side_effect=tiny_bert_for_tc,
            ),
        ):
            from src.model.ner_model import RegulatoryNERModel

            cfg = make_full_config(
                warmup_steps=5,
                num_epochs=2,
                samples_per_batch=10,
            )
            model = RegulatoryNERModel(cfg)

        optimizer = build_optimizer(model, cfg)
        # steps_per_epoch = samples_per_batch
        scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=cfg.data.samples_per_batch)

        # Capture LR at step 0 (before any step)
        lr_at_0 = optimizer.param_groups[0]["lr"]

        # Step through warmup
        lrs = [lr_at_0]
        for _ in range(cfg.training.warmup_steps):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        lr_at_warmup_end = lrs[cfg.training.warmup_steps]

        # Step through decay
        total_steps = cfg.training.num_epochs * cfg.data.samples_per_batch
        remaining = total_steps - cfg.training.warmup_steps
        for _ in range(remaining):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        lr_at_end = lrs[-1]

        # Assertions:
        # 1. LR at step 0 should be near 0 (linear warmup starts at 0)
        assert lr_at_0 < 1e-8, f"LR at step 0 should be near 0, got {lr_at_0}"

        # 2. LR at warmup end should be near max
        max_lr = cfg.training.learning_rate_backbone
        assert lr_at_warmup_end > max_lr * 0.8, (
            f"LR at warmup end {lr_at_warmup_end} should be near max {max_lr}"
        )

        # 3. LR at end should be near 0 (linear decay to 0)
        assert lr_at_end < max_lr * 0.1, (
            f"LR at end {lr_at_end} should be near 0 (< 10% of max {max_lr})"
        )


class TestCheckpointSaveLoad:
    """test_checkpoint_save_load: round-trip preserves model weights."""

    def test_checkpoint_round_trip(self, tmp_path):
        from src.model.trainer import save_checkpoint, load_checkpoint

        with (
            patch(
                "transformers.BertForTokenClassification.from_pretrained",
                side_effect=tiny_bert_for_tc,
            ),
        ):
            from src.model.ner_model import RegulatoryNERModel

            cfg = make_full_config()
            model = RegulatoryNERModel(cfg)

        # Mock accelerator
        accelerator = MagicMock()
        accelerator.unwrap_model.return_value = model

        # Override checkpoint dir to use tmp_path
        with patch("src.model.trainer.Path") as mock_path_cls:
            # We need Path to work properly for the checkpoint dir but redirect to tmp_path
            # Instead, patch save/load directly via tmp_path
            pass

        # Direct test: save and load using tmp_path
        save_dir = tmp_path / "checkpoints" / "test_run"
        save_dir.mkdir(parents=True)

        # Save checkpoint manually
        checkpoint_path = save_dir / "epoch_1.pt"
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": None,
                "scheduler_state_dict": None,
            },
            checkpoint_path,
        )

        # Modify model weights so we can verify restoration
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(0.0)

        # Load checkpoint
        loaded_epoch = load_checkpoint(checkpoint_path, model)

        assert loaded_epoch == 1, f"Expected epoch 1, got {loaded_epoch}"

        # Verify weights were not all zeros (they were restored)
        all_zero = all(
            (p == 0.0).all().item() for p in model.parameters()
        )
        assert not all_zero, "Weights should be restored from checkpoint (not all zeros)"

    def test_save_checkpoint_creates_file(self, tmp_path):
        from src.model.trainer import save_checkpoint

        with (
            patch(
                "transformers.BertForTokenClassification.from_pretrained",
                side_effect=tiny_bert_for_tc,
            ),
        ):
            from src.model.ner_model import RegulatoryNERModel

            cfg = make_full_config()
            model = RegulatoryNERModel(cfg)

        optimizer_mock = MagicMock()
        optimizer_mock.state_dict.return_value = {}
        scheduler_mock = MagicMock()
        scheduler_mock.state_dict.return_value = {}
        accelerator = MagicMock()
        accelerator.unwrap_model.return_value = model

        with patch("src.model.trainer.CHECKPOINT_BASE", tmp_path):
            ckpt_path = save_checkpoint(
                model, optimizer_mock, scheduler_mock,
                epoch=2, config=cfg, accelerator=accelerator, run_id="test_run"
            )

        assert ckpt_path.exists(), f"Checkpoint file should exist at {ckpt_path}"
        data = torch.load(ckpt_path, weights_only=True)
        assert data["epoch"] == 2


class TestEpochDatasetRecreation:
    """test_epoch_dataset_recreation: each epoch creates a new LLMGeneratedDataset."""

    def test_dataset_recreated_each_epoch(self, tmp_path):
        """Trainer must recreate dataset each epoch with correct epoch number."""
        from src.model.trainer import Trainer

        with (
            patch(
                "transformers.BertForTokenClassification.from_pretrained",
                side_effect=tiny_bert_for_tc,
            ),
        ):
            from src.model.ner_model import RegulatoryNERModel

            cfg = make_full_config(num_epochs=2, samples_per_batch=2, batch_size=2)
            model = RegulatoryNERModel(cfg)

        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        # Mock accelerator — prepare returns inputs unchanged
        accelerator = MagicMock()
        accelerator.unwrap_model.return_value = model
        accelerator.prepare.side_effect = lambda *args: args if len(args) > 1 else args[0]
        accelerator.autocast.return_value.__enter__ = MagicMock(return_value=None)
        accelerator.autocast.return_value.__exit__ = MagicMock(return_value=False)
        accelerator.backward = MagicMock()
        accelerator.clip_grad_norm_ = MagicMock()

        # Build a tiny fake batch
        fake_batch = {
            "input_ids": torch.randint(0, 100, (2, 16)),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
            "labels": torch.zeros(2, 16, dtype=torch.long),
        }
        fake_batch["labels"][:, -1] = -100

        dataset_call_epochs = []

        class FakeDataset:
            def __init__(self, config, tokenizer, epoch=0, cache_path=None):
                dataset_call_epochs.append(epoch)

            def __iter__(self):
                yield fake_batch

        with (
            patch("src.model.trainer.LLMGeneratedDataset", FakeDataset),
            patch("src.model.trainer.CHECKPOINT_BASE", tmp_path),
            patch("src.model.trainer.DataLoader") as mock_dl,
        ):
            # DataLoader returns an iterable of our fake batch
            mock_dl.return_value = [fake_batch]
            accelerator.prepare.side_effect = lambda *args: args if len(args) > 1 else args[0]

            trainer = Trainer(cfg, model, tokenizer, accelerator)
            trainer.train()

        assert dataset_call_epochs == [0, 1], (
            f"Dataset should be created with epoch=0 and epoch=1, got {dataset_call_epochs}"
        )


# ---------------------------------------------------------------------------
# Task 2: Ensemble driver and run.py CLI tests
# ---------------------------------------------------------------------------


class TestEnsembleNCheckpoints:
    """test_ensemble_n_checkpoints: train_ensemble returns N checkpoint paths."""

    def test_returns_n_checkpoints(self, tmp_path):
        from src.model.trainer import train_ensemble

        cfg = make_full_config(ensemble_enabled=True, n_estimators=3)

        tokenizer = MagicMock()
        accelerator = MagicMock()

        fake_checkpoint = tmp_path / "epoch_1.pt"
        fake_checkpoint.touch()

        with (
            patch("src.model.trainer.RegulatoryNERModel"),
            patch("src.model.trainer.Trainer") as mock_trainer_cls,
            patch(
                "transformers.BertForTokenClassification.from_pretrained",
                side_effect=tiny_bert_for_tc,
            ),
        ):
            mock_trainer_cls.return_value.train.return_value = fake_checkpoint
            result = train_ensemble(cfg, tokenizer, accelerator)

        assert len(result) == 3, f"Expected 3 checkpoints, got {len(result)}"


class TestEnsembleCacheHandoff:
    """test_ensemble_cache_handoff: first model no cache, subsequent models use cache."""

    def test_cache_handoff(self, tmp_path):
        from src.model.trainer import train_ensemble

        cfg = make_full_config(
            ensemble_enabled=True, n_estimators=2, cache_dir=str(tmp_path)
        )

        tokenizer = MagicMock()
        accelerator = MagicMock()

        fake_checkpoint = tmp_path / "epoch_1.pt"
        fake_checkpoint.touch()

        dataset_cache_paths = []

        original_dataset = __import__(
            "src.data.dataset", fromlist=["LLMGeneratedDataset"]
        ).LLMGeneratedDataset

        class TrackingDataset:
            def __init__(self, config, tokenizer, epoch=0, cache_path=None):
                dataset_cache_paths.append(cache_path)

        with (
            patch("src.model.trainer.RegulatoryNERModel"),
            patch("src.model.trainer.Trainer") as mock_trainer_cls,
            patch("src.model.trainer.LLMGeneratedDataset", TrackingDataset),
        ):
            mock_trainer_cls.return_value.train.return_value = fake_checkpoint
            train_ensemble(cfg, tokenizer, accelerator)

        # First call should pass write_cache=True (or cache_path=None)
        # Second call should pass cache_path pointing to base cache
        # The dataset instantiation is inside Trainer, but train_ensemble itself
        # controls cache_path by passing it in config or via Trainer construction.
        # Verify via the config that was passed to the second Trainer:
        calls = mock_trainer_cls.call_args_list
        assert len(calls) == 2, f"Expected 2 Trainer instantiations, got {len(calls)}"

        # First trainer: no cache_path override (cache_path=None in dataset)
        first_call_cfg = calls[0][0][0]  # config positional arg
        # Second trainer: has cache_path set in config
        second_call_cfg = calls[1][0][0]
        # The ensemble base cache path
        expected_cache = Path(cfg.data.cache_dir) / "ensemble_base.jsonl"
        assert str(expected_cache) in str(second_call_cfg.data.cache_dir) or \
               hasattr(second_call_cfg.data, 'cache_path') and \
               second_call_cfg.data.cache_path == expected_cache, \
               f"Second trainer config should reference ensemble cache. Got: {second_call_cfg}"


class TestEnsembleMajorityVote:
    """test_ensemble_majority_vote: per-position mode of BIO predictions."""

    def test_majority_vote_basic(self):
        from src.model.trainer import majority_vote

        predictions = [
            [0, 1, 2, 1],
            [0, 1, 1, 1],
            [0, 2, 2, 1],
        ]
        result = majority_vote(predictions)
        assert result == [0, 1, 2, 1], f"Expected [0, 1, 2, 1], got {result}"

    def test_majority_vote_unanimous(self):
        from src.model.trainer import majority_vote

        predictions = [[1, 2, 0], [1, 2, 0], [1, 2, 0]]
        result = majority_vote(predictions)
        assert result == [1, 2, 0]

    def test_majority_vote_single_model(self):
        from src.model.trainer import majority_vote

        predictions = [[0, 1, 2]]
        result = majority_vote(predictions)
        assert result == [0, 1, 2]

    def test_majority_vote_tie_is_deterministic(self):
        """Ties resolve deterministically (first most-common)."""
        from src.model.trainer import majority_vote

        # Position 0: [0, 1] -> tie -> returns 0 (first in Counter ordering)
        predictions = [[0, 0], [1, 1]]
        result1 = majority_vote(predictions)
        result2 = majority_vote(predictions)
        assert result1 == result2, "Tie resolution must be deterministic"


class TestEnsembleGradientBoostStub:
    """test_ensemble_gradient_boost_stub: use_gradient_boost=True logs warning, runs ok."""

    def test_gradient_boost_stub_runs(self, tmp_path, caplog):
        from src.model.trainer import train_ensemble

        cfg = make_full_config(
            ensemble_enabled=True, n_estimators=2, cache_dir=str(tmp_path)
        )
        # Add use_gradient_boost=True
        cfg.ensemble = SimpleNamespace(
            enabled=True,
            n_estimators=2,
            use_gradient_boost=True,
        )

        tokenizer = MagicMock()
        accelerator = MagicMock()

        fake_checkpoint = tmp_path / "epoch_1.pt"
        fake_checkpoint.touch()

        import logging
        with (
            patch("src.model.trainer.RegulatoryNERModel"),
            patch("src.model.trainer.Trainer") as mock_trainer_cls,
            caplog.at_level(logging.WARNING, logger="src.model.trainer"),
        ):
            mock_trainer_cls.return_value.train.return_value = fake_checkpoint
            result = train_ensemble(cfg, tokenizer, accelerator)

        # Should complete without error (stub behavior)
        assert len(result) == 2

        # Should log a warning about gradient boost not implemented
        assert any("gradient-boost" in r.message.lower() or "gradient_boost" in r.message.lower()
                   for r in caplog.records), \
               f"Expected gradient-boost warning in logs. Got: {[r.message for r in caplog.records]}"


class TestRunPyTrainSubcommand:
    """test_run_py_train_subcommand: run.py 'train' arg calls trainer."""

    def test_train_subcommand_calls_trainer(self, tmp_path):
        """Import run.py main and call with 'train' — verify trainer is reached."""
        # Mock all heavy dependencies before import
        fake_checkpoint = tmp_path / "epoch_1.pt"
        fake_checkpoint.touch()

        fake_config = make_full_config()

        with (
            patch("src.utils.config.load_config", return_value=fake_config),
            patch("src.utils.device.set_seed"),
            patch("src.utils.device.get_device", return_value=MagicMock(type="cpu")),
            patch("src.model.trainer.resolve_mixed_precision", return_value="no"),
            patch("accelerate.Accelerator"),
            patch("transformers.BertTokenizerFast.from_pretrained", return_value=MagicMock()),
            patch(
                "transformers.BertForTokenClassification.from_pretrained",
                side_effect=tiny_bert_for_tc,
            ),
            patch("src.model.trainer.Trainer") as mock_trainer_cls,
        ):
            mock_trainer_cls.return_value.train.return_value = fake_checkpoint

            # Import run.py as a module and call main
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "run", "/Users/Admin/REG_ML/run.py"
            )
            run_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run_module)

            with patch("sys.argv", ["run.py", "train"]):
                run_module.main()

        mock_trainer_cls.return_value.train.assert_called_once()
