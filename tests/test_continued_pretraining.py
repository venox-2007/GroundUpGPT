"""Focused tests for Step 20: Continued pretraining, resumption, and checkpoint protection."""

import json
from pathlib import Path
import tempfile
import pytest
import torch

from src.bpe_tokenizer import BPETokenizer
from src.model import GPTConfig, GroundUpGPT
from src.token_cache import PretokenizedDataPipeline
from src.trainer import Trainer, TrainingConfig
from scripts.train_continued_5m import safe_save_checkpoint


@pytest.fixture
def mini_setup():
    """Create a minimal model and pipeline for fast checkpoint tests."""
    tokenizer = BPETokenizer()
    vocab_size = tokenizer.vocab_size  # 256 base bytes
    train_tokens = torch.randint(0, vocab_size, (500,), dtype=torch.long)
    val_tokens = torch.randint(0, vocab_size, (200,), dtype=torch.long)
    pipeline = PretokenizedDataPipeline(train_tokens=train_tokens, val_tokens=val_tokens, tokenizer=tokenizer)

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=16,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
    )
    model = GroundUpGPT(config)
    train_config = TrainingConfig(
        max_steps=100,
        batch_size=4,
        block_size=16,
        learning_rate=1e-3,
        min_lr=1e-4,
        warmup_steps=10,
        grad_accum_steps=1,
        eval_interval=10,
        checkpoint_interval=10,
        device="cpu",
        seed=42,
    )
    trainer = Trainer(model=model, pipeline=pipeline, config=train_config)
    return model, trainer, pipeline, tokenizer


def test_atomic_checkpoint_saving():
    """Verify safe_save_checkpoint writes cleanly and atomically."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        target = Path(tmp_dir) / "sub" / "test_model.pt"
        data = {"step": 42, "weight": torch.tensor([1.0, 2.0, 3.0])}
        safe_save_checkpoint(data, target)

        assert target.is_file(), "Target checkpoint must exist"
        assert not target.with_suffix(".tmp").exists(), "Temporary file must not linger"

        loaded = torch.load(target, map_location="cpu", weights_only=False)
        assert loaded["step"] == 42
        assert torch.equal(loaded["weight"], torch.tensor([1.0, 2.0, 3.0]))


def test_checkpoint_state_restoration(mini_setup):
    """Verify that Trainer.load_checkpoint restores step, weights, and optimizer state."""
    model, trainer, pipeline, tokenizer = mini_setup

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "checkpoint_step5.pt"

        # Advance trainer 5 steps
        for _ in range(5):
            trainer.train_step()
        assert trainer.current_step == 5
        trainer.save_checkpoint(ckpt_path, val_loss=2.5)

        # Create a fresh model and trainer
        fresh_model = GroundUpGPT(model.config)
        fresh_trainer = Trainer(model=fresh_model, pipeline=pipeline, config=trainer.config)
        assert fresh_trainer.current_step == 0

        # Load saved checkpoint
        restored_step = fresh_trainer.load_checkpoint(ckpt_path)
        assert restored_step == 5
        assert fresh_trainer.current_step == 5

        # Check weights match
        for p1, p2 in zip(trainer.model.parameters(), fresh_trainer.model.parameters()):
            assert torch.allclose(p1, p2), "Model parameters must match after restoration"

        # Check that the next train_step advances from step 5 with correct lr
        next_loss = fresh_trainer.train_step()
        assert fresh_trainer.current_step == 6
        expected_lr_6 = fresh_trainer.get_lr_for_step(5)
        for pg in fresh_trainer.optimizer.param_groups:
            assert abs(pg["lr"] - expected_lr_6) < 1e-8


def test_cosine_decay_schedule_continuity():
    """Verify learning rate schedule continues smoothly past step 200 in a 1,000-step budget."""
    config = TrainingConfig(
        max_steps=1000,
        learning_rate=6e-4,
        min_lr=6e-5,
        warmup_steps=20,
    )
    tokenizer = BPETokenizer()
    dummy_model = GroundUpGPT(GPTConfig(vocab_size=tokenizer.vocab_size, block_size=8, embed_dim=16, num_heads=2, num_layers=1))
    dummy_pipe = PretokenizedDataPipeline(train_tokens=torch.zeros(100, dtype=torch.long), val_tokens=torch.zeros(50, dtype=torch.long), tokenizer=tokenizer)
    trainer = Trainer(model=dummy_model, pipeline=dummy_pipe, config=config)

    lr_200 = trainer.get_lr_for_step(200)
    lr_250 = trainer.get_lr_for_step(250)
    lr_600 = trainer.get_lr_for_step(600)
    lr_1000 = trainer.get_lr_for_step(1000)

    assert 6e-5 < lr_200 < 6e-4
    assert lr_250 < lr_200, "LR must decrease monotonically during cosine decay"
    assert lr_600 < lr_250, "LR must decrease monotonically during cosine decay"
    assert abs(lr_1000 - 6e-5) < 1e-7, "LR at max_steps must clamp to min_lr"


def test_baseline_step200_protection():
    """Verify that the Step 200 checkpoint remains intact and unchanged."""
    repo_root = Path(__file__).resolve().parent.parent
    step200_path = repo_root / "checkpoints" / "5m_initial" / "model_5m_step200.pt"
    assert step200_path.is_file(), "Baseline Step 200 checkpoint must exist on disk"

    ckpt = torch.load(step200_path, map_location="cpu", weights_only=False)
    assert ckpt["step"] == 200, "Step 200 checkpoint step must be 200"
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert ckpt["model_config"]["vocab_size"] == 4096
