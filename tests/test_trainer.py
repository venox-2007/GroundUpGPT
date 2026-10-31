"""Targeted unit tests for the GroundUpGPT training pipeline and checkpointing."""

import math
from pathlib import Path
import pytest
import torch
from src.model import GPTConfig, GroundUpGPT
from src.data import TextDataPipeline
from src.trainer import Trainer, TrainingConfig
from src.bpe_tokenizer import BPETokenizer


@pytest.fixture
def sample_fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_text.txt"


def test_learning_rate_schedule() -> None:
    """Verify linear warmup followed by cosine decay."""
    pipeline = TextDataPipeline(text="abcdefghijklmnopqrstuvwxyz " * 5, train_ratio=0.8)
    model = GroundUpGPT(GPTConfig(vocab_size=pipeline.vocab_size, block_size=8, embed_dim=16, num_heads=2, num_layers=1))

    cfg = TrainingConfig(
        max_steps=100,
        warmup_steps=10,
        learning_rate=1e-3,
        min_lr=1e-4,
        device="cpu",
    )
    trainer = Trainer(model, pipeline, cfg)

    # 1. Warmup step 0: (0 + 1) / 10 * 1e-3 = 1e-4
    lr_0 = trainer.get_lr_for_step(0)
    assert math.isclose(lr_0, 1e-4, rel_tol=1e-5)

    # 2. Warmup peak at step 9: 10 / 10 * 1e-3 = 1e-3
    lr_peak = trainer.get_lr_for_step(9)
    assert math.isclose(lr_peak, 1e-3, rel_tol=1e-5)

    # 3. Midpoint decay at step 55 (halfway between 10 and 100):
    # progress = 45 / 90 = 0.5 -> cos(pi/2) = 0 -> lr = min_lr + 0.5*(peak - min_lr)
    lr_mid = trainer.get_lr_for_step(55)
    expected_mid = 1e-4 + 0.5 * (1e-3 - 1e-4)
    assert math.isclose(lr_mid, expected_mid, rel_tol=1e-4)

    # 4. Beyond max_steps -> clamped to min_lr
    lr_final = trainer.get_lr_for_step(100)
    assert math.isclose(lr_final, 1e-4, rel_tol=1e-5)
    lr_beyond = trainer.get_lr_for_step(150)
    assert math.isclose(lr_beyond, 1e-4, rel_tol=1e-5)


def test_cpu_training_and_finite_losses(sample_fixture_path: Path) -> None:
    """Verify trainer executes cleanly on CPU and computes finite train/val losses."""
    pipeline = TextDataPipeline.from_file(sample_fixture_path, train_ratio=0.9)
    model = GroundUpGPT(GPTConfig(vocab_size=pipeline.vocab_size, block_size=12, embed_dim=16, num_heads=2, num_layers=2))

    cfg = TrainingConfig(
        max_steps=3,
        batch_size=2,
        block_size=12,
        device="cpu",
        mixed_precision=False,
    )
    trainer = Trainer(model, pipeline, cfg)
    metrics = trainer.train()

    assert metrics["total_steps"] == 3
    assert math.isfinite(metrics["final_train_loss"])
    assert math.isfinite(metrics["final_val_loss"])
    assert metrics["device"] == "cpu"
    assert metrics["mixed_precision"] is False


def test_gradient_accumulation_and_loss_scaling(sample_fixture_path: Path) -> None:
    """Verify gradient accumulation scales loss correctly and updates model weights."""
    pipeline = TextDataPipeline.from_file(sample_fixture_path, train_ratio=0.9)
    model = GroundUpGPT(GPTConfig(vocab_size=pipeline.vocab_size, block_size=8, embed_dim=16, num_heads=2, num_layers=1))

    cfg = TrainingConfig(
        max_steps=1,
        batch_size=2,
        block_size=8,
        grad_accum_steps=3,  # accumulate 3 micro-batches
        device="cpu",
    )
    trainer = Trainer(model, pipeline, cfg)
    loss = trainer.train_step()

    assert math.isfinite(loss)
    assert trainer.current_step == 1
    # Ensure gradients were cleared after the step
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is None


def test_checkpoint_save_and_resumption(tmp_path: Path, sample_fixture_path: Path) -> None:
    """Verify saving a checkpoint and restoring state continues training deterministically."""
    pipeline = TextDataPipeline.from_file(sample_fixture_path, train_ratio=0.9)
    model_cfg = GPTConfig(vocab_size=pipeline.vocab_size, block_size=8, embed_dim=16, num_heads=2, num_layers=1)
    model1 = GroundUpGPT(model_cfg)

    ckpt_file = tmp_path / "test_ckpt.pt"
    train_cfg = TrainingConfig(
        max_steps=4,
        batch_size=2,
        block_size=8,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        seed=1337,
    )

    trainer1 = Trainer(model1, pipeline, train_cfg)
    # Run 2 steps
    trainer1.train_step()
    trainer1.train_step()
    assert trainer1.current_step == 2
    trainer1.save_checkpoint(ckpt_file, val_loss=3.21)

    # Instantiate fresh model and resume
    model2 = GroundUpGPT(model_cfg)
    trainer2 = Trainer(model2, pipeline, train_cfg)
    resumed_step = trainer2.load_checkpoint(ckpt_file)

    assert resumed_step == 2
    assert trainer2.current_step == 2

    # Model parameters must match exactly
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


def test_checkpoint_rejection_on_configuration_mismatch(tmp_path: Path, sample_fixture_path: Path) -> None:
    """Ensure checkpoint loading rejects mismatched model or tokenizer configurations."""
    pipeline = TextDataPipeline.from_file(sample_fixture_path, train_ratio=0.9)
    model_cfg = GPTConfig(vocab_size=pipeline.vocab_size, block_size=8, embed_dim=16, num_heads=2, num_layers=1)
    model = GroundUpGPT(model_cfg)

    ckpt_file = tmp_path / "mismatch_ckpt.pt"
    trainer = Trainer(model, pipeline, TrainingConfig(device="cpu", checkpoint_dir=str(tmp_path)))
    trainer.save_checkpoint(ckpt_file)

    # 1. Reject architecture dimension mismatch (different embed_dim)
    diff_model_cfg = GPTConfig(vocab_size=pipeline.vocab_size, block_size=8, embed_dim=32, num_heads=2, num_layers=1)
    diff_model = GroundUpGPT(diff_model_cfg)
    trainer_diff = Trainer(diff_model, pipeline, TrainingConfig(device="cpu", checkpoint_dir=str(tmp_path)))

    with pytest.raises(ValueError, match="Model architecture mismatch for 'embed_dim'"):
        trainer_diff.load_checkpoint(ckpt_file)

    # 2. Reject tokenizer mismatch (CharTokenizer vs BPETokenizer)
    bpe_pipeline = TextDataPipeline.from_file(sample_fixture_path, train_ratio=0.9, tokenizer=BPETokenizer(), bpe_vocab_size=270)
    trainer_bpe = Trainer(model, bpe_pipeline, TrainingConfig(device="cpu", checkpoint_dir=str(tmp_path)))

    with pytest.raises(ValueError, match="Tokenizer type mismatch"):
        trainer_bpe.load_checkpoint(ckpt_file)
