"""Unit tests for the 5.05M GroundUpGPT training infrastructure and checkpoint compatibility."""

from pathlib import Path
import pytest
import torch

from src.bpe_tokenizer import BPETokenizer
from src.model import GPTConfig, GroundUpGPT
from src.token_cache import PretokenizedDataPipeline
from src.trainer import Trainer, TrainingConfig


@pytest.fixture
def mock_5m_setup(tmp_path: Path):
    """Fixture providing a lightweight 5M-compatible pipeline and model on CPU."""
    # Build a tiny mock tokenizer with 4096 vocabulary
    tokenizer = BPETokenizer()
    for i in range(4096 - 256):
        tokenizer.vocab[256 + i] = bytes([i % 256])
    tokenizer._is_fitted = True

    # Synthetic tokens
    train_tokens = torch.randint(0, 4096, (2048,), dtype=torch.int32)
    val_tokens = torch.randint(0, 4096, (512,), dtype=torch.int32)

    pipeline = PretokenizedDataPipeline(
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        tokenizer=tokenizer,
    )

    config = GPTConfig(
        vocab_size=4096,
        block_size=64,  # Small block for fast unit test
        embed_dim=256,
        num_heads=4,
        num_layers=5,
        dropout=0.1,
        bias=False,
        weight_tying=True,
    )
    model = GroundUpGPT(config)
    return model, pipeline, tmp_path


def test_batch_forward_backward_finite_loss(mock_5m_setup):
    """Verify that batches execute forward and backward passes with finite losses."""
    model, pipeline, _ = mock_5m_setup
    x, y = pipeline.get_batch("train", batch_size=4, block_size=64)

    logits, loss = model(x, targets=y)
    assert loss is not None
    assert torch.isfinite(loss).item()
    assert logits.shape == (4, 64, 4096)

    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_checkpoint_save_restore_and_resumption(mock_5m_setup):
    """Verify saving, loading, and continuing training from a checkpoint."""
    model, pipeline, tmp_path = mock_5m_setup
    ckpt_dir = tmp_path / "checkpoints"

    cfg = TrainingConfig(
        max_steps=10,
        batch_size=2,
        block_size=64,
        learning_rate=1e-3,
        grad_accum_steps=1,
        eval_interval=5,
        checkpoint_interval=5,
        checkpoint_dir=str(ckpt_dir),
        device="cpu",
    )

    trainer1 = Trainer(model=model, pipeline=pipeline, config=cfg)
    results1 = trainer1.train(stop_after_steps=4)
    assert results1["total_steps"] == 4

    ckpt_file = ckpt_dir / "step_4.pt"
    assert ckpt_file.is_file()

    # Create fresh model and resume
    model2 = GroundUpGPT(model.config)
    trainer2 = Trainer(model=model2, pipeline=pipeline, config=cfg)
    restored_step = trainer2.load_checkpoint(ckpt_file)
    assert restored_step == 4

    # Continue training from step 4 to step 8
    results2 = trainer2.train(stop_after_steps=8)
    assert results2["total_steps"] == 8
    assert results2["steps_this_run"] == 4


def test_time_budget_stopping(mock_5m_setup):
    """Verify that train() halts when time_budget_seconds is exceeded."""
    model, pipeline, tmp_path = mock_5m_setup
    cfg = TrainingConfig(
        max_steps=1000,
        batch_size=2,
        block_size=64,
        checkpoint_dir=str(tmp_path / "ckpts"),
        device="cpu",
    )

    trainer = Trainer(model=model, pipeline=pipeline, config=cfg)
    results = trainer.train(time_budget_seconds=0.1)

    assert results["stopped_due_to_time"] is True
    assert results["total_steps"] < 1000
