"""Test suite for Step 19: Reproducible language model evaluation and train/eval RNG isolation."""

from pathlib import Path
import pytest
import torch

from src.bpe_tokenizer import BPETokenizer
from src.data import TextDataPipeline
from src.model import GPTConfig, GroundUpGPT
from src.token_cache import PretokenizedDataPipeline
from src.trainer import Trainer, TrainingConfig
from scripts.evaluate_model import (
    compute_fixed_validation_benchmark,
    load_model_and_tokenizer,
    evaluate_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_fixed_validation_benchmark_is_strictly_deterministic() -> None:
    """Verify compute_fixed_validation_benchmark produces identical loss/perplexity on multiple calls."""
    torch.manual_seed(42)
    cfg = GPTConfig(vocab_size=64, block_size=16, embed_dim=32, num_heads=4, num_layers=2)
    model = GroundUpGPT(cfg)
    model.eval()

    tokens = torch.randint(0, 64, (500,), dtype=torch.long)

    res1 = compute_fixed_validation_benchmark(
        model=model,
        tokens=tokens,
        block_size=16,
        batch_size=8,
        num_eval_windows=16,
        seed=1337,
        device="cpu",
    )

    res2 = compute_fixed_validation_benchmark(
        model=model,
        tokens=tokens,
        block_size=16,
        batch_size=8,
        num_eval_windows=16,
        seed=1337,
        device="cpu",
    )

    assert res1["loss"] == res2["loss"]
    assert res1["perplexity"] == res2["perplexity"]
    assert res1["tokens_evaluated"] == res2["tokens_evaluated"]
    assert res1["benchmark_seed"] == 1337


def test_incompatible_tokenizer_fingerprint_rejection(tmp_path: Path) -> None:
    """Verify load_model_and_tokenizer rejects checkpoints with mismatched tokenizer fingerprints."""
    # 1. Create a dummy tokenizer
    tok_path = tmp_path / "tok.json"
    tok = BPETokenizer()
    tok.fit("abc def ghi", target_vocab_size=260)
    tok.save(tok_path)

    # 2. Create checkpoint with mismatched fingerprint
    ckpt_path = tmp_path / "mismatch.pt"
    m_cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=16, embed_dim=16, num_heads=2, num_layers=1)
    model = GroundUpGPT(m_cfg)
    ckpt = {
        "step": 1,
        "model_config": {
            "vocab_size": m_cfg.vocab_size,
            "block_size": m_cfg.block_size,
            "embed_dim": m_cfg.embed_dim,
            "num_heads": m_cfg.num_heads,
            "num_layers": m_cfg.num_layers,
            "dropout": 0.0,
            "bias": False,
            "weight_tying": True,
        },
        "model_state_dict": model.state_dict(),
        "tokenizer_type": "BPETokenizer",
        "tokenizer_vocab_size": tok.vocab_size,
        "tokenizer_fingerprint": "0000000000000000000000000000000000000000000000000000000000000000",
    }
    torch.save(ckpt, ckpt_path)

    with pytest.raises(ValueError, match="Tokenizer fingerprint mismatch"):
        load_model_and_tokenizer(ckpt_path, tok_path, device="cpu")


def test_training_eval_rng_isolation(tmp_path: Path) -> None:
    """Verify that calling estimate_loss does NOT advance or perturb the training batch RNG sequence."""
    tok = BPETokenizer()
    tok.fit("abc def ghi jkl mno pqr stu vwx yz", target_vocab_size=270)
    tokens = torch.randint(0, tok.vocab_size, (1000,), dtype=torch.long)
    pipeline1 = PretokenizedDataPipeline(train_tokens=tokens, val_tokens=tokens, tokenizer=tok)
    pipeline2 = PretokenizedDataPipeline(train_tokens=tokens, val_tokens=tokens, tokenizer=tok)

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=16, embed_dim=32, num_heads=4, num_layers=2)

    train_cfg1 = TrainingConfig(
        max_steps=5,
        batch_size=4,
        block_size=16,
        grad_accum_steps=1,
        seed=42,
        checkpoint_dir=str(tmp_path / "ckpt1"),
    )
    train_cfg2 = TrainingConfig(
        max_steps=5,
        batch_size=4,
        block_size=16,
        grad_accum_steps=1,
        seed=42,
        checkpoint_dir=str(tmp_path / "ckpt2"),
    )

    # Trainer 1: Never calls estimate_loss between steps
    torch.manual_seed(42)
    model1 = GroundUpGPT(cfg)
    trainer1 = Trainer(model1, pipeline1, train_cfg1)
    losses1 = []
    for _ in range(5):
        losses1.append(trainer1.train_step())

    # Trainer 2: Calls estimate_loss multiple times between steps
    torch.manual_seed(42)
    model2 = GroundUpGPT(cfg)
    trainer2 = Trainer(model2, pipeline2, train_cfg2)
    losses2 = []
    for _ in range(5):
        # Frequent evaluation
        trainer2.estimate_loss(eval_iters=3)
        trainer2.estimate_loss(eval_iters=2)
        losses2.append(trainer2.train_step())

    # Because evaluation RNG is isolated from training batch RNG, training losses must be identical
    for l1, l2 in zip(losses1, losses2):
        assert abs(l1 - l2) < 1e-6, f"Training sequence diverged! {l1} vs {l2}"


def test_step200_baseline_evaluation_record() -> None:
    """Verify the saved Step 200 baseline evaluation file exists, matches schema, and contains valid metrics."""
    baseline_file = REPO_ROOT / "logs" / "baseline_step200_evaluation.json"
    assert baseline_file.is_file(), "Baseline Step 200 evaluation file is missing!"

    import json
    with open(baseline_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["step"] == 200
    assert data["architecture"]["parameters"] == 5051904
    assert data["architecture"]["vocab_size"] == 4096
    assert data["architecture"]["num_layers"] == 5
    assert "evaluation" in data
    assert data["evaluation"]["loss"] > 0
    assert data["evaluation"]["perplexity"] > 0
    assert data["evaluation"]["benchmark_seed"] == 1337
    assert data["evaluation"]["tokens_evaluated"] == 16384
    assert "dataset" in data
    assert "samples" in data
    assert len(data["samples"]) == 4
