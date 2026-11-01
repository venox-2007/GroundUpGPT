"""Tests for Step 24: Phase 2 Quality Pretraining Pipeline.

Verifies:
1. Phase 2 learning rate schedule math (warmup, cosine decay, min_lr clamp).
2. Checkpoint compatibility, parameter count (5,051,904), and tokenizer fingerprint verification.
3. Safe atomic checkpoint saving logic.
4. Dual-track validation benchmark independence.
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest
import torch

from src.bpe_tokenizer import BPETokenizer
from src.model import GPTConfig, GroundUpGPT
from scripts.train_phase2_5m import compute_phase2_lr, safe_save_checkpoint
from scripts.evaluate_model import compute_fixed_validation_benchmark

EXPECTED_TOKENIZER_FINGERPRINT = "4138461786ea6a199ff232b2427f3cb71c60f351afa7241f75f8196e51bdae72"
EXPECTED_PARAMS = 5_051_904


def test_phase2_lr_schedule():
    """Verify Phase 2 learning rate schedule properties."""
    start_step = 1000
    total_steps = 2000
    peak = 2.5e-4
    min_lr = 3.0e-5
    warmup = 50

    # 1. Start of warmup
    lr_0 = compute_phase2_lr(1000, phase2_start_step=start_step, phase2_total_steps=total_steps, peak_lr=peak, min_lr=min_lr, warmup_steps=warmup)
    assert lr_0 > min_lr
    assert lr_0 < peak

    # 2. End of warmup
    lr_warmup = compute_phase2_lr(1000 + warmup - 1, phase2_start_step=start_step, phase2_total_steps=total_steps, peak_lr=peak, min_lr=min_lr, warmup_steps=warmup)
    assert math.isclose(lr_warmup, peak, rel_tol=1e-3)

    # 3. Midpoint of cosine decay
    mid_step = 1000 + warmup + (total_steps - warmup) // 2
    lr_mid = compute_phase2_lr(mid_step, phase2_start_step=start_step, phase2_total_steps=total_steps, peak_lr=peak, min_lr=min_lr, warmup_steps=warmup)
    expected_mid = min_lr + 0.5 * (peak - min_lr)
    assert math.isclose(lr_mid, expected_mid, rel_tol=1e-2)

    # 4. Beyond total steps -> min_lr
    lr_end = compute_phase2_lr(1000 + total_steps + 100, phase2_start_step=start_step, phase2_total_steps=total_steps, peak_lr=peak, min_lr=min_lr, warmup_steps=warmup)
    assert lr_end == min_lr


def test_step1000_checkpoint_resume_integrity():
    """Verify that the Step 1000 source checkpoint is valid, compatible, and uncorrupted."""
    ckpt_path = Path("checkpoints/5m_continued/model_5m_best.pt")
    assert ckpt_path.is_file(), f"Step 1000 checkpoint missing: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["step"] == 1000
    assert ckpt["tokenizer_fingerprint"] == EXPECTED_TOKENIZER_FINGERPRINT

    m_cfg = GPTConfig(**ckpt["model_config"])
    model = GroundUpGPT(m_cfg)
    assert model.get_num_params() == EXPECTED_PARAMS

    # Load weights
    model.load_state_dict(ckpt["model_state_dict"])
    # Check that weights contain no NaNs or Infs
    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), f"Non-finite weight detected in {name}"


def test_safe_save_checkpoint_atomic(tmp_path: Path):
    """Verify safe_save_checkpoint writes atomically and replaces target file cleanly."""
    dest = tmp_path / "test_model.pt"
    data = {"step": 42, "loss": 5.123}
    safe_save_checkpoint(data, dest)

    assert dest.is_file()
    # Temporary file should not remain
    assert not dest.with_suffix(".tmp").exists()

    loaded = torch.load(dest, weights_only=False)
    assert loaded["step"] == 42
    assert loaded["loss"] == 5.123


def test_dual_track_validation_runs_correctly():
    """Verify that both Track 1 (frozen val) and Track 2 (expanded val) can be evaluated."""
    frozen_path = Path("data/cache/val_tokens.pt")
    expanded_path = Path("data/cache_expanded/val_expanded_tokens.pt")

    if not frozen_path.exists() or not expanded_path.exists():
        pytest.skip("Validation token caches not yet present.")

    frozen_tokens = torch.load(frozen_path, weights_only=True).long()
    expanded_tokens = torch.load(expanded_path, weights_only=True).long()

    # Load model
    ckpt_path = Path("checkpoints/5m_continued/model_5m_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m_cfg = GPTConfig(**ckpt["model_config"])
    model = GroundUpGPT(m_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Benchmark on small subset (4 windows)
    bench_frozen = compute_fixed_validation_benchmark(
        model=model,
        tokens=frozen_tokens,
        block_size=m_cfg.block_size,
        batch_size=4,
        num_eval_windows=4,
        seed=1337,
        device="cpu",
    )
    bench_expanded = compute_fixed_validation_benchmark(
        model=model,
        tokens=expanded_tokens,
        block_size=m_cfg.block_size,
        batch_size=4,
        num_eval_windows=4,
        seed=1337,
        device="cpu",
    )

    assert math.isfinite(bench_frozen["loss"])
    assert math.isfinite(bench_frozen["perplexity"])
    assert math.isfinite(bench_expanded["loss"])
    assert math.isfinite(bench_expanded["perplexity"])
    assert bench_frozen["loss"] > 0
    assert bench_expanded["loss"] > 0
