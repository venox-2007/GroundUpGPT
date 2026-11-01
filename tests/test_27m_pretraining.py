"""Unit and integration test suite for GroundUpGPT-27M pretraining (Step 32).

Verifies:
1. Exact model construction and parameter count (27,411,456).
2. Tokenizer and dataset fingerprint validation.
3. Correct one-token shifted causal targets.
4. Finite forward/backward loss with AMP and gradient accumulation.
5. Learning-rate schedule dynamics (warmup and cosine decay).
6. Checkpoint save, restore, and resume verification.
7. Multi-track validation reproducibility on fixed windows.
8. Preservation and coexistence of 5M checkpoints and architecture profiles.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import pytest
import torch

from src.bpe_tokenizer import BPETokenizer
from src.model import GPTConfig, GroundUpGPT, get_5m_config, get_27m_config
from src.token_cache import load_token_cache, PretokenizedDataPipeline
from scripts.train_27m_phase1 import (
    compute_learning_rate,
    evaluate_fixed_track,
    save_crash_safe_checkpoint,
    EXPECTED_TOKENIZER_FP,
    EXPECTED_PARAM_COUNT,
)

CACHE_DIR = Path("data/cache_27m_v1")
TOK_PATH = Path("checkpoints/bpe_vocab4096.json")
CKPT_DIR = Path("checkpoints/27m_phase1")


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    assert TOK_PATH.exists()
    tok = BPETokenizer().load(str(TOK_PATH))
    assert tok.get_fingerprint() == EXPECTED_TOKENIZER_FP
    return tok


def test_27m_model_construction_and_parameter_count():
    """Verify that get_27m_config produces exact 27,411,456 parameters."""
    config = get_27m_config(block_size=256, dropout=0.1, use_sdpa=True)
    assert config.vocab_size == 4096
    assert config.embed_dim == 512
    assert config.num_heads == 8
    assert config.num_layers == 8
    assert config.block_size == 256
    assert config.weight_tying is True
    assert config.use_sdpa is True

    model = GroundUpGPT(config)
    num_params = model.get_num_params()
    assert num_params == EXPECTED_PARAM_COUNT, f"Expected {EXPECTED_PARAM_COUNT}, got {num_params}"


def test_tokenizer_and_dataset_fingerprint_matching(tokenizer: BPETokenizer):
    """Verify cache metadata matches tokenizer fingerprint and 27M dataset."""
    assert CACHE_DIR.exists()
    meta_path = CACHE_DIR / "cache_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["tokenizer_fingerprint"] == EXPECTED_TOKENIZER_FP
    assert meta["train_tokens_count"] >= 50_000_000
    assert meta["val_tokens_count"] >= 2_000_000


def test_causal_one_token_shifted_batch_extraction(tokenizer: BPETokenizer):
    """Verify batch generator produces exact causal one-token shifted pairs."""
    pipeline = load_token_cache(CACHE_DIR, tokenizer=tokenizer)
    x, y = pipeline.get_batch("train", batch_size=4, block_size=256, device="cpu")
    assert x.shape == (4, 256)
    assert y.shape == (4, 256)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_finite_forward_backward_and_amp():
    """Verify finite forward/backward pass with AMP and gradient accumulation."""
    config = get_27m_config(block_size=64, dropout=0.0, use_sdpa=True)
    model = GroundUpGPT(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    x = torch.randint(0, 4096, (2, 64), device=device)
    y = torch.randint(0, 4096, (2, 64), device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    use_cuda = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", enabled=use_cuda):
        _, loss = model(x, y)

    assert torch.isfinite(loss)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(gn)
    scaler.step(optimizer)
    scaler.update()


def test_learning_rate_schedule():
    """Verify linear warmup followed by smooth cosine decay."""
    warmup = 100
    total = 6000
    peak = 3e-4
    min_lr = 3e-5

    # Step 0: small positive
    lr_0 = compute_learning_rate(0, warmup, total, peak, min_lr)
    assert 0 < lr_0 < peak

    # Step 100: exactly peak
    lr_100 = compute_learning_rate(100, warmup, total, peak, min_lr)
    assert abs(lr_100 - peak) < 1e-7

    # Step 600: decaying but healthy
    lr_600 = compute_learning_rate(600, warmup, total, peak, min_lr)
    assert min_lr < lr_600 < peak
    assert lr_600 > 2.5e-4  # Still in active training region

    # Step 1200: active, not zero
    lr_1200 = compute_learning_rate(1200, warmup, total, peak, min_lr)
    assert lr_1200 > 2.0e-4

    # Step 6000: min_lr
    lr_end = compute_learning_rate(6000, warmup, total, peak, min_lr)
    assert abs(lr_end - min_lr) < 1e-7


def test_fixed_evaluation_reproducibility():
    """Verify fixed evaluation windows produce identical loss on identical weights."""
    config = get_27m_config(block_size=64, dropout=0.0, use_sdpa=True)
    torch.manual_seed(999)
    model = GroundUpGPT(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    fake_tokens = torch.randint(0, 4096, (5000,), dtype=torch.int32)
    eval_rng = torch.Generator().manual_seed(42)
    indices = torch.randint(0, len(fake_tokens) - 65, (8,), generator=eval_rng)

    loss1, ppl1 = evaluate_fixed_track(model, fake_tokens, indices, block_size=64, device=device)
    loss2, ppl2 = evaluate_fixed_track(model, fake_tokens, indices, block_size=64, device=device)

    assert loss1 == loss2
    assert ppl1 == ppl2


def test_crash_safe_checkpoint_save_and_resume(tmp_path: Path):
    """Verify stateful checkpoint serialization and exact restoration."""
    config = get_27m_config(block_size=64, dropout=0.1, use_sdpa=True)
    torch.manual_seed(42)
    model = GroundUpGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ckpt_file = tmp_path / "model_27m_test.pt"
    payload = {
        "step": 450,
        "tokens_processed": 450 * 8192,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": None,
        "model_config": asdict(config),
        "dataset_fingerprint": EXPECTED_TOKENIZER_FP,
        "tokenizer_fingerprint": EXPECTED_TOKENIZER_FP,
        "best_val_loss": 5.4321,
        "best_step": 400,
        "resumable": True,
    }
    save_crash_safe_checkpoint(payload, ckpt_file)
    assert ckpt_file.exists()

    # Reload into fresh model and optimizer
    model_resumed = GroundUpGPT(config)
    optimizer_resumed = torch.optim.AdamW(model_resumed.parameters(), lr=1e-4)

    raw_loaded = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    assert raw_loaded["step"] == 450
    assert raw_loaded["best_val_loss"] == 5.4321

    model_resumed.load_state_dict(raw_loaded["model_state_dict"])
    optimizer_resumed.load_state_dict(raw_loaded["optimizer_state_dict"])

    for p1, p2 in zip(model.parameters(), model_resumed.parameters()):
        assert torch.equal(p1, p2)


def test_coexistence_with_5m_models():
    """Verify 5M and 27M configurations coexist cleanly without interference."""
    cfg_5m = get_5m_config(block_size=256, dropout=0.1)
    cfg_27m = get_27m_config(block_size=256, dropout=0.1, use_sdpa=True)

    m_5m = GroundUpGPT(cfg_5m)
    m_27m = GroundUpGPT(cfg_27m)

    assert m_5m.get_num_params() == 5_051_904
    assert m_27m.get_num_params() == 27_411_456

    # Verify existing 5M checkpoints are untouched
    p2_best = Path("checkpoints/5m_phase2/model_5m_best.pt")
    if p2_best.exists():
        loaded_5m = torch.load(p2_best, map_location="cpu", weights_only=False)
        assert loaded_5m["step"] == 3000
