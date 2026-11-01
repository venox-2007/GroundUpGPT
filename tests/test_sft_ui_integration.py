"""Targeted test suite for Step 28: SFT Model Web UI Integration.

Verifies:
1. SFT checkpoint loading, metadata parsing, and tokenizer backward compatibility.
2. Canonical profile resolution and default SFT profile selection.
3. Conversational prompt formatting.
4. Delimiter prefix buffering and zero-leakage streaming.
5. Exact token-weighted gradient accumulation behavior.
6. FastAPI endpoints (/api/model/info, /api/model/profiles, /api/generate, /api/generate/stream).
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

from backend.app import app
from backend.config import resolve_default_profile, get_available_profiles, get_trusted_profile
from backend.model_loader import ModelManager
from src.generation import (
    GenerationConfig,
    TextGenerator,
    find_longest_stop_prefix_len,
    format_sft_prompt,
    generate_sft_response,
)
from src.model import GPTConfig, GroundUpGPT
from src.bpe_tokenizer import BPETokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
SFT_CKPT = REPO_ROOT / "checkpoints" / "5m_sft" / "model_5m_sft_best.pt"
TOKENIZER_JSON = REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"


def test_sft_checkpoint_loading_and_metadata():
    """Verify loading real SFT checkpoint extracts all architectural & SFT metadata."""
    assert SFT_CKPT.is_file(), f"Missing SFT checkpoint at {SFT_CKPT}"
    assert TOKENIZER_JSON.is_file(), f"Missing tokenizer JSON at {TOKENIZER_JSON}"

    manager = ModelManager()
    manager.load_model(
        checkpoint_path=str(SFT_CKPT),
        tokenizer_path=str(TOKENIZER_JSON),
        device_name="cpu",
        active_profile_key="5m_sft_best",
    )

    assert manager.is_loaded is True
    assert manager.model_info is not None

    info = manager.model_info
    assert info.is_sft is True
    assert info.model_mode == "chat"
    assert info.sft_step == 650
    assert info.global_step == 3650
    assert info.source_step == 3000
    assert info.total_parameters == 5_051_904
    assert info.vocab_size == 4096
    assert info.block_size == 256
    assert info.tokenizer_type.lower() in ("bpe", "bpetokenizer")
    assert info.sft_val_loss is not None and info.sft_val_loss > 0
    assert info.sft_val_perplexity is not None and info.sft_val_perplexity > 0
    assert info.lm_val_loss is not None and info.lm_val_loss > 0
    assert info.lm_val_perplexity is not None and info.lm_val_perplexity > 0


def test_tokenizer_type_backward_compatibility(tmp_path: Path):
    """Verify model loader accepts 'bpe', 'BPETokenizer', and rejects unknown types."""
    # Create minimal model and state dict
    cfg = GPTConfig(vocab_size=280, block_size=32, embed_dim=32, num_heads=4, num_layers=2)
    model = GroundUpGPT(cfg)

    # Use existing demo tokenizer JSON
    demo_tok = REPO_ROOT / "checkpoints" / "demo_tokenizer.json"
    assert demo_tok.is_file()

    for tok_type in ["bpe", "BPETokenizer", "BPE", "bpetokenizer"]:
        ckpt_path = tmp_path / f"ckpt_{tok_type}.pt"
        torch.save(
            {
                "model_config": {
                    "vocab_size": 280,
                    "block_size": 32,
                    "embed_dim": 32,
                    "num_heads": 4,
                    "num_layers": 2,
                },
                "model_state_dict": model.state_dict(),
                "tokenizer_type": tok_type,
                "tokenizer_vocab_size": 280,
                "step": 1,
                "val_loss": 5.0,
            },
            ckpt_path,
        )

        mgr = ModelManager()
        mgr.load_model(ckpt_path, demo_tok, device_name="cpu")
        assert mgr.is_loaded is True
        assert mgr.model_info.tokenizer_type == tok_type

    # Invalid tokenizer type must raise ValueError
    bad_ckpt = tmp_path / "ckpt_bad.pt"
    torch.save(
        {
            "model_config": {
                "vocab_size": 280,
                "block_size": 32,
                "embed_dim": 32,
                "num_heads": 4,
                "num_layers": 2,
            },
            "model_state_dict": model.state_dict(),
            "tokenizer_type": "SentencePieceUnigram",
            "step": 1,
        },
        bad_ckpt,
    )
    mgr_bad = ModelManager()
    with pytest.raises(ValueError, match="Unknown tokenizer type"):
        mgr_bad.load_model(bad_ckpt, demo_tok, device_name="cpu")


def test_profile_resolution_prefers_sft_best():
    """Verify resolve_default_profile prefers 5m_sft_best when available."""
    default_prof = resolve_default_profile()
    assert default_prof == "5m_sft_best"

    available = get_available_profiles()
    assert "5m_sft_best" in available
    assert available["5m_sft_best"]["is_sft"] is True
    assert "5m_sft_latest" in available
    assert available["5m_sft_latest"]["is_sft"] is True

    trusted = get_trusted_profile("5m_sft_best")
    assert trusted is not None
    assert trusted["is_sft"] is True


def test_format_sft_prompt():
    """Verify conversational turn formatting with and without context."""
    p1 = format_sft_prompt("What is Python?")
    assert p1 == "<|user|>\nWhat is Python?\n<|assistant|>\n"

    p2 = format_sft_prompt("Summarize the text.", context="Python is a language.")
    assert p2 == "<|user|>\nSummarize the text.\n\nContext:\nPython is a language.\n<|assistant|>\n"


def test_find_longest_stop_prefix_len():
    """Verify delimiter prefix matching logic."""
    stops = ["<|endoftext|>", "<|user|>", "<|assistant|>"]

    assert find_longest_stop_prefix_len("", stops) == 0
    assert find_longest_stop_prefix_len("hello world", stops) == 0
    assert find_longest_stop_prefix_len("hello <", stops) == 1
    assert find_longest_stop_prefix_len("hello <|", stops) == 2
    assert find_longest_stop_prefix_len("hello <|user", stops) == 6
    assert find_longest_stop_prefix_len("hello <|assi", stops) == 6
    assert find_longest_stop_prefix_len("hello <3", stops) == 0
    assert find_longest_stop_prefix_len("hello <<", stops) == 1


def test_delimiter_prefix_buffering_no_leakage():
    """Verify streaming generator buffers partial stop sequences and never leaks delimiters."""
    # Build minimal mock model and tokenizer
    tokenizer = BPETokenizer.load(str(REPO_ROOT / "checkpoints" / "demo_tokenizer.json"))
    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=32,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
    )
    model = GroundUpGPT(cfg)
    model.eval()
    generator = TextGenerator(model, tokenizer, device="cpu")

    # Verify stop sequences in GenerationConfig
    config = GenerationConfig(
        max_new_tokens=20,
        stop_sequences=["<|endoftext|>", "<|user|>", "<|assistant|>"],
        seed=42,
    )

    chunks = list(generator.generate_stream(prompt="The ", config=config))
    full_streamed = "".join(c.text_chunk for c in chunks)

    # Delimiter characters should never be present in full_streamed
    for delim in config.stop_sequences:
        assert delim not in full_streamed


def test_generate_sft_response_cleanliness():
    """Verify generate_sft_response cleans delimiters and updates full_text."""
    tokenizer = BPETokenizer.load(str(REPO_ROOT / "checkpoints" / "demo_tokenizer.json"))
    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=32,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
    )
    model = GroundUpGPT(cfg)
    model.eval()
    generator = TextGenerator(model, tokenizer, device="cpu")

    user_prompt = "Hello"
    result = generate_sft_response(
        generator=generator,
        prompt=user_prompt,
        config=GenerationConfig(max_new_tokens=15, seed=42),
    )

    assert result.prompt == user_prompt
    assert result.full_text == user_prompt + result.generated_text
    for delim in ["<|endoftext|>", "<|user|>", "<|assistant|>"]:
        assert not result.generated_text.endswith(delim)


def test_exact_gradient_accumulation_weighting():
    """Verify exact token-weighted gradient accumulation is mathematically equivalent to batch sum."""
    cfg = GPTConfig(vocab_size=64, block_size=16, embed_dim=32, num_heads=2, num_layers=2, dropout=0.0)
    torch.manual_seed(42)
    model = GroundUpGPT(cfg)
    model.eval()  # ensure deterministic dropout

    # Two micro-batches with different active token masks
    b1_x = torch.randint(0, 64, (2, 8))
    b1_y = torch.randint(0, 64, (2, 8))
    b1_mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    b1_y[~b1_mask] = -100

    b2_x = torch.randint(0, 64, (2, 8))
    b2_y = torch.randint(0, 64, (2, 8))
    b2_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    b2_y[~b2_mask] = -100

    total_active = int(b1_mask.sum().item() + b2_mask.sum().item())
    assert total_active > 0

    # Path A: Combined single batch
    combined_x = torch.cat([b1_x, b2_x], dim=0)
    combined_y = torch.cat([b1_y, b2_y], dim=0)

    model.zero_grad()
    logits_comb, _ = model(combined_x)
    loss_comb = F.cross_entropy(logits_comb.view(-1, 64), combined_y.view(-1), ignore_index=-100, reduction="sum") / total_active
    loss_comb.backward()
    grads_comb = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    # Path B: Micro-batch accumulated
    model.zero_grad()
    for x, y in [(b1_x, b1_y), (b2_x, b2_y)]:
        logits, _ = model(x)
        loss_mb = F.cross_entropy(logits.view(-1, 64), y.view(-1), ignore_index=-100, reduction="sum") / total_active
        loss_mb.backward()
    grads_accum = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    for g_c, g_a in zip(grads_comb, grads_accum):
        torch.testing.assert_close(g_c, g_a, rtol=1e-5, atol=1e-5)


def test_fastapi_endpoints_with_sft():
    """Verify FastAPI endpoints return SFT metadata and handle generation properly."""
    client = TestClient(app)

    # 1. /api/health
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"

    # 2. Switch to 5m_sft_best profile
    res_load = client.post("/api/model/load", json={"profile": "5m_sft_best"})
    assert res_load.status_code == 200
    load_data = res_load.json()
    assert load_data["is_sft"] is True
    assert load_data["model_mode"] == "chat"
    assert load_data["sft_step"] == 650

    # 3. /api/model/info
    res_info = client.get("/api/model/info")
    assert res_info.status_code == 200
    info_data = res_info.json()
    assert info_data["is_sft"] is True
    assert info_data["active_profile"] == "5m_sft_best"
    assert info_data["sft_val_loss"] is not None

    # 4. /api/model/profiles
    res_profiles = client.get("/api/model/profiles")
    assert res_profiles.status_code == 200
    prof_data = res_profiles.json()
    assert "5m_sft_best" in prof_data["profiles"]
    assert prof_data["profiles"]["5m_sft_best"]["is_sft"] is True

    # 5. /api/generate
    res_gen = client.post(
        "/api/generate",
        json={"prompt": "What is Python?", "max_new_tokens": 10, "temperature": 0.0},
    )
    assert res_gen.status_code == 200
    gen_data = res_gen.json()
    assert gen_data["prompt"] == "What is Python?"
    assert isinstance(gen_data["generated_text"], str)
    for delim in ["<|endoftext|>", "<|user|>", "<|assistant|>"]:
        assert delim not in gen_data["generated_text"]

    # 6. /api/generate/stream SSE
    res_stream = client.post(
        "/api/generate/stream",
        json={"prompt": "What is Python?", "max_new_tokens": 10, "temperature": 0.0},
    )
    assert res_stream.status_code == 200
    lines = res_stream.text.strip().split("\n")
    events = [l for l in lines if l.startswith("event:")]
    assert "event: token" in events
    assert "event: done" in events
