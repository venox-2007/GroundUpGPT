"""Unit and integration test suite for model evaluation and dynamic model profile loading."""

import os
from pathlib import Path
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app import app
from backend.config import resolve_default_profile, resolve_profile_paths, Settings, MODEL_PROFILES
from backend.model_loader import model_manager
from scripts.evaluate_model import compute_loss_and_perplexity, benchmark_generation
from src.bpe_tokenizer import BPETokenizer
from src.generation import TextGenerator
from src.model import GPTConfig, GroundUpGPT

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_evaluation_loss_and_perplexity_calculation() -> None:
    """Verify compute_loss_and_perplexity calculates valid loss and exp(loss) perplexity."""
    torch.manual_seed(42)
    config = GPTConfig(
        vocab_size=64,
        block_size=16,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )
    model = GroundUpGPT(config)
    model.eval()

    # Generate synthetic random tokens
    tokens = torch.randint(0, 64, (200,), dtype=torch.long)

    stats = compute_loss_and_perplexity(
        model=model,
        tokens=tokens,
        block_size=16,
        batch_size=4,
        eval_batches=5,
        device="cpu",
    )

    assert "loss" in stats
    assert "perplexity" in stats
    assert stats["loss"] > 0
    assert stats["perplexity"] > 0
    # Cross-entropy loss for uniform vocab 64 is ln(64) ~= 4.158
    assert abs(stats["loss"] - 4.158) < 1.0
    # Perplexity should equal exp(loss) within float precision
    expected_ppl = round(torch.exp(torch.tensor(stats["loss"])).item(), 2)
    assert abs(stats["perplexity"] - expected_ppl) < 0.2


def test_evaluation_short_tokens_error() -> None:
    """Verify compute_loss_and_perplexity raises ValueError on undersized token tensor."""
    config = GPTConfig(vocab_size=32, block_size=16, embed_dim=16, num_heads=2, num_layers=1)
    model = GroundUpGPT(config)
    short_tokens = torch.tensor([1, 2, 3], dtype=torch.long)

    with pytest.raises(ValueError, match="too short for block_size"):
        compute_loss_and_perplexity(model, short_tokens, block_size=16, device="cpu")


def test_benchmark_generation_metrics() -> None:
    """Verify benchmark_generation returns non-zero throughput and latency metrics."""
    tok = BPETokenizer()
    tok.fit("The quick brown fox jumps over the lazy dog.", target_vocab_size=260)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=32, embed_dim=32, num_heads=4, num_layers=2)
    model = GroundUpGPT(cfg)
    gen = TextGenerator(model, tok, device="cpu")

    bench = benchmark_generation(gen, prompt="The quick", max_new_tokens=10, temperature=0.0, device="cpu")
    assert bench["tokens_generated"] == 10
    assert bench["elapsed_sec"] > 0
    assert bench["tokens_per_second"] > 0
    assert isinstance(bench["generated_text"], str)


def test_backend_profile_configuration_resolution(monkeypatch) -> None:
    """Verify profile resolution respects environment variables and default discovery."""
    monkeypatch.delenv("GROUNDUPGPT_MODEL_PROFILE", raising=False)
    default_prof = resolve_default_profile()
    assert default_prof in ("5m_initial", "demo")

    monkeypatch.setenv("GROUNDUPGPT_MODEL_PROFILE", "demo")
    assert resolve_default_profile() == "demo"
    demo_paths = resolve_profile_paths("demo")
    assert "demo_checkpoint.pt" in demo_paths["checkpoint_path"]
    assert "demo_tokenizer.json" in demo_paths["tokenizer_path"]

    monkeypatch.setenv("GROUNDUPGPT_MODEL_PROFILE", "5m")
    assert resolve_default_profile() == "5m"
    p5m_paths = resolve_profile_paths("5m")
    assert "model_5m_step200.pt" in p5m_paths["checkpoint_path"]
    assert "bpe_vocab4096.json" in p5m_paths["tokenizer_path"]


def test_profiles_api_endpoint() -> None:
    """Verify /api/model/profiles reports available profiles accurately."""
    client = TestClient(app)
    res = client.get("/api/model/profiles")
    assert res.status_code == 200
    data = res.json()
    assert "profiles" in data
    assert "demo" in data["profiles"]
    assert "5m_initial" in data["profiles"]
    assert "checkpoint_exists" in data["profiles"]["demo"]
    assert "checkpoint_exists" in data["profiles"]["5m_initial"]


def test_dynamic_profile_switching_and_streaming() -> None:
    """Verify switching between demo and 5M profiles updates /api/model/info and streams SSE."""
    demo_ckpt = REPO_ROOT / "checkpoints" / "demo_checkpoint.pt"
    m5_ckpt = REPO_ROOT / "checkpoints" / "5m_initial" / "model_5m_step200.pt"

    client = TestClient(app)

    # 1. Load Demo profile if artifacts exist
    if demo_ckpt.is_file():
        load_res = client.post("/api/model/load", json={"profile": "demo"})
        assert load_res.status_code == 200
        info = load_res.json()
        assert info["model_name"] == "GroundUpGPT"
        assert info["vocab_size"] == 280
        assert info["block_size"] == 32
        assert info["total_parameters"] == 34880

    # 2. Switch to 5M profile if artifacts exist
    if m5_ckpt.is_file():
        load_5m_res = client.post("/api/model/load", json={"profile": "5m"})
        assert load_5m_res.status_code == 200
        info_5m = load_5m_res.json()
        assert info_5m["model_name"] == "GroundUpGPT"
        assert info_5m["vocab_size"] == 4096
        assert info_5m["block_size"] == 256
        assert info_5m["num_layers"] == 5
        assert info_5m["total_parameters"] == 5051904
        assert info_5m["profile_name"] == "GroundUpGPT-5M (5.05M)"

        # 3. Test real-time SSE streaming with 5M model
        stream_res = client.post(
            "/api/generate/stream",
            json={"prompt": "Once upon a time", "max_new_tokens": 15, "temperature": 0.0},
        )
        assert stream_res.status_code == 200
        assert "text/event-stream" in stream_res.headers["content-type"]
        blocks = stream_res.text.strip().split("\n\n")
        assert len(blocks) >= 2
        # Final event must be 'done' event with telemetry
        assert "event: done" in blocks[-1]
        assert "generated_tokens" in blocks[-1]

    # 4. Unknown profile rejection
    bad_res = client.post("/api/model/load", json={"profile": "nonexistent_model"})
    assert bad_res.status_code == 400
    assert "Unknown profile" in bad_res.json()["detail"]
