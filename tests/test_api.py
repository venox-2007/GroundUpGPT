"""Targeted unit and integration tests for the GroundUpGPT FastAPI inference backend."""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import torch

from src.model import GPTConfig, GroundUpGPT
from src.bpe_tokenizer import BPETokenizer
from backend.app import app
from backend.model_loader import ModelManager, model_manager
from backend.schemas import GenerateRequest


@pytest.fixture(scope="module")
def initialized_client(tmp_path_factory) -> TestClient:
    """Provide a TestClient with a lightweight model loaded once."""
    tmp_dir = tmp_path_factory.mktemp("api_test_artifacts")
    tok_path = tmp_dir / "test_tok.json"
    ckpt_path = tmp_dir / "test_ckpt.pt"

    # 1. Create and save test BPE tokenizer
    corpus = "Hello world! GroundUpGPT is a decoder-only language model built from scratch. 1234567890."
    tok = BPETokenizer().fit(corpus, target_vocab_size=275)
    tok.save(tok_path)

    # 2. Create small model and save compatible checkpoint
    m_cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=16,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        bias=False,
        weight_tying=True,
    )
    model = GroundUpGPT(m_cfg)
    checkpoint = {
        "step": 10,
        "val_loss": 3.45,
        "model_config": {
            "vocab_size": m_cfg.vocab_size,
            "block_size": m_cfg.block_size,
            "embed_dim": m_cfg.embed_dim,
            "num_heads": m_cfg.num_heads,
            "num_layers": m_cfg.num_layers,
            "dropout": m_cfg.dropout,
            "bias": m_cfg.bias,
            "weight_tying": m_cfg.weight_tying,
            "init_std": m_cfg.init_std,
        },
        "model_state_dict": model.state_dict(),
        "tokenizer_type": "BPETokenizer",
        "tokenizer_vocab_size": tok.vocab_size,
        "tokenizer_fingerprint": tok.get_fingerprint(),
    }
    torch.save(checkpoint, ckpt_path)

    # 3. Load model into model_manager
    model_manager.load_model(
        checkpoint_path=ckpt_path,
        tokenizer_path=tok_path,
        device_name="cpu",
    )

    with TestClient(app) as client:
        yield client


def test_health_endpoint(initialized_client: TestClient) -> None:
    """Verify /api/health reports healthy status and model load state."""
    response = initialized_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["device"] == "cpu"
    assert "version" in data


def test_model_info_endpoint(initialized_client: TestClient) -> None:
    """Verify /api/model/info returns complete architectural metadata."""
    response = initialized_client.get("/api/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "GroundUpGPT"
    assert data["block_size"] == 16
    assert data["embed_dim"] == 32
    assert data["num_heads"] == 4
    assert data["num_layers"] == 2
    assert data["total_parameters"] > 0
    assert data["tokenizer_type"] == "BPETokenizer"
    assert data["checkpoint_step"] == 10
    assert data["checkpoint_val_loss"] == 3.45


def test_model_loaded_once_behavior(initialized_client: TestClient) -> None:
    """Verify inference calls reuse the single loaded model instance without reloading."""
    original_model_id = id(model_manager.model)
    original_generator_id = id(model_manager.generator)

    # Execute two consecutive generation calls
    res1 = initialized_client.post("/api/generate", json={"prompt": "Hello", "max_new_tokens": 3})
    assert res1.status_code == 200
    res2 = initialized_client.post("/api/generate", json={"prompt": "World", "max_new_tokens": 3})
    assert res2.status_code == 200

    # Ensure memory addresses of loaded model and generator have not changed
    assert id(model_manager.model) == original_model_id
    assert id(model_manager.generator) == original_generator_id


def test_non_streaming_generation(initialized_client: TestClient) -> None:
    """Verify /api/generate produces valid response structure and metrics."""
    req_payload = {
        "prompt": "Hello ",
        "max_new_tokens": 8,
        "temperature": 0.0,
    }
    response = initialized_client.post("/api/generate", json=req_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prompt"] == "Hello "
    assert isinstance(data["generated_text"], str)
    assert data["full_text"].startswith("Hello ")
    assert data["generated_tokens"] == 8
    assert data["prompt_tokens"] > 0
    assert data["total_tokens"] == data["prompt_tokens"] + 8
    assert data["active_context_tokens"] <= 16
    assert data["context_limit"] == 16
    assert data["elapsed_time_sec"] >= 0.0
    assert data["tokens_per_second"] >= 0.0


def test_context_and_token_usage_clamping(initialized_client: TestClient) -> None:
    """Verify active_context_tokens is clamped to block_size and context_cropped is True on overflow."""
    # Context limit is 16; generating 20 tokens will cause total > 16
    req_payload = {
        "prompt": "GroundUpGPT is an experimental ",
        "max_new_tokens": 20,
        "temperature": 0.0,
    }
    response = initialized_client.post("/api/generate", json=req_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_tokens"] > 16
    assert data["active_context_tokens"] == 16
    assert data["context_limit"] == 16
    assert data["context_cropped"] is True


def test_streaming_sse_framing_and_termination(initialized_client: TestClient) -> None:
    """Verify /api/generate/stream formats SSE events (token and done) correctly."""
    req_payload = {
        "prompt": "Hello ",
        "max_new_tokens": 5,
        "temperature": 0.0,
    }
    response = initialized_client.post("/api/generate/stream", json=req_payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    lines = response.text.strip().split("\n\n")
    events = []
    for block in lines:
        if not block.strip():
            continue
        parts = block.strip().split("\n")
        event_name = parts[0].replace("event: ", "").strip()
        data_json = json.loads(parts[1].replace("data: ", "").strip())
        events.append((event_name, data_json))

    # Assert token events were streamed followed by done event
    token_events = [e for e in events if e[0] == "token"]
    done_events = [e for e in events if e[0] == "done"]

    assert len(token_events) == 5
    for i, (_, payload) in enumerate(token_events):
        assert "token_id" in payload
        assert "text" in payload
        assert payload["step"] == i
        assert payload["generated_tokens"] == i + 1

    assert len(done_events) == 1
    done_payload = done_events[0][1]
    assert done_payload["generated_tokens"] == 5
    assert done_payload["prompt_tokens"] > 0
    assert "tokens_per_second" in done_payload


def test_invalid_request_validation(initialized_client: TestClient) -> None:
    """Verify Pydantic rejects invalid parameters with 422 Unprocessable Entity."""
    # 1. Empty prompt
    res1 = initialized_client.post("/api/generate", json={"prompt": "", "max_new_tokens": 5})
    assert res1.status_code == 422

    # 2. Negative temperature
    res2 = initialized_client.post("/api/generate", json={"prompt": "Hi", "temperature": -1.0})
    assert res2.status_code == 422

    # 3. Excessive max_new_tokens (> 512)
    res3 = initialized_client.post("/api/generate", json={"prompt": "Hi", "max_new_tokens": 1000})
    assert res3.status_code == 422

    # 4. Zero max_new_tokens (< 1)
    res4 = initialized_client.post("/api/generate", json={"prompt": "Hi", "max_new_tokens": 0})
    assert res4.status_code == 422


def test_unloaded_model_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify 503 Service Unavailable when model is not loaded."""
    monkeypatch.setenv("GROUNDUPGPT_AUTO_LOAD", "false")

    # Fresh ModelManager that has not loaded any model
    empty_manager = ModelManager()
    assert empty_manager.is_loaded is False

    # Swap in empty manager temporarily
    from backend import app as app_module
    original = app_module.model_manager
    try:
        app_module.model_manager = empty_manager
        with TestClient(app) as client:
            res_info = client.get("/api/model/info")
            assert res_info.status_code == 503

            res_gen = client.post("/api/generate", json={"prompt": "Hello", "max_new_tokens": 5})
            assert res_gen.status_code == 503

            res_stream = client.post("/api/generate/stream", json={"prompt": "Hello", "max_new_tokens": 5})
            assert res_stream.status_code == 503
    finally:
        app_module.model_manager = original


def test_missing_and_incompatible_artifact_rejection(tmp_path: Path) -> None:
    """Verify loader raises FileNotFoundError or ValueError for missing/incompatible files."""
    mgr = ModelManager()

    # 1. Nonexistent checkpoint
    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        mgr.load_model(tmp_path / "missing.pt", tmp_path / "missing.json")

    # 2. Tokenizer fingerprint mismatch
    tok = BPETokenizer().fit("abc", target_vocab_size=260)
    tok_file = tmp_path / "tok.json"
    tok.save(tok_file)

    bad_ckpt = tmp_path / "bad.pt"
    m_cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=8, embed_dim=16, num_heads=2, num_layers=1)
    m = GroundUpGPT(m_cfg)
    torch.save(
        {
            "model_config": {
                "vocab_size": m_cfg.vocab_size,
                "block_size": m_cfg.block_size,
                "embed_dim": m_cfg.embed_dim,
                "num_heads": m_cfg.num_heads,
                "num_layers": m_cfg.num_layers,
            },
            "model_state_dict": m.state_dict(),
            "tokenizer_type": "BPETokenizer",
            "tokenizer_vocab_size": tok.vocab_size,
            "tokenizer_fingerprint": "mismatched_fake_fingerprint",
        },
        bad_ckpt,
    )

    with pytest.raises(ValueError, match="Tokenizer fingerprint mismatch"):
        mgr.load_model(bad_ckpt, tok_file)
