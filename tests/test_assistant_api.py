"""Integration tests for GroundUpGPT dedicated assistant API endpoints."""

import json
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.model_loader import model_manager
from src.model import GPTConfig, GroundUpGPT
from src.bpe_tokenizer import BPETokenizer


@pytest.fixture(scope="module")
def api_client(tmp_path_factory) -> TestClient:
    """Initialize a fast lightweight model instance for API testing."""
    tmp_dir = tmp_path_factory.mktemp("assistant_api_test")
    tok_path = tmp_dir / "test_tok.json"
    ckpt_path = tmp_dir / "test_ckpt.pt"

    corpus = "Hello world! GroundUpGPT is a decoder-only language model built from scratch. 1234567890."
    tok = BPETokenizer().fit(corpus, target_vocab_size=275)
    tok.save(tok_path)

    m_cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=32,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        bias=False,
        weight_tying=True,
    )
    model = GroundUpGPT(m_cfg)
    checkpoint = {
        "step": 100,
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
        },
        "model_state_dict": model.state_dict(),
        "is_sft": True,
        "profile_name": "test_profile",
    }
    import torch
    torch.save(checkpoint, ckpt_path)

    model_manager.load_model(
        checkpoint_path=ckpt_path,
        tokenizer_path=tok_path,
        device_name="cpu",
        model_name="test_model",
        active_profile_key="test_profile",
    )

    client = TestClient(app)
    return client


def test_mode_strict_validation(api_client: TestClient):
    """Ensure invalid modes are strictly rejected with HTTP 400 or 422."""
    res_valid_native = api_client.post(
        "/api/assistant/respond",
        json={"prompt": "Hello", "mode": "native"},
    )
    assert res_valid_native.status_code == 200

    res_valid_assisted = api_client.post(
        "/api/assistant/respond",
        json={"prompt": "Hello", "mode": "assisted"},
    )
    assert res_valid_assisted.status_code == 200

    res_invalid = api_client.post(
        "/api/assistant/respond",
        json={"prompt": "Hello", "mode": "unsupported_mode"},
    )
    assert res_invalid.status_code in (400, 422)


def test_native_mode_neural_execution(api_client: TestClient):
    """In native mode, even arithmetic questions are sent to the neural model."""
    res = api_client.post(
        "/api/assistant/respond",
        json={"prompt": "What is 7 + 9?", "mode": "native", "max_new_tokens": 10},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "native"
    assert data["origin"] == "neural"
    assert data["model_generated"] is True
    assert data["prompt_tokens"] > 0
    assert data["generated_tokens"] > 0


def test_assisted_calculator_respond(api_client: TestClient):
    """In assisted mode, arithmetic returns SafeCalculator result directly."""
    res = api_client.post(
        "/api/assistant/respond",
        json={"prompt": "What is 7 + 9?", "mode": "assisted"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "assisted"
    assert data["origin"] == "safe_calculator"
    assert data["answer_text"] == "16"
    assert data["tool_name"] == "SafeCalculator"
    assert data["evaluated_expression"] == "7 + 9"
    assert data["model_generated"] is False
    assert data["prompt_tokens"] == 0
    assert data["generated_tokens"] == 0


def test_assisted_calculator_division_by_zero(api_client: TestClient):
    """In assisted mode, division by zero returns tool_error."""
    res = api_client.post(
        "/api/assistant/respond",
        json={"prompt": "What is 10 / 0?", "mode": "assisted"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "assisted"
    assert data["origin"] == "tool_error"
    assert data["tool_name"] == "SafeCalculator"
    assert data["error_code"] == "division_by_zero"
    assert "division by zero" in data["answer_text"].lower()
    assert data["model_generated"] is False


def test_assisted_local_fact_respond(api_client: TestClient):
    """In assisted mode, validated facts return exact local answer with provenance."""
    res = api_client.post(
        "/api/assistant/respond",
        json={"prompt": "What is the capital of Japan?", "mode": "assisted"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "assisted"
    assert data["origin"] == "local_fact"
    assert data["answer_text"] == "Tokyo"
    assert data["fact_id"] == "geo_japan_capital"
    assert data["provenance"] == "curated_groundup_facts_v1"
    assert data["model_generated"] is False


def test_assisted_synthetic_fact_exclusion(api_client: TestClient):
    """In assisted mode, synthetic test facts are excluded."""
    res = api_client.post(
        "/api/assistant/respond",
        json={"prompt": "What is the capital of Eldoria?", "mode": "assisted"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["origin"] == "local_fact"
    assert data["fact_id"] is None
    assert data["error_code"] == "unsupported_fact"
    assert "Quellix" not in data["answer_text"]


def test_assisted_relation_mismatch_protection(api_client: TestClient):
    """In assisted mode, relation mismatch does not return incorrect answer."""
    res = api_client.post(
        "/api/assistant/respond",
        json={"prompt": "What is the population of Japan?", "mode": "assisted"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["origin"] == "local_fact"
    assert data["fact_id"] is None
    assert data["error_code"] == "unsupported_fact"
    assert data["answer_text"] != "Tokyo"


def test_assisted_streaming_tool_result_event(api_client: TestClient):
    """SSE streaming delivers tool_result event without streaming individual fake tokens."""
    res = api_client.post(
        "/api/assistant/stream",
        json={"prompt": "What is 7 + 9?", "mode": "assisted"},
    )
    assert res.status_code == 200
    text = res.text
    assert "event: tool_result" in text
    assert "event: done" in text
    assert "event: token" not in text

    # Parse tool_result JSON block
    blocks = text.strip().split("\n\n")
    tool_block = next(b for b in blocks if "event: tool_result" in b)
    data_line = next(line for line in tool_block.split("\n") if line.startswith("data:"))
    tool_data = json.loads(data_line[5:].strip())
    assert tool_data["origin"] == "safe_calculator"
    assert tool_data["answer_text"] == "16"
    assert tool_data["model_generated"] is False


def test_assisted_streaming_neural_tokens(api_client: TestClient):
    """SSE streaming delivers token events for conversational queries."""
    res = api_client.post(
        "/api/assistant/stream",
        json={"prompt": "Hello!", "mode": "assisted", "max_new_tokens": 5},
    )
    assert res.status_code == 200
    text = res.text
    assert "event: token" in text
    assert "event: done" in text
    assert "event: tool_result" not in text


def test_backward_compatibility_endpoints(api_client: TestClient):
    """Verify all existing endpoints remain completely functional."""
    # Existing /api/generate
    res_gen = api_client.post("/api/generate", json={"prompt": "Hello", "max_new_tokens": 5})
    assert res_gen.status_code == 200

    # Existing /api/model/info
    res_info = api_client.get("/api/model/info")
    assert res_info.status_code == 200

    # Existing /api/model/profiles
    res_prof = api_client.get("/api/model/profiles")
    assert res_prof.status_code == 200
