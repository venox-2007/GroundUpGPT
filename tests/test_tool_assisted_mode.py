"""End-to-end integration and telemetry tests for GroundUpGPT tool-assisted mode.

Verifies:
- Scenarios A through J covering native vs assisted behavior.
- Zero neural model calls and zero GPU tokens for deterministic tools.
- Strict isolation between tool execution and model generation.
- Model profile switching invariance for deterministic tools.
"""

import json
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.model_loader import model_manager
from src.model import GPTConfig, GroundUpGPT
from src.bpe_tokenizer import BPETokenizer


@pytest.fixture(scope="module")
def app_client(tmp_path_factory) -> TestClient:
    """Setup app with test checkpoints for profile switching."""
    tmp_dir = tmp_path_factory.mktemp("tool_assisted_test")
    tok_path = tmp_dir / "test_tok.json"
    ckpt1_path = tmp_dir / "ckpt_profile1.pt"
    ckpt2_path = tmp_dir / "ckpt_profile2.pt"

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
    state = model.state_dict()

    ckpt1 = {
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
        "model_state_dict": state,
        "is_sft": True,
        "profile_name": "profile_one",
    }
    ckpt2 = dict(ckpt1)
    ckpt2["profile_name"] = "profile_two"
    ckpt2["step"] = 200

    import torch
    torch.save(ckpt1, ckpt1_path)
    torch.save(ckpt2, ckpt2_path)

    model_manager.load_model(
        checkpoint_path=ckpt1_path,
        tokenizer_path=tok_path,
        device_name="cpu",
        model_name="test_model_1",
        active_profile_key="profile_one",
    )

    client = TestClient(app)
    return client


def test_scenario_a_native_mode(app_client: TestClient):
    """Scenario A: Native mode 'What is 7 + 9?' must produce neural generation, not tool output."""
    res = app_client.post(
        "/api/assistant/respond",
        json={"prompt": "What is 7 + 9?", "mode": "native", "max_new_tokens": 10},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["origin"] == "neural"
    assert data["model_generated"] is True
    assert data["tool_name"] is None
    assert data["generated_tokens"] > 0


def test_scenario_b_assisted_mode_math(app_client: TestClient):
    """Scenario B: Assisted mode 'What is 7 + 9?' returns exact tool calculation '16'."""
    res = app_client.post(
        "/api/assistant/respond",
        json={"prompt": "What is 7 + 9?", "mode": "assisted"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["origin"] == "safe_calculator"
    assert data["answer_text"] == "16"
    assert data["tool_name"] == "SafeCalculator"
    assert data["evaluated_expression"] == "7 + 9"
    assert data["model_generated"] is False


def test_scenario_c_assisted_mode_japan_capital(app_client: TestClient):
    """Scenario C: Assisted mode 'What is the capital of Japan?' returns 'Tokyo'."""
    res = app_client.post(
        "/api/assistant/respond",
        json={"prompt": "What is the capital of Japan?", "mode": "assisted"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["origin"] == "local_fact"
    assert data["answer_text"] == "Tokyo"
    assert data["fact_id"] == "geo_japan_capital"
    assert data["provenance"] == "curated_groundup_facts_v1"
    assert data["model_generated"] is False


def test_scenario_d_assisted_unsupported_country(app_client: TestClient):
    """Scenario D: Assisted mode 'What is the capital of an unsupported country?' returns limitation."""
    res = app_client.post(
        "/api/assistant/respond",
        json={"prompt": "What is the capital of an unsupported country?", "mode": "assisted"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["origin"] == "local_fact"
    assert data["fact_id"] is None
    assert data["error_code"] == "unsupported_fact"
    assert "supported countries are" in data["answer_text"].lower()


def test_scenario_e_assisted_hello_neural(app_client: TestClient):
    """Scenario E: Assisted mode 'Hello!' routes to neural generation."""
    res = app_client.post(
        "/api/assistant/respond",
        json={"prompt": "Hello!", "mode": "assisted", "max_new_tokens": 10},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["origin"] == "neural"
    assert data["model_generated"] is True
    assert data["generated_tokens"] > 0


def test_scenario_f_assisted_story_streaming(app_client: TestClient):
    """Scenario F: Assisted mode 'Tell me a short story.' streams neural tokens."""
    res = app_client.post(
        "/api/assistant/stream",
        json={"prompt": "Tell me a short story.", "mode": "assisted", "max_new_tokens": 5},
    )
    assert res.status_code == 200
    assert "event: token" in res.text
    assert "event: done" in res.text


def test_scenario_g_assisted_division_by_zero(app_client: TestClient):
    """Scenario G: Assisted mode 'What is 10 / 0?' returns explicit division-by-zero error."""
    res = app_client.post(
        "/api/assistant/respond",
        json={"prompt": "What is 10 / 0?", "mode": "assisted"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["origin"] == "tool_error"
    assert data["error_code"] == "division_by_zero"
    assert "division by zero is undefined" in data["answer_text"].lower()


def test_scenario_h_assisted_fictional_fact_protection(app_client: TestClient):
    """Scenario H: Assisted mode 'What is the capital of Eldoria?' does NOT return Quellix."""
    res = app_client.post(
        "/api/assistant/respond",
        json={"prompt": "What is the capital of Eldoria?", "mode": "assisted"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["origin"] == "local_fact"
    assert data["fact_id"] is None
    assert "Quellix" not in data["answer_text"]


def test_scenario_i_assisted_relation_mismatch(app_client: TestClient):
    """Scenario I: Assisted mode 'What is the population of Japan?' does NOT return Tokyo."""
    res = app_client.post(
        "/api/assistant/respond",
        json={"prompt": "What is the population of Japan?", "mode": "assisted"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["origin"] == "local_fact"
    assert data["fact_id"] is None
    assert data["answer_text"] != "Tokyo"


def test_zero_model_calls_for_tools(app_client: TestClient):
    """Section 17: Verify zero model calls and zero forward passes for deterministic tools."""
    # Reset telemetry counters
    model_manager.reset_telemetry_counters()
    initial_counters = model_manager.get_telemetry_counters()
    assert initial_counters["inference_call_count"] == 0
    assert initial_counters["forward_pass_count"] == 0
    assert initial_counters["tokens_generated_count"] == 0

    # Execute Calculator query via respond
    res_math = app_client.post(
        "/api/assistant/respond",
        json={"prompt": "What is 15 * 6?", "mode": "assisted"},
    )
    assert res_math.status_code == 200
    assert res_math.json()["answer_text"] == "90"
    math_counters = model_manager.get_telemetry_counters()
    assert math_counters["inference_call_count"] == 0, "Inference calls must equal 0 for calculator"
    assert math_counters["forward_pass_count"] == 0, "Forward passes must equal 0 for calculator"
    assert math_counters["tokens_generated_count"] == 0, "GPU/model tokens must equal 0 for calculator"

    # Execute Local Fact query via respond
    res_fact = app_client.post(
        "/api/assistant/respond",
        json={"prompt": "What is the capital of France?", "mode": "assisted"},
    )
    assert res_fact.status_code == 200
    assert res_fact.json()["answer_text"] == "Paris"
    fact_counters = model_manager.get_telemetry_counters()
    assert fact_counters["inference_call_count"] == 0, "Inference calls must equal 0 for local fact"
    assert fact_counters["forward_pass_count"] == 0, "Forward passes must equal 0 for local fact"
    assert fact_counters["tokens_generated_count"] == 0, "Model tokens must equal 0 for local fact"

    # Execute Streaming Tool query
    res_stream = app_client.post(
        "/api/assistant/stream",
        json={"prompt": "Calculate (20 + 5) * 2", "mode": "assisted"},
    )
    assert res_stream.status_code == 200
    assert "event: tool_result" in res_stream.text
    stream_counters = model_manager.get_telemetry_counters()
    assert stream_counters["inference_call_count"] == 0, "Streaming tools must have 0 inference calls"
    assert stream_counters["forward_pass_count"] == 0, "Streaming tools must have 0 forward passes"
    assert stream_counters["tokens_generated_count"] == 0, "Streaming tools must have 0 generated tokens"

    # Now execute Neural request and verify telemetry increments
    res_neural = app_client.post(
        "/api/assistant/respond",
        json={"prompt": "Hello!", "mode": "assisted", "max_new_tokens": 5},
    )
    assert res_neural.status_code == 200
    neural_counters = model_manager.get_telemetry_counters()
    assert neural_counters["inference_call_count"] == 1
    assert neural_counters["tokens_generated_count"] > 0
    assert neural_counters["forward_pass_count"] > 0


def test_scenario_j_profile_switch_tool_invariance(app_client: TestClient):
    """Scenario J: Switching model profiles does not affect deterministic tool results."""
    # Calculator with current profile
    res1 = app_client.post("/api/assistant/respond", json={"prompt": "What is 40 + 2?", "mode": "assisted"})
    assert res1.json()["answer_text"] == "42"

    # Simulate profile switch
    curr_key = model_manager.active_profile_key
    model_manager.active_profile_key = "profile_two"
    try:
        # Calculator still yields 42
        res2 = app_client.post("/api/assistant/respond", json={"prompt": "What is 40 + 2?", "mode": "assisted"})
        assert res2.json()["answer_text"] == "42"

        # Local fact still yields Tokyo
        res3 = app_client.post("/api/assistant/respond", json={"prompt": "What is the capital of Japan?", "mode": "assisted"})
        assert res3.json()["answer_text"] == "Tokyo"
    finally:
        model_manager.active_profile_key = curr_key
