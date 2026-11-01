"""Comprehensive test suite for Step 38: GroundUpGPT-27M SFT Web UI and Inference Integration.

Verifies:
1. Trusted profile registration for 27m_sft_best and 27m_sft_latest.
2. Available profile discovery and missing-checkpoint filtering.
3. Strict 27M checkpoint loading and exact parameter count (27,411,456).
4. Accurate SFT, source pretraining, and validation metadata parsing.
5. Backward compatibility with 5M SFT and pretrained models.
6. Multi-token BPE delimiter encoding and prompt format alignment.
7. Streaming and non-streaming parity with deterministic seeds.
8. Runtime model switching between 5M and 27M with failure rollback.
9. Context-length validation and overlong prompt rejection (256-token limit).
10. Delimiter residue stripping and end-of-turn stop sequence termination.
11. Browser-facing FastAPI endpoints (/api/model/info, /api/model/profiles, /api/model/load, /api/generate).
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app import app
from backend.config import (
    MODEL_PROFILES,
    CANONICAL_PROFILE_ORDER,
    get_available_profiles,
    get_trusted_profile,
    resolve_default_profile,
)
from backend.model_loader import ModelManager, model_manager
from src.bpe_tokenizer import BPETokenizer
from src.generation import (
    GenerationConfig,
    TextGenerator,
    format_sft_prompt,
    generate_sft_response,
)
from src.model import GPTConfig, GroundUpGPT

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_27M_BEST = REPO_ROOT / "checkpoints" / "27m_sft_phase1" / "model_27m_sft_best.pt"
CKPT_27M_LATEST = REPO_ROOT / "checkpoints" / "27m_sft_phase1" / "model_27m_sft_latest.pt"
CKPT_5M_BEST = REPO_ROOT / "checkpoints" / "5m_sft" / "model_5m_sft_best.pt"
TOKENIZER_JSON = REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"


def test_27m_trusted_profile_registration():
    """Verify 27m_sft_best and 27m_sft_latest are registered in trusted profiles."""
    assert "27m_sft_best" in MODEL_PROFILES
    assert "27m_sft_latest" in MODEL_PROFILES

    prof_best = get_trusted_profile("27m_sft_best")
    assert prof_best is not None
    assert prof_best["model_name"] == "GroundUpGPT-27M SFT Best (Experimental Chat)"
    assert prof_best["is_sft"] is True
    assert Path(prof_best["checkpoint_path"]).name == "model_27m_sft_best.pt"

    prof_latest = get_trusted_profile("27m_sft_latest")
    assert prof_latest is not None
    assert prof_latest["model_name"] == "GroundUpGPT-27M SFT Latest (Experimental Chat)"
    assert prof_latest["is_sft"] is True
    assert Path(prof_latest["checkpoint_path"]).name == "model_27m_sft_latest.pt"

    # Canonical order begins with 27M SFT profiles
    assert CANONICAL_PROFILE_ORDER[0] == "27m_sft_best"
    assert CANONICAL_PROFILE_ORDER[1] == "27m_sft_latest"

    # Default profile is preserved (5m_sft_best)
    assert resolve_default_profile() == "5m_sft_best"


def test_available_profiles_discovery_and_filtering(tmp_path: Path, monkeypatch):
    """Verify available profiles discovers real checkpoints and filters missing ones."""
    available = get_available_profiles()
    if CKPT_27M_BEST.is_file():
        assert "27m_sft_best" in available
    if CKPT_27M_LATEST.is_file():
        assert "27m_sft_latest" in available
    if CKPT_5M_BEST.is_file():
        assert "5m_sft_best" in available

    # Test filtering of missing profiles
    missing_prof = {
        "model_name": "Missing Model",
        "checkpoint_path": str(tmp_path / "nonexistent.pt"),
        "tokenizer_path": str(TOKENIZER_JSON),
    }
    monkeypatch.setitem(MODEL_PROFILES, "test_missing", missing_prof)
    filtered = get_available_profiles()
    assert "test_missing" not in filtered


def test_27m_checkpoint_architecture_and_parameter_count():
    """Verify 27M SFT checkpoint strictly matches expected 8L/8H/512d/256ctx architecture and 27,411,456 params."""
    assert CKPT_27M_BEST.is_file(), f"Missing 27M best checkpoint at {CKPT_27M_BEST}"
    assert TOKENIZER_JSON.is_file(), f"Missing tokenizer JSON at {TOKENIZER_JSON}"

    mgr = ModelManager()
    mgr.load_model(
        checkpoint_path=str(CKPT_27M_BEST),
        tokenizer_path=str(TOKENIZER_JSON),
        device_name="cpu",
        active_profile_key="27m_sft_best",
    )

    assert mgr.is_loaded is True
    info = mgr.model_info
    assert info is not None
    assert info.total_parameters == 27_411_456
    assert info.num_layers == 8
    assert info.num_heads == 8
    assert info.embed_dim == 512
    assert info.block_size == 256
    assert info.vocab_size == 4096
    assert info.is_sft is True
    assert info.model_mode == "chat"
    assert info.sft_step == 600
    assert info.source_step == 5949
    assert info.global_step == 6549
    assert info.sft_val_loss is not None
    assert round(info.sft_val_loss, 4) == 2.8769
    assert info.sft_val_perplexity is not None
    assert round(info.sft_val_perplexity, 2) == 17.76
    assert info.source_val_loss is not None
    assert round(info.source_val_loss, 4) == 3.5768
    assert info.tokenizer_fingerprint == "4138461786ea6a199ff232b2427f3cb71c60f351afa7241f75f8196e51bdae72"


def test_5m_profile_backward_compatibility():
    """Verify 5M SFT profile remains loadable and correctly reports 5M metadata."""
    if not CKPT_5M_BEST.is_file():
        pytest.skip("5M SFT checkpoint not present.")

    mgr = ModelManager()
    mgr.load_model(
        checkpoint_path=str(CKPT_5M_BEST),
        tokenizer_path=str(TOKENIZER_JSON),
        device_name="cpu",
        active_profile_key="5m_sft_best",
    )

    assert mgr.is_loaded is True
    info = mgr.model_info
    assert info is not None
    assert info.total_parameters == 5_051_904
    assert info.num_layers == 5
    assert info.num_heads == 4
    assert info.embed_dim == 256
    assert info.sft_step == 650
    assert info.source_step == 3000
    assert info.global_step == 3650
    assert info.is_sft is True


def test_delimiter_bpe_encoding_and_prompt_alignment():
    """Verify that conversation delimiters encode to standard multi-token BPE sequences and match training."""
    tok = BPETokenizer.load(TOKENIZER_JSON)

    user_prefix = "<|user|>\n"
    asst_prefix = "<|assistant|>\n"
    eot_delim = "<|endoftext|>\n"

    u_enc = tok.encode(user_prefix)
    a_enc = tok.encode(asst_prefix)
    e_enc = tok.encode(eot_delim)

    # Delimiters are verified multi-token sequences
    assert len(u_enc) == 7
    assert len(a_enc) == 7
    assert len(e_enc) == 9

    # Verify lossless decode roundtrip
    assert tok.decode(u_enc) == user_prefix
    assert tok.decode(a_enc) == asst_prefix
    assert tok.decode(e_enc) == eot_delim

    # Verify format_sft_prompt produces exact prefix sequence
    p_text = "What is the capital of Japan?"
    formatted = format_sft_prompt(p_text)
    assert formatted.startswith(user_prefix)
    assert formatted.endswith(asst_prefix)
    assert p_text in formatted

    p_encoded = tok.encode(formatted)
    assert p_encoded[:len(u_enc)] == u_enc
    assert p_encoded[-len(a_enc):] == a_enc


def test_streaming_and_non_streaming_parity():
    """Verify non-streaming generate() and streaming generate_stream() yield equivalent token sequences with fixed seed."""
    assert CKPT_27M_BEST.is_file()

    mgr = ModelManager()
    mgr.load_model(
        checkpoint_path=str(CKPT_27M_BEST),
        tokenizer_path=str(TOKENIZER_JSON),
        device_name="cpu",
        active_profile_key="27m_sft_best",
    )
    generator = mgr.generator
    assert generator is not None

    test_prompt = "Hello! How are you today?"
    cfg = GenerationConfig(
        max_new_tokens=15,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        seed=1337,
        stop_sequences=["<|endoftext|>", "<|user|>", "<|assistant|>"],
    )

    # 1. Non-streaming
    res_non_stream = generate_sft_response(generator, prompt=test_prompt, config=cfg)

    # 2. Streaming (with identical seed)
    formatted = format_sft_prompt(test_prompt)
    stream_chunks = []
    for chunk in generator.generate_stream(prompt=formatted, config=cfg):
        stream_chunks.append(chunk.text_chunk)
    stream_raw = "".join(stream_chunks)

    # Clean delimiters to compare assistant content
    for delim in ["<|endoftext|>", "<|user|>", "<|assistant|>"]:
        if delim in stream_raw:
            stream_raw = stream_raw.split(delim)[0]

    assert res_non_stream.generated_text.strip() == stream_raw.strip()


def test_runtime_model_switching_and_failure_rollback(tmp_path: Path):
    """Verify model switching from 5M to 27M and back, and verify rollback on corrupt checkpoint."""
    if not (CKPT_5M_BEST.is_file() and CKPT_27M_BEST.is_file()):
        pytest.skip("Both 5M and 27M checkpoints required.")

    mgr = ModelManager()

    # 1. Load 5M
    mgr.load_model(
        checkpoint_path=str(CKPT_5M_BEST),
        tokenizer_path=str(TOKENIZER_JSON),
        device_name="cpu",
        active_profile_key="5m_sft_best",
    )
    assert mgr.model_info.total_parameters == 5_051_904
    assert mgr.active_profile_key == "5m_sft_best"

    # 2. Switch to 27M
    mgr.load_model(
        checkpoint_path=str(CKPT_27M_BEST),
        tokenizer_path=str(TOKENIZER_JSON),
        device_name="cpu",
        active_profile_key="27m_sft_best",
    )
    assert mgr.model_info.total_parameters == 27_411_456
    assert mgr.active_profile_key == "27m_sft_best"

    # 3. Attempt to load corrupt/invalid checkpoint
    bad_ckpt = tmp_path / "corrupt_ckpt.pt"
    bad_ckpt.write_bytes(b"corrupt non-torch bytes")

    with pytest.raises(Exception):
        mgr.load_model(
            checkpoint_path=str(bad_ckpt),
            tokenizer_path=str(TOKENIZER_JSON),
            device_name="cpu",
            active_profile_key="corrupt",
        )

    # Working 27M model must remain intact after failed load
    assert mgr.is_loaded is True
    assert mgr.model_info.total_parameters == 27_411_456
    assert mgr.active_profile_key == "27m_sft_best"

    # 4. Switch back to 5M
    mgr.load_model(
        checkpoint_path=str(CKPT_5M_BEST),
        tokenizer_path=str(TOKENIZER_JSON),
        device_name="cpu",
        active_profile_key="5m_sft_best",
    )
    assert mgr.model_info.total_parameters == 5_051_904
    assert mgr.active_profile_key == "5m_sft_best"


def test_context_length_validation_rejects_overlong_prompts():
    """Verify that a prompt exceeding the 256-token context limit returns HTTP 400 Bad Request."""
    client = TestClient(app)

    # Ensure 27M model is loaded
    res_load = client.post("/api/model/load", json={"profile": "27m_sft_best"})
    assert res_load.status_code == 200

    # Create a prompt that exceeds 256 tokens when BPE encoded
    huge_prompt = "word " * 300
    res_gen = client.post("/api/generate", json={"prompt": huge_prompt, "max_new_tokens": 10})
    assert res_gen.status_code == 400
    err_json = res_gen.json()
    assert "exceeds or equals the model's maximum context window" in err_json["detail"]

    # Streaming endpoint also rejects overlong prompt with HTTP 400
    res_stream = client.post("/api/generate/stream", json={"prompt": huge_prompt, "max_new_tokens": 10})
    assert res_stream.status_code == 400
    assert "exceeds or equals the model's maximum context window" in res_stream.json()["detail"]


def test_fastapi_endpoints_with_27m_sft():
    """Verify /api/health, /api/model/info, /api/model/profiles, and /api/generate with 27M model."""
    client = TestClient(app)

    # 1. Switch to 27M
    res_load = client.post("/api/model/load", json={"profile": "27m_sft_best"})
    assert res_load.status_code == 200

    # 2. GET /api/health
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    assert res_health.json()["model_loaded"] is True

    # 3. GET /api/model/info
    res_info = client.get("/api/model/info")
    assert res_info.status_code == 200
    info = res_info.json()
    assert info["total_parameters"] == 27_411_456
    assert info["active_profile"] == "27m_sft_best"
    assert info["is_sft"] is True
    assert info["model_mode"] == "chat"
    assert info["sft_step"] == 600
    assert info["source_step"] == 5949
    assert info["source_val_loss"] == 3.5768
    assert round(info["sft_val_loss"], 4) == 2.8769

    # 4. GET /api/model/profiles
    res_profs = client.get("/api/model/profiles")
    assert res_profs.status_code == 200
    profs = res_profs.json()
    assert profs["active_profile"] == "27m_sft_best"
    assert "27m_sft_best" in profs["profiles"]
    assert profs["profiles"]["27m_sft_best"]["is_sft"] is True

    # 5. POST /api/generate with normal conversational prompt
    res_gen = client.post(
        "/api/generate",
        json={
            "prompt": "Hello! How are you today?",
            "max_new_tokens": 20,
            "temperature": 0.8,
            "top_k": 40,
            "top_p": 0.9,
            "seed": 42,
        },
    )
    assert res_gen.status_code == 200
    gen_data = res_gen.json()
    assert len(gen_data["generated_text"]) > 0
    assert "<|user|>" not in gen_data["generated_text"]
    assert "<|assistant|>" not in gen_data["generated_text"]
    assert "<|endoftext|>" not in gen_data["generated_text"]
    assert gen_data["prompt_tokens"] > 0
    assert gen_data["generated_tokens"] > 0

    # 6. Switch back to 5M SFT default
    if CKPT_5M_BEST.is_file():
        res_back = client.post("/api/model/load", json={"profile": "5m_sft_best"})
        assert res_back.status_code == 200
        assert res_back.json()["total_parameters"] == 5_051_904
