"""Integration and regression tests for GroundUpGPT-27M anchor fine-tuned model and backend/UI APIs.

Verifies:
- Registration and canonical discovery of 27m_anchors_best and 27m_anchors_latest
- Chat mode detection and backward-compatible metadata resolution
- Pretraining, original SFT, and anchor SFT step distinction
- Safe model switching and state rollback on failure
- Conversational formatting parity between original and anchor SFT models
- FastAPI endpoints (health, profiles, info, load, generate) with TestClient
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from backend.app import app
from backend.config import get_available_profiles, get_trusted_profile, MODEL_PROFILES
from backend.model_loader import ModelManager, model_manager
from src.generation import GenerationConfig, format_sft_prompt, generate_sft_response

client = TestClient(app)


def test_backend_profile_registration_and_discovery() -> None:
    """Verify trusted profiles exist and are discovered canonically."""
    profs = get_available_profiles()
    assert "27m_anchors_best" in profs, "27m_anchors_best missing from available profiles"
    assert "27m_anchors_latest" in profs, "27m_anchors_latest missing from available profiles"
    assert "27m_sft_best" in profs, "27m_sft_best missing from available profiles"

    best_prof = profs["27m_anchors_best"]
    assert "Experimental Chat" in best_prof["model_name"]
    assert best_prof["is_sft"] is True
    assert Path(best_prof["checkpoint_path"]).is_file()
    assert Path(best_prof["tokenizer_path"]).is_file()


def test_chat_mode_and_metadata_compatibility() -> None:
    """Verify anchor checkpoint is recognized as chat model with distinct step and loss metadata."""
    profs = get_available_profiles()
    mgr = ModelManager()

    # Load 27m_anchors_best
    mgr.load_model(
        checkpoint_path=profs["27m_anchors_best"]["checkpoint_path"],
        tokenizer_path=profs["27m_anchors_best"]["tokenizer_path"],
        device_name="cpu",
        model_name=profs["27m_anchors_best"]["model_name"],
        active_profile_key="27m_anchors_best",
    )

    info = mgr.model_info
    assert info is not None
    assert info.is_sft is True
    assert info.model_mode == "chat"
    assert info.total_parameters == 27_411_456
    assert info.pretrain_steps == 5949
    assert info.original_sft_steps == 600
    assert info.anchor_sft_steps == 250
    assert info.global_step == 6799
    assert info.is_anchor_sft is True
    assert abs(info.anchor_val_loss - 1.4802) < 0.01
    assert abs(info.dolly_val_loss - 2.8864) < 0.01

    # Load 27m_sft_best (backward compatibility check)
    mgr_sft = ModelManager()
    mgr_sft.load_model(
        checkpoint_path=profs["27m_sft_best"]["checkpoint_path"],
        tokenizer_path=profs["27m_sft_best"]["tokenizer_path"],
        device_name="cpu",
        model_name=profs["27m_sft_best"]["model_name"],
        active_profile_key="27m_sft_best",
    )

    info_sft = mgr_sft.model_info
    assert info_sft is not None
    assert info_sft.is_sft is True
    assert info_sft.model_mode == "chat"
    assert info_sft.total_parameters == 27_411_456
    assert info_sft.pretrain_steps == 5949
    assert info_sft.sft_step == 600
    assert info_sft.anchor_sft_steps is None
    assert info_sft.is_anchor_sft is False
    assert abs(info_sft.dolly_val_loss - 2.8769) < 0.01


def test_conversational_inference_formatting_parity() -> None:
    """Verify that both original and anchor SFT models receive identical prompt wrapping."""
    prompt = "Is wood a conductor or an insulator?"
    wrapped = format_sft_prompt(prompt)
    assert wrapped == "<|user|>\nIs wood a conductor or an insulator?\n<|assistant|>\n"
    assert wrapped.startswith("<|user|>\n")
    assert wrapped.endswith("\n<|assistant|>\n")


def test_api_endpoints_and_safe_switching() -> None:
    """Verify full FastAPI lifecycle: health, profiles, info, dynamic switching, and generation."""
    # 1. Health check
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["status"] == "ok"

    # 2. Profiles list
    res_prof = client.get("/api/model/profiles")
    assert res_prof.status_code == 200
    data_prof = res_prof.json()
    assert "27m_anchors_best" in data_prof["profiles"]
    assert "27m_sft_best" in data_prof["profiles"]

    # 3. Switch to 27m_anchors_best
    res_load_anchor = client.post("/api/model/load", json={"profile": "27m_anchors_best"})
    assert res_load_anchor.status_code == 200
    info_anchor = res_load_anchor.json()
    assert info_anchor["is_sft"] is True
    assert info_anchor["model_mode"] == "chat"
    assert info_anchor["total_parameters"] == 27_411_456
    assert info_anchor["anchor_sft_steps"] == 250

    # 4. Generate on 27m_anchors_best
    res_gen1 = client.post("/api/generate", json={
        "prompt": "Is wood a conductor or an insulator?",
        "max_new_tokens": 15,
        "temperature": 0.0,
    })
    assert res_gen1.status_code == 200
    data_gen1 = res_gen1.json()
    assert len(data_gen1["generated_text"]) > 0
    assert "<|endoftext|>" not in data_gen1["generated_text"]

    # 5. Switch back to 27m_sft_best
    res_load_sft = client.post("/api/model/load", json={"profile": "27m_sft_best"})
    assert res_load_sft.status_code == 200
    info_sft = res_load_sft.json()
    assert info_sft["is_sft"] is True
    assert info_sft["sft_step"] == 600

    # 6. Generate on 27m_sft_best
    res_gen2 = client.post("/api/generate", json={
        "prompt": "Is wood a conductor or an insulator?",
        "max_new_tokens": 15,
        "temperature": 0.0,
    })
    assert res_gen2.status_code == 200

    # 7. Untrusted profile rejection and state preservation (safe rollback)
    res_bad = client.post("/api/model/load", json={"profile": "malicious_untrusted_profile"})
    assert res_bad.status_code == 400
    # Confirm active model remains loaded and valid
    res_info_after_bad = client.get("/api/model/info")
    assert res_info_after_bad.status_code == 200
    assert res_info_after_bad.json()["total_parameters"] == 27_411_456
