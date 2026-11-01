"""Focused unit tests for Step 21: Dynamic checkpoint discovery, profile switching, and UI backend integration."""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app import app, model_manager
from backend.config import MODEL_PROFILES, register_trusted_profile, get_available_profiles


@pytest.fixture
def client():
    """Create test client ensuring application lifespan runs."""
    with TestClient(app) as test_client:
        yield test_client


def test_checkpoint_discovery_only_existing(client):
    """Verify /api/model/profiles only returns profiles whose required files exist on disk."""
    res = client.get("/api/model/profiles")
    assert res.status_code == 200
    data = res.json()

    assert "profiles" in data
    assert "active_model" in data
    profiles = data["profiles"]

    # Every reported profile must have verified checkpoint and tokenizer
    for prof_key, prof_info in profiles.items():
        assert prof_info["checkpoint_exists"] is True
        assert prof_info["tokenizer_exists"] is True
        assert Path(prof_info["checkpoint_path"]).is_file()
        assert Path(prof_info["tokenizer_path"]).is_file()

    # The verified trained models must be present
    assert "5m_best" in profiles or "5m_step200" in profiles or "demo" in profiles


def test_profile_switching_best_and_step200(client):
    """Verify switching between Phase 2 Best, 5M Best, 5M Step 200, and Demo profiles."""
    profiles = get_available_profiles()

    # 1. Switch to 5m_phase2_best if available
    if "5m_phase2_best" in profiles:
        res = client.post("/api/model/load", json={"profile": "5m_phase2_best"})
        assert res.status_code == 200
        info = res.json()
        assert info["total_parameters"] == 5051904
        assert info["block_size"] == 256
        assert info["checkpoint_step"] == 3000
        assert info["checkpoint_val_loss"] is not None
        assert info["expanded_val_loss"] is not None
        assert info["original_val_loss"] is not None

    # 2. Switch to 5m_best if available
    if "5m_best" in profiles:
        res = client.post("/api/model/load", json={"profile": "5m_best"})
        assert res.status_code == 200
        info = res.json()
        assert info["total_parameters"] == 5051904
        assert info["block_size"] == 256
        assert info["checkpoint_step"] == 1000
        assert info["checkpoint_val_loss"] == pytest.approx(5.4116, abs=0.01)

    # 3. Switch to 5m_step200 if available
    if "5m_step200" in profiles:
        res = client.post("/api/model/load", json={"profile": "5m_step200"})
        assert res.status_code == 200
        info = res.json()
        assert info["total_parameters"] == 5051904
        assert info["block_size"] == 256
        assert info["checkpoint_step"] == 200

    # 4. Switch to Demo
    if "demo" in profiles:
        res = client.post("/api/model/load", json={"profile": "demo"})
        assert res.status_code == 200
        info = res.json()
        assert info["total_parameters"] < 1000000


def test_generation_after_switching(client):
    """Verify text generation and SSE streaming function correctly after switching profiles."""
    profiles = get_available_profiles()
    target_prof = "5m_best" if "5m_best" in profiles else ("5m_step200" if "5m_step200" in profiles else "demo")

    load_res = client.post("/api/model/load", json={"profile": target_prof})
    assert load_res.status_code == 200

    # Test non-streaming generation
    gen_res = client.post(
        "/api/generate",
        json={"prompt": "Once upon a time", "max_new_tokens": 15, "temperature": 0.0},
    )
    assert gen_res.status_code == 200
    gen_data = gen_res.json()
    assert gen_data["generated_tokens"] == 15
    assert len(gen_data["generated_text"]) > 0

    # Test streaming SSE generation
    with client.stream(
        "POST",
        "/api/generate/stream",
        json={"prompt": "The captain looked", "max_new_tokens": 10, "temperature": 0.0},
    ) as stream_res:
        assert stream_res.status_code == 200
        events = stream_res.iter_lines()
        found_token = False
        found_done = False
        for line in events:
            if line.startswith("event: token"):
                found_token = True
            elif line.startswith("event: done"):
                found_done = True
        assert found_token, "Must stream token events"
        assert found_done, "Must stream terminal done event"


def test_missing_profile_and_arbitrary_path_rejection(client):
    """Verify unknown profiles and arbitrary file paths are rejected."""
    # Unknown profile
    res1 = client.post("/api/model/load", json={"profile": "nonexistent_profile_xyz"})
    assert res1.status_code == 400
    assert "Unknown profile" in res1.json()["detail"]

    # Extra fields rejected by extra='forbid'
    res2 = client.post(
        "/api/model/load",
        json={"profile": "demo", "custom_path": "C:/Windows/System32/calc.exe"},
    )
    assert res2.status_code == 422


def test_failed_checkpoint_load_preserves_active_model(client):
    """Verify that a corrupted checkpoint load failure preserves the previously active model intact."""
    profiles = get_available_profiles()
    initial_prof = "demo" if "demo" in profiles else list(profiles.keys())[0]

    client.post("/api/model/load", json={"profile": initial_prof})
    active_before = model_manager.model_info.model_name

    # Register an invalid profile pointing to a missing/invalid checkpoint
    register_trusted_profile(
        "test_broken_ckpt",
        {
            "model_name": "Broken Model",
            "checkpoint_path": "checkpoints/nonexistent_ckpt_12345.pt",
            "tokenizer_path": "checkpoints/demo_tokenizer.json",
        },
    )

    fail_res = client.post("/api/model/load", json={"profile": "test_broken_ckpt"})
    assert fail_res.status_code in (404, 500)

    # Active model must be completely preserved
    assert model_manager.is_loaded is True
    assert model_manager.model_info.model_name == active_before


def test_profile_deduplication_no_duplicate_artifacts(client):
    """Verify each distinct model artifact is returned only once (no duplicate profiles)."""
    res = client.get("/api/model/profiles")
    assert res.status_code == 200
    data = res.json()
    profiles = data["profiles"]

    # No duplicate checkpoint file paths across exposed profiles
    seen_paths = set()
    for key, p in profiles.items():
        resolved_ckpt = str(Path(p["checkpoint_path"]).resolve())
        assert resolved_ckpt not in seen_paths, f"Duplicate checkpoint artifact exposed: {key} ({resolved_ckpt})"
        seen_paths.add(resolved_ckpt)

    # Ensure 5m_step200 and 5m_initial are not both in profiles
    assert not ("5m_step200" in profiles and "5m_initial" in profiles), (
        "Both 5m_step200 and 5m_initial are exposed simultaneously. Distinct artifacts must appear only once."
    )


def test_backward_compatible_aliases(client):
    """Verify legacy profile aliases ('5m', '5m_initial') load correctly and map to canonical active identity."""
    profiles = get_available_profiles()
    if "5m_step200" in profiles or "5m_initial" in MODEL_PROFILES:
        for alias in ("5m", "5m_initial"):
            res = client.post("/api/model/load", json={"profile": alias})
            assert res.status_code == 200
            data = res.json()
            assert data["active_profile"] == "5m_step200"
            assert data["total_parameters"] == 5051904

            # Verify /api/model/profiles reports canonical active profile
            prof_res = client.get("/api/model/profiles")
            assert prof_res.status_code == 200
            prof_data = prof_res.json()
            assert prof_data["active_profile"] == "5m_step200"


def test_active_profile_consistency_across_all_profiles(client):
    """Verify active profile identity is consistent between /api/model/info and /api/model/profiles."""
    profiles = get_available_profiles()
    for prof_key in profiles.keys():
        load_res = client.post("/api/model/load", json={"profile": prof_key})
        assert load_res.status_code == 200
        info_data = load_res.json()
        assert info_data["active_profile"] == prof_key

        prof_res = client.get("/api/model/profiles")
        assert prof_res.status_code == 200
        prof_data = prof_res.json()
        assert prof_data["active_profile"] == prof_key
