"""Test suite for Step 18: Inference safety, profile switching security, and streaming lifecycle."""

import asyncio
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.config import MODEL_PROFILES, register_trusted_profile
from backend.model_loader import model_manager

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_rejected_arbitrary_checkpoint_paths() -> None:
    """Verify arbitrary checkpoint, tokenizer, or device overrides are strictly rejected with 422."""
    client = TestClient(app)

    # 1. Attempting arbitrary checkpoint path injection
    res1 = client.post("/api/model/load", json={"profile": "demo", "checkpoint_path": "C:/evil/model.pt"})
    assert res1.status_code == 422
    assert "extra_forbidden" in str(res1.json()) or "Extra inputs are not permitted" in str(res1.json())

    # 2. Attempting arbitrary tokenizer path injection
    res2 = client.post("/api/model/load", json={"profile": "demo", "tokenizer_path": "C:/evil/tok.json"})
    assert res2.status_code == 422

    # 3. Attempting device override injection
    res3 = client.post("/api/model/load", json={"profile": "demo", "device": "cuda:99"})
    assert res3.status_code == 422

    # 4. Unknown / untrusted profile name
    res4 = client.post("/api/model/load", json={"profile": "../../malicious_profile"})
    assert res4.status_code == 400
    assert "Unknown profile" in res4.json()["detail"]


def test_valid_trusted_profile_switching() -> None:
    """Verify switching among predefined trusted profiles succeeds cleanly."""
    demo_ckpt = REPO_ROOT / "checkpoints" / "demo_checkpoint.pt"
    m5_ckpt = REPO_ROOT / "checkpoints" / "5m_initial" / "model_5m_step200.pt"

    client = TestClient(app)

    if demo_ckpt.is_file():
        res_demo = client.post("/api/model/load", json={"profile": "demo"})
        assert res_demo.status_code == 200
        info_demo = res_demo.json()
        assert info_demo["total_parameters"] == 34880
        assert info_demo["block_size"] == 32

    if m5_ckpt.is_file():
        res_5m = client.post("/api/model/load", json={"profile": "5m"})
        assert res_5m.status_code == 200
        info_5m = res_5m.json()
        assert info_5m["total_parameters"] == 5051904
        assert info_5m["block_size"] == 256


def test_failed_profile_load_preserves_active_model() -> None:
    """Verify a failed model load never leaves a partially initialized model or corrupts active state."""
    client = TestClient(app)

    # 1. Ensure a valid model profile is loaded first
    res = client.post("/api/model/load", json={"profile": "5m"})
    if res.status_code != 200:
        res = client.post("/api/model/load", json={"profile": "demo"})
    assert res.status_code == 200

    prev_info = model_manager.model_info
    assert prev_info is not None
    prev_params = prev_info.total_parameters

    # 2. Register a mock profile with non-existent checkpoint path
    register_trusted_profile(
        "broken_mock",
        {
            "model_name": "Broken Mock Model",
            "checkpoint_path": str(REPO_ROOT / "checkpoints" / "nonexistent_ckpt_12345.pt"),
            "tokenizer_path": str(REPO_ROOT / "checkpoints" / "demo_tokenizer.json"),
        },
    )

    try:
        # 3. Attempt to load the broken profile
        failed_res = client.post("/api/model/load", json={"profile": "broken_mock"})
        assert failed_res.status_code == 404

        # 4. Verify previous model is completely preserved and still operational
        assert model_manager.is_loaded is True
        assert model_manager.model is not None
        assert model_manager.generator is not None
        assert model_manager.model_info is not None
        assert model_manager.model_info.total_parameters == prev_params

        # 5. Verify generation still succeeds with the preserved model
        gen_res = client.post(
            "/api/generate",
            json={"prompt": "The captain looked", "max_new_tokens": 10, "temperature": 0.0},
        )
        assert gen_res.status_code == 200
        assert len(gen_res.json()["generated_text"]) > 0

    finally:
        # Clean up temporary mock profile
        MODEL_PROFILES.pop("broken_mock", None)


def test_stream_cancellation_and_resource_release() -> None:
    """Verify client disconnection cooperatively halts generation and releases model lock."""
    client = TestClient(app)

    # Ensure a model is active
    client.post("/api/model/load", json={"profile": "5m"})

    # Connect to stream and abort after receiving 2 tokens
    chunks_received = 0
    with client.stream(
        "POST",
        "/api/generate/stream",
        json={"prompt": "Once upon a time in a faraway land", "max_new_tokens": 50, "temperature": 0.0},
    ) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("event: token"):
                chunks_received += 1
                if chunks_received >= 2:
                    break  # Abruptly disconnect client

    assert chunks_received >= 2
    # Verify the model lock is cleanly released and not leaked
    assert not model_manager.lock.locked()


def test_immediate_new_request_after_cancellation() -> None:
    """Verify sending an immediate generation request after stream cancellation succeeds with no deadlocks."""
    client = TestClient(app)

    # 1. Abort a stream early
    with client.stream(
        "POST",
        "/api/generate/stream",
        json={"prompt": "The young man walked", "max_new_tokens": 40, "temperature": 0.0},
    ) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("event: token"):
                break  # Abort on first token

    # 2. Immediately execute a non-streaming generation
    new_res = client.post(
        "/api/generate",
        json={"prompt": "The young man walked", "max_new_tokens": 10, "temperature": 0.0},
    )
    assert new_res.status_code == 200
    assert new_res.json()["generated_tokens"] == 10
    assert not model_manager.lock.locked()


@pytest.mark.asyncio
async def test_concurrent_generation_and_model_switching() -> None:
    """Verify model switching serializes safely and cannot overlap active inference."""
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
        # Ensure 5M profile is loaded
        await async_client.post("/api/model/load", json={"profile": "5m"})

        # Task 1: Slow generation request
        gen_task = asyncio.create_task(
            async_client.post(
                "/api/generate",
                json={"prompt": "She thought that everything", "max_new_tokens": 30, "temperature": 0.0},
            )
        )

        # Allow generation task to start and acquire lock
        await asyncio.sleep(0.02)

        # Task 2: Model switch request to demo
        switch_task = asyncio.create_task(
            async_client.post("/api/model/load", json={"profile": "demo"})
        )

        gen_resp, switch_resp = await asyncio.gather(gen_task, switch_task)

        # Both must succeed without deadlock or corrupted model state
        assert gen_resp.status_code == 200
        assert gen_resp.json()["generated_tokens"] == 30

        assert switch_resp.status_code == 200
        assert switch_resp.json()["total_parameters"] == 34880

        # Subsequent generation uses the newly switched demo model
        final_gen = await async_client.post(
            "/api/generate",
            json={"prompt": "Once", "max_new_tokens": 5, "temperature": 0.0},
        )
        assert final_gen.status_code == 200
        assert not model_manager.lock.locked()
