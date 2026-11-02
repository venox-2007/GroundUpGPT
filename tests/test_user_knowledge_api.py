"""Integration tests for User Knowledge REST API endpoints and Assistant integration."""

import pytest
from starlette.testclient import TestClient

from backend.app import app
from src.user_knowledge import knowledge_store, MAX_SOURCE_BYTES


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_session():
    """Ensure test session is clean before and after each test."""
    knowledge_store.clear_session("test_api_session")
    yield
    knowledge_store.clear_session("test_api_session")


def test_knowledge_ingestion_api(client: TestClient):
    """Test document creation, listing, retrieval, and deletion via REST API."""
    session_id = "test_api_session"

    # 1. Ingest document
    payload = {
        "session_id": session_id,
        "display_name": "missions.txt",
        "content": (
            "The Apollo 11 mission was launched on July 16, 1969. "
            "Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969.\n\n"
            "Project Voyager was an American scientific program that launched two robotic interstellar probes, Voyager 1 and Voyager 2, in 1977."
        ),
        "source_type": "txt",
    }
    resp = client.post("/api/knowledge/source", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["session_id"] == session_id
    assert data["display_name"] == "missions.txt"
    assert data["chunk_count"] >= 2
    source_id = data["source_id"]

    # 2. List sources
    list_resp = client.get(f"/api/knowledge/sources?session_id={session_id}")
    assert list_resp.status_code == 200
    sources = list_resp.json()
    assert len(sources) == 1
    assert sources[0]["source_id"] == source_id

    # 3. Retrieve chunks
    ret_resp = client.post(
        "/api/knowledge/retrieve",
        json={"session_id": session_id, "query": "When did Voyager launch?", "top_k": 2},
    )
    assert ret_resp.status_code == 200
    ret_data = ret_resp.json()
    assert ret_data["status"] == "relevant_passage"
    assert "1977" in (ret_data["text"] or "")
    assert ret_data["source_display_name"] == "missions.txt"

    # 4. Delete source
    del_resp = client.delete(f"/api/knowledge/source/{source_id}?session_id={session_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["success"] is True

    # 5. Verify source is gone
    list_resp2 = client.get(f"/api/knowledge/sources?session_id={session_id}")
    assert list_resp2.status_code == 200
    assert len(list_resp2.json()) == 0


def test_knowledge_ingestion_validations(client: TestClient):
    """Test rejection of oversized documents and empty content."""
    session_id = "test_api_session"

    # Oversized document
    big_content = "y" * (MAX_SOURCE_BYTES + 10)
    resp_big = client.post(
        "/api/knowledge/source",
        json={"session_id": session_id, "display_name": "big.txt", "content": big_content},
    )
    assert resp_big.status_code in (400, 422)

    # Empty content
    resp_empty = client.post(
        "/api/knowledge/source",
        json={"session_id": session_id, "display_name": "empty.txt", "content": "   \n  "},
    )
    assert resp_empty.status_code in (400, 422)


def test_assistant_evidence_mode(client: TestClient):
    """Evidence mode returns exact document excerpt with zero model calls."""
    session_id = "test_api_session"
    client.post(
        "/api/knowledge/source",
        json={
            "session_id": session_id,
            "display_name": "telescopes.md",
            "content": "The James Webb Space Telescope was launched on December 25, 2021 from French Guiana.",
            "source_type": "md",
        },
    )

    resp = client.post(
        "/api/assistant/respond",
        json={
            "prompt": "When was the James Webb Space Telescope launched?",
            "mode": "assisted",
            "knowledge_mode": "evidence",
            "knowledge_session_id": session_id,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["origin"] == "document_excerpt"
    assert data["model_generated"] is False
    assert data["tool_name"] == "UserKnowledgeRetriever"
    assert data["provenance"] == "telescopes.md"
    assert "December 25, 2021" in data["answer_text"]
    assert data["stop_reason"] == "tool_complete"
    assert data["prompt_tokens"] == 0
    assert data["generated_tokens"] == 0


def test_assistant_evidence_mode_no_match(client: TestClient):
    """Evidence mode politely indicates when no match is found without hallucinating."""
    session_id = "test_api_session"
    client.post(
        "/api/knowledge/source",
        json={
            "session_id": session_id,
            "display_name": "chemistry.txt",
            "content": "Helium is a chemical element with symbol He and atomic number 2.",
        },
    )

    resp = client.post(
        "/api/assistant/respond",
        json={
            "prompt": "What is the capital of Mars?",
            "mode": "assisted",
            "knowledge_mode": "evidence",
            "knowledge_session_id": session_id,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["origin"] == "document_excerpt"
    assert data["model_generated"] is False
    assert "No unambiguous evidence passage found" in data["answer_text"]
    assert data["details"]["status"] == "no_match"
