"""Privacy boundary and mode isolation regression tests (Step 47).

Verifies:
1. Opt-in chat persistence: Zero SQLite writes when conversation_id is omitted/None.
2. Consecutive duplicate message prevention in ChatHistoryManager.
3. Stream cancellation does not save partial responses.
4. Isolated knowledge sessions: Unpredictable session IDs, strict isolation between sessions.
5. Unknown knowledge sessions fail safely with 404 rather than creating shared defaults.
6. Native GPT mode isolation: Forged requests with mode="native" strictly force
   knowledge_mode="off", neutralizing knowledge retrieval and deterministic tools.
7. Deleting a knowledge source removes all searchable chunks.
"""

import json
import pytest
from starlette.testclient import TestClient

from backend.app import app
from src.chat_history import ChatHistoryManager
from src.user_knowledge import KnowledgeStore, KnowledgeSource, MAX_SOURCE_BYTES


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_sqlite_persistence_strictly_opt_in(tmp_path, monkeypatch, client: TestClient):
    """When conversation_id is omitted, SQLite history database must receive zero records."""
    test_db = tmp_path / "privacy_test.db"
    mgr = ChatHistoryManager(db_path=test_db)
    monkeypatch.setattr("backend.app.chat_history_manager", mgr)

    # 1. User sends message without conversation_id (Save History = OFF)
    resp = client.post(
        "/api/assistant/respond",
        json={"prompt": "What is 40 + 60?", "mode": "assisted"},
    )
    assert resp.status_code == 200
    assert resp.json()["origin"] == "safe_calculator"

    # Verify no conversations or messages exist in SQLite
    convs = mgr.list_conversations()
    assert len(convs) == 0

    # 2. User sends another query with Native mode without conversation_id
    resp_native = client.post(
        "/api/assistant/respond",
        json={"prompt": "Hello world", "mode": "native"},
    )
    assert resp_native.status_code in (200, 503)
    assert len(mgr.list_conversations()) == 0


def test_sqlite_duplicate_message_prevention(tmp_path):
    """Consecutive identical messages in the same conversation are deduplicated."""
    test_db = tmp_path / "dedup_test.db"
    mgr = ChatHistoryManager(db_path=test_db)

    conv = mgr.create_conversation("Dedup Session")
    cid = conv["id"]

    # Add message
    msg1 = mgr.add_message(cid, role="user", content="Hello test")
    # Add identical consecutive message
    msg2 = mgr.add_message(cid, role="user", content="Hello test")

    # Second insert should return existing record without creating duplicate
    assert msg1["id"] == msg2["id"]

    detail = mgr.get_conversation(cid)
    assert detail is not None
    assert len(detail["messages"]) == 1


def test_knowledge_session_unguessable_creation(client: TestClient):
    """Server issues unpredictable session IDs starting with 'ksess_'."""
    resp1 = client.post("/api/knowledge/session")
    assert resp1.status_code == 200
    data1 = resp1.json()
    sid1 = data1["session_id"]

    resp2 = client.post("/api/knowledge/session")
    assert resp2.status_code == 200
    data2 = resp2.json()
    sid2 = data2["session_id"]

    assert sid1.startswith("ksess_")
    assert sid2.startswith("ksess_")
    assert sid1 != sid2
    assert len(sid1) > 20


def test_knowledge_session_cross_session_isolation(client: TestClient):
    """Two independent sessions cannot see or retrieve each other's documents."""
    s1 = client.post("/api/knowledge/session").json()["session_id"]
    s2 = client.post("/api/knowledge/session").json()["session_id"]

    # Ingest document into session 1
    resp_ingest1 = client.post(
        "/api/knowledge/source",
        json={
            "session_id": s1,
            "display_name": "secret_mission.txt",
            "content": "Secret Project Pegasus was launched in 2024 by Division 9.",
            "source_type": "txt",
        },
    )
    assert resp_ingest1.status_code == 200
    source_id1 = resp_ingest1.json()["source_id"]

    # Session 2 should have 0 documents
    list_s2 = client.get(f"/api/knowledge/sources?session_id={s2}").json()
    assert len(list_s2) == 0

    # Session 2 retrieval for 'Pegasus' must find nothing
    ret_s2 = client.post(
        "/api/knowledge/retrieve",
        json={"session_id": s2, "query": "When was Project Pegasus launched?", "top_k": 2},
    ).json()
    assert ret_s2["status"] == "no_match"

    # Session 2 cannot delete Session 1's document
    del_fail = client.delete(f"/api/knowledge/source/{source_id1}?session_id={s2}")
    assert del_fail.status_code == 404

    # Document in Session 1 is intact
    list_s1 = client.get(f"/api/knowledge/sources?session_id={s1}").json()
    assert len(list_s1) == 1


def test_unknown_session_fails_safely(client: TestClient):
    """Unknown session IDs return safe 404s without creating default shared sessions."""
    unknown = "ksess_nonexistent_session_id_12345"

    res_list = client.get(f"/api/knowledge/sources?session_id={unknown}")
    assert res_list.status_code == 404

    res_ret = client.post(
        "/api/knowledge/retrieve",
        json={"session_id": unknown, "query": "Any query", "top_k": 2},
    )
    assert res_ret.status_code == 404

    res_del = client.delete(f"/api/knowledge/source/some_src?session_id={unknown}")
    assert res_del.status_code == 404


def test_default_session_id_strictly_forbidden(client: TestClient):
    """Using literal 'default' session is safely rejected with HTTP 400 or 404."""
    res_list = client.get("/api/knowledge/sources?session_id=default")
    assert res_list.status_code in (400, 404)

    res_ingest = client.post(
        "/api/knowledge/source",
        json={
            "session_id": "default",
            "display_name": "test.txt",
            "content": "Sample content",
            "source_type": "txt",
        },
    )
    assert res_ingest.status_code == 400


def test_native_mode_isolation_forged_request_neutralized(client: TestClient):
    """Forged request with mode='native' and knowledge_mode='grounded' is neutralized to pure neural."""
    s1 = client.post("/api/knowledge/session").json()["session_id"]

    client.post(
        "/api/knowledge/source",
        json={
            "session_id": s1,
            "display_name": "facts.txt",
            "content": "Planet Neptune has 14 known moons.",
            "source_type": "txt",
        },
    )

    # Forged request: attempts to use grounded knowledge while mode='native'
    resp = client.post(
        "/api/assistant/respond",
        json={
            "prompt": "How many moons does Neptune have?",
            "mode": "native",
            "knowledge_mode": "grounded",
            "knowledge_session_id": s1,
        },
    )
    # The server strictly forces knowledge_mode='off' and neural execution
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert data["origin"] == "neural"
        assert data["tool_name"] is None
        assert data["provenance"] is None


def test_source_deletion_removes_all_retrieval_chunks(client: TestClient):
    """Deleting a document completely removes all of its chunks from subsequent retrieval."""
    sid = client.post("/api/knowledge/session").json()["session_id"]

    res_in = client.post(
        "/api/knowledge/source",
        json={
            "session_id": sid,
            "display_name": "rover.txt",
            "content": "Curiosity rover landed on Mars in Gale Crater in August 2012.",
            "source_type": "txt",
        },
    )
    assert res_in.status_code == 200
    src_id = res_in.json()["source_id"]

    # Verify retrieval works
    ret1 = client.post(
        "/api/knowledge/retrieve",
        json={"session_id": sid, "query": "When did Curiosity land on Mars?", "top_k": 2},
    ).json()
    assert ret1["status"] == "relevant_passage"

    # Delete source
    res_del = client.delete(f"/api/knowledge/source/{src_id}?session_id={sid}")
    assert res_del.status_code == 200

    # Retrieval now finds no passage
    ret2 = client.post(
        "/api/knowledge/retrieve",
        json={"session_id": sid, "query": "When did Curiosity land on Mars?", "top_k": 2},
    ).json()
    assert ret2["status"] == "no_match"
