"""Integration tests for local chat history REST API endpoints."""

from starlette.testclient import TestClient

from backend.app import app
from src.chat_history import ChatHistoryManager


def test_chat_history_crud_endpoints(tmp_path, monkeypatch):
    test_db = tmp_path / "api_test_history.db"
    test_mgr = ChatHistoryManager(db_path=test_db)
    monkeypatch.setattr("backend.app.chat_history_manager", test_mgr)

    with TestClient(app) as client:
        # 1. Initially empty
        res = client.get("/api/history/conversations")
        assert res.status_code == 200
        assert res.json() == []

        # 2. Create conversation
        res = client.post("/api/history/conversations", json={"title": "Session One"})
        assert res.status_code == 200
        data = res.json()
        cid = data["id"]
        assert data["title"] == "Session One"
        assert data["message_count"] == 0

        # 3. Add message
        res = client.post(
            f"/api/history/conversations/{cid}/messages",
            json={
                "role": "user",
                "content": "What is 42 * 2?",
                "origin": "neural",
            },
        )
        assert res.status_code == 200
        mid = res.json()["id"]
        assert res.json()["role"] == "user"

        # 4. Fetch detail
        res = client.get(f"/api/history/conversations/{cid}")
        assert res.status_code == 200
        detail = res.json()
        assert detail["id"] == cid
        assert detail["message_count"] == 1
        assert len(detail["messages"]) == 1
        assert detail["messages"][0]["id"] == mid

        # 5. Update title
        res = client.patch(
            f"/api/history/conversations/{cid}",
            json={"title": "Arithmetic Questions"},
        )
        assert res.status_code == 200
        assert res.json()["title"] == "Arithmetic Questions"

        # 6. Delete conversation
        res = client.delete(f"/api/history/conversations/{cid}")
        assert res.status_code == 200
        assert res.json()["success"] is True

        # 7. Verify deletion
        res = client.get(f"/api/history/conversations/{cid}")
        assert res.status_code == 404


def test_assistant_respond_persists_history(tmp_path, monkeypatch):
    test_db = tmp_path / "assistant_history.db"
    test_mgr = ChatHistoryManager(db_path=test_db)
    monkeypatch.setattr("backend.app.chat_history_manager", test_mgr)

    with TestClient(app) as client:
        conv = test_mgr.create_conversation(title="Assisted Test")
        cid = conv["id"]

        # Call assistant endpoint with conversation_id for arithmetic
        res = client.post(
            "/api/assistant/respond",
            json={
                "prompt": "What is 15 + 27?",
                "mode": "assisted",
                "conversation_id": cid,
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["origin"] == "safe_calculator"
        assert data["answer_text"] == "42"

        # Verify conversation messages were persisted
        fetched = test_mgr.get_conversation(cid)
        assert fetched["message_count"] == 2
        assert len(fetched["messages"]) == 2
        assert fetched["messages"][0]["role"] == "user"
        assert fetched["messages"][0]["content"] == "What is 15 + 27?"
        assert fetched["messages"][1]["role"] == "assistant"
        assert fetched["messages"][1]["content"] == "42"
        assert fetched["messages"][1]["origin"] == "safe_calculator"
