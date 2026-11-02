"""Full 16-step regression and integration test suite for Step 46.

Validates end-to-end integration across:
- Native GPT chat
- Assisted SafeCalculator arithmetic & error handling
- Assisted LocalFact retrieval
- User knowledge ingestion, evidence excerpt & grounded prompt injection
- Conflicting evidence hallucination guardrails
- Opt-in local chat history CRUD, persistence & origin preservation
"""

from starlette.testclient import TestClient

from backend.app import app
from src.chat_history import ChatHistoryManager
from src.user_knowledge import KnowledgeStore


def test_16_step_integrated_assistant_flow(tmp_path, monkeypatch):
    test_db = tmp_path / "flow_chat_history.db"
    test_mgr = ChatHistoryManager(db_path=test_db)
    test_kstore = KnowledgeStore()

    monkeypatch.setattr("backend.app.chat_history_manager", test_mgr)
    monkeypatch.setattr("backend.app.knowledge_store", test_kstore)

    with TestClient(app) as client:
        # Step 1: Health and Model Info verification
        res_health = client.get("/api/health")
        assert res_health.status_code == 200
        assert res_health.json()["status"] == "ok"

        # Step 2: History - Create a new conversation session
        res_c = client.post("/api/history/conversations", json={"title": "Step 46 Flow"})
        assert res_c.status_code == 200
        cid = res_c.json()["id"]

        # Step 3: Assisted Mode - SafeCalculator exact arithmetic (0 model calls)
        res_math = client.post(
            "/api/assistant/respond",
            json={"prompt": "What is 100 * 4 + 25?", "mode": "assisted", "conversation_id": cid},
        )
        assert res_math.status_code == 200
        data_math = res_math.json()
        assert data_math["origin"] == "safe_calculator"
        assert data_math["answer_text"] == "425"
        assert data_math["model_generated"] is False

        # Step 4: Assisted Mode - SafeCalculator error handling (division by zero)
        res_div0 = client.post(
            "/api/assistant/respond",
            json={"prompt": "What is 50 / 0?", "mode": "assisted", "conversation_id": cid},
        )
        assert res_div0.status_code == 200
        data_div0 = res_div0.json()
        assert data_div0["origin"] == "tool_error"
        assert "zero" in data_div0["answer_text"].lower()

        # Step 5: Assisted Mode - Curated LocalFact lookup (0 model calls)
        res_fact = client.post(
            "/api/assistant/respond",
            json={"prompt": "What is the capital of Japan?", "mode": "assisted", "conversation_id": cid},
        )
        assert res_fact.status_code == 200
        data_fact = res_fact.json()
        assert data_fact["origin"] == "local_fact"
        assert "Tokyo" in data_fact["answer_text"]
        assert data_fact["model_generated"] is False

        # Step 6: Native Mode - Arithmetic does NOT trigger calculator in native mode
        # In native mode, routing produces neural origin
        # (if model is loaded, generates; if not loaded, 503)
        res_native = client.post(
            "/api/assistant/respond",
            json={"prompt": "What is 7 + 9?", "mode": "native"},
        )
        assert res_native.status_code in (200, 503)
        if res_native.status_code == 200:
            assert res_native.json()["origin"] == "neural"

        # Step 7: Knowledge Ingestion - Document upload
        doc_content = "Project Chimera was founded in 2022 by Team Delta using PyTorch and CUDA."
        res_upload = client.post(
            "/api/knowledge/source",
            json={
                "session_id": "test_session",
                "display_name": "chimera_specs.txt",
                "content": doc_content,
                "source_type": "txt",
            },
        )
        assert res_upload.status_code == 200
        data_up = res_upload.json()
        assert data_up["chunk_count"] >= 1

        # Step 8: Knowledge Mode - Show Evidence (0 model calls)
        res_ev = client.post(
            "/api/assistant/respond",
            json={
                "prompt": "When was Project Chimera founded?",
                "mode": "assisted",
                "knowledge_mode": "evidence",
                "knowledge_session_id": "test_session",
                "conversation_id": cid,
            },
        )
        assert res_ev.status_code == 200
        data_ev = res_ev.json()
        assert data_ev["origin"] == "document_excerpt"
        assert "2022" in data_ev["answer_text"]
        assert data_ev["model_generated"] is False

        # Step 9: Ingest Contradictory Document
        res_upload2 = client.post(
            "/api/knowledge/source",
            json={
                "session_id": "test_session",
                "display_name": "chimera_alternate.txt",
                "content": "Project Chimera was founded in 2025 by Team Omega.",
                "source_type": "txt",
            },
        )
        assert res_upload2.status_code == 200

        # Step 10: Grounding Guardrail - Halts generation on conflicting evidence
        res_guard = client.post(
            "/api/assistant/respond",
            json={
                "prompt": "When was Project Chimera founded?",
                "mode": "assisted",
                "knowledge_mode": "grounded",
                "knowledge_session_id": "test_session",
                "conversation_id": cid,
            },
        )
        assert res_guard.status_code == 200
        data_guard = res_guard.json()
        assert "Conflicting Evidence Detected" in data_guard["answer_text"]
        assert data_guard["error_code"] == "conflicting_evidence"
        assert data_guard["model_generated"] is False

        # Step 11: History - Check conversation message count and order
        res_hist = client.get(f"/api/history/conversations/{cid}")
        assert res_hist.status_code == 200
        detail = res_hist.json()
        # Message count = 5 user prompts + 5 assistant responses = 10
        assert detail["message_count"] == 10
        assert len(detail["messages"]) == 10

        # Step 12: Verify message origins preserved in history
        origins = [m["origin"] for m in detail["messages"] if m["role"] == "assistant"]
        assert "safe_calculator" in origins
        assert "tool_error" in origins
        assert "local_fact" in origins
        assert "document_excerpt" in origins

        # Step 13: History - Rename conversation
        res_rename = client.patch(
            f"/api/history/conversations/{cid}",
            json={"title": "Completed 16-Step Flow"},
        )
        assert res_rename.status_code == 200
        assert res_rename.json()["title"] == "Completed 16-Step Flow"

        # Step 14: History - List conversations
        res_list = client.get("/api/history/conversations")
        assert res_list.status_code == 200
        assert len(res_list.json()) >= 1
        assert res_list.json()[0]["title"] == "Completed 16-Step Flow"

        # Step 15: Clean up documents from session
        res_del_docs = client.delete("/api/knowledge/sources?session_id=test_session")
        assert res_del_docs.status_code == 200

        # Step 16: History - Delete conversation and verify cascade
        res_del_conv = client.delete(f"/api/history/conversations/{cid}")
        assert res_del_conv.status_code == 200
        res_get_deleted = client.get(f"/api/history/conversations/{cid}")
        assert res_get_deleted.status_code == 404
