"""Unit tests for local chat history manager (src/chat_history.py)."""

from pathlib import Path
import pytest
import tempfile

from src.chat_history import ChatHistoryManager


@pytest.fixture
def temp_history_manager(tmp_path: Path):
    db_file = tmp_path / "test_chat_history.db"
    return ChatHistoryManager(db_path=db_file)


def test_create_and_get_conversation(temp_history_manager: ChatHistoryManager):
    conv = temp_history_manager.create_conversation(title="Test Discussion")
    assert conv["id"] is not None
    assert conv["title"] == "Test Discussion"
    assert conv["message_count"] == 0

    fetched = temp_history_manager.get_conversation(conv["id"])
    assert fetched is not None
    assert fetched["id"] == conv["id"]
    assert fetched["title"] == "Test Discussion"
    assert fetched["messages"] == []


def test_add_messages_and_auto_title(temp_history_manager: ChatHistoryManager):
    conv = temp_history_manager.create_conversation()
    cid = conv["id"]

    # First user message should update default title
    msg1 = temp_history_manager.add_message(
        conversation_id=cid,
        role="user",
        content="What is the speed of light in vacuum?",
    )
    assert msg1["role"] == "user"
    assert msg1["origin"] == "neural"

    # Verify title was auto-updated
    fetched = temp_history_manager.get_conversation(cid)
    assert "speed of light" in fetched["title"].lower()
    assert fetched["message_count"] == 1

    # Assistant response with tool origin
    msg2 = temp_history_manager.add_message(
        conversation_id=cid,
        role="assistant",
        content="299,792 km/s",
        origin="local_fact",
        details={"source": "physics_record", "confidence": 1.0},
    )
    assert msg2["origin"] == "local_fact"
    assert msg2["details"]["source"] == "physics_record"

    fetched2 = temp_history_manager.get_conversation(cid)
    assert fetched2["message_count"] == 2
    assert len(fetched2["messages"]) == 2
    assert fetched2["messages"][0]["role"] == "user"
    assert fetched2["messages"][1]["role"] == "assistant"
    assert fetched2["messages"][1]["origin"] == "local_fact"


def test_list_and_update_conversations(temp_history_manager: ChatHistoryManager):
    c1 = temp_history_manager.create_conversation(title="Chat One")
    c2 = temp_history_manager.create_conversation(title="Chat Two")

    convs = temp_history_manager.list_conversations()
    assert len(convs) == 2

    # Update title
    ok = temp_history_manager.update_conversation_title(c1["id"], "Renamed Chat One")
    assert ok is True
    updated = temp_history_manager.get_conversation(c1["id"])
    assert updated["title"] == "Renamed Chat One"


def test_delete_and_cascade(temp_history_manager: ChatHistoryManager):
    c = temp_history_manager.create_conversation(title="To Delete")
    cid = c["id"]
    temp_history_manager.add_message(cid, "user", "Hello")
    temp_history_manager.add_message(cid, "assistant", "Hi there")

    # Delete conversation
    deleted = temp_history_manager.delete_conversation(cid)
    assert deleted is True

    # Conversation and messages should be gone
    assert temp_history_manager.get_conversation(cid) is None
    convs = temp_history_manager.list_conversations()
    assert len(convs) == 0


def test_sql_injection_safety(temp_history_manager: ChatHistoryManager):
    # Malicious injection attempt in title and content
    malicious_title = "'); DROP TABLE conversations; --"
    conv = temp_history_manager.create_conversation(title=malicious_title)
    cid = conv["id"]

    malicious_content = "Robert'); DROP TABLE messages; --"
    temp_history_manager.add_message(cid, "user", malicious_content)

    # Tables should still exist and conversation intact
    fetched = temp_history_manager.get_conversation(cid)
    assert fetched is not None
    assert fetched["title"] == malicious_title
    assert len(fetched["messages"]) == 1
    assert fetched["messages"][0]["content"] == malicious_content
