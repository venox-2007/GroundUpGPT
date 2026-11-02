"""Local chat history management using SQLite with parameterized queries.

GroundUpGPT Architecture Boundary:
- Stores conversations and messages locally for user review and session recall.
- GroundUpGPT uses a strictly single-turn context window of 256 tokens.
- Chat history is NOT fed as an unbounded multi-turn prompt into the model,
  preserving causal context integrity and preventing context window overflow.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Union
import uuid

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "chat_history.db"


def utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ChatMessageRecord:
    id: str
    conversation_id: str
    role: str  # 'user' | 'assistant'
    content: str
    origin: str  # 'neural' | 'safe_calculator' | 'local_fact' | 'document_excerpt' | 'grounded_neural' | 'tool_error'
    model_profile: Optional[str]
    created_at: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "origin": self.origin,
            "model_profile": self.model_profile,
            "created_at": self.created_at,
            "details": self.details or {},
        }


@dataclass
class ConversationRecord:
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    messages: List[ChatMessageRecord] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
        }
        if self.messages is not None:
            d["messages"] = [m.to_dict() for m in self.messages]
        return d


class ChatHistoryManager:
    """Thread-safe SQLite manager for local conversation history."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock, self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    origin TEXT NOT NULL DEFAULT 'neural',
                    model_profile TEXT,
                    created_at TEXT NOT NULL,
                    details_json TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation 
                ON messages(conversation_id, created_at);
                """
            )
            conn.commit()

    def create_conversation(
        self,
        title: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        cid = conversation_id or str(uuid.uuid4())
        now = utc_now_iso()
        t = (title or "New Conversation").strip() or "New Conversation"

        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, title, created_at, updated_at, message_count)
                VALUES (?, ?, ?, ?, 0)
                """,
                (cid, t, now, now),
            )
            conn.commit()

        return {
            "id": cid,
            "title": t,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            c_row = conn.execute(
                "SELECT id, title, created_at, updated_at, message_count FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()

            if not c_row:
                return None

            m_rows = conn.execute(
                """
                SELECT id, conversation_id, role, content, origin, model_profile, created_at, details_json
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            ).fetchall()

            messages = []
            for row in m_rows:
                details = None
                if row["details_json"]:
                    try:
                        details = json.loads(row["details_json"])
                    except Exception:
                        details = {}
                messages.append(
                    ChatMessageRecord(
                        id=row["id"],
                        conversation_id=row["conversation_id"],
                        role=row["role"],
                        content=row["content"],
                        origin=row["origin"],
                        model_profile=row["model_profile"],
                        created_at=row["created_at"],
                        details=details,
                    )
                )

            record = ConversationRecord(
                id=c_row["id"],
                title=c_row["title"],
                created_at=c_row["created_at"],
                updated_at=c_row["updated_at"],
                message_count=c_row["message_count"],
                messages=messages,
            )
            return record.to_dict()

    def list_conversations(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, title, created_at, updated_at, message_count
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

            return [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "message_count": r["message_count"],
                }
                for r in rows
            ]

    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        t = title.strip()
        if not t:
            return False
        with self._lock, self._get_connection() as conn:
            cur = conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (t, utc_now_iso(), conversation_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()
            return cur.rowcount > 0

    def clear_all_conversations(self) -> int:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("DELETE FROM conversations")
            conn.commit()
            return cur.rowcount

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        origin: str = "neural",
        model_profile: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        mid = message_id or str(uuid.uuid4())
        now = utc_now_iso()
        details_str = json.dumps(details or {}) if details else None

        with self._lock, self._get_connection() as conn:
            # Check conversation exists; create if not
            c_row = conn.execute(
                "SELECT id, title, message_count FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()

            if not c_row:
                auto_title = content.strip()[:40] if role == "user" else "New Conversation"
                conn.execute(
                    """
                    INSERT INTO conversations (id, title, created_at, updated_at, message_count)
                    VALUES (?, ?, ?, ?, 0)
                    """,
                    (conversation_id, auto_title or "New Conversation", now, now),
                )
                c_title = auto_title or "New Conversation"
                c_count = 0
            else:
                c_title = c_row["title"]
                c_count = c_row["message_count"]

            # If title is default and user sends first message, derive clean title
            if role == "user" and c_title in ("New Conversation", "Untitled"):
                clean_title = " ".join(content.strip().split())[:45]
                if clean_title:
                    c_title = clean_title
                    conn.execute(
                        "UPDATE conversations SET title = ? WHERE id = ?",
                        (c_title, conversation_id),
                    )

            # Insert message
            conn.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, origin, model_profile, created_at, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (mid, conversation_id, role, content, origin, model_profile, now, details_str),
            )

            # Update conversation stats
            conn.execute(
                """
                UPDATE conversations
                SET updated_at = ?, message_count = message_count + 1
                WHERE id = ?
                """,
                (now, conversation_id),
            )
            conn.commit()

        return {
            "id": mid,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "origin": origin,
            "model_profile": model_profile,
            "created_at": now,
            "details": details or {},
        }
