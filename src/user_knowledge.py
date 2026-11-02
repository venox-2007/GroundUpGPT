"""User knowledge ingestion, token-aware chunking, and session store for GroundUpGPT.

Manages user-provided documents (pasted text, text files, markdown) in scoped
in-memory sessions. Chunks documents into 30-70 BPE token segments, preserving
exact substring offsets for reliable source attribution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import logging
import os
from pathlib import Path
import re
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

from src.bpe_tokenizer import BPETokenizer

logger = logging.getLogger(__name__)

# System limits
MAX_SOURCE_BYTES = 16 * 1024  # 16 KB per document
MAX_SOURCES_PER_SESSION = 10   # Max 10 documents per session
MAX_SESSION_BYTES = 64 * 1024  # 64 KB total per session
MAX_DISPLAY_NAME_CHARS = 100

DEFAULT_MIN_CHUNK_TOKENS = 30
DEFAULT_MAX_CHUNK_TOKENS = 70

DEFAULT_TOKENIZER_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "bpe_vocab4096.json"

_tokenizer_lock = threading.Lock()
_cached_tokenizer: Optional[BPETokenizer] = None


def get_default_tokenizer() -> BPETokenizer:
    """Lazily load and cache the default BPE tokenizer."""
    global _cached_tokenizer
    if _cached_tokenizer is None:
        with _tokenizer_lock:
            if _cached_tokenizer is None:
                if DEFAULT_TOKENIZER_PATH.exists():
                    _cached_tokenizer = BPETokenizer.load(str(DEFAULT_TOKENIZER_PATH))
                else:
                    logger.warning("Default BPE tokenizer checkpoint not found; creating empty BPETokenizer.")
                    _cached_tokenizer = BPETokenizer()
    return _cached_tokenizer


def sanitize_display_name(name: str) -> str:
    """Sanitize document display name by removing directory traversal and control characters."""
    if not name or not name.strip():
        return "Untitled Document"
    # Strip paths
    cleaned = os.path.basename(name.replace("\\", "/"))
    # Remove control characters
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", cleaned)
    # Remove path traversal tokens
    cleaned = cleaned.replace("..", "").strip()
    if not cleaned:
        return "Untitled Document"
    return cleaned[:MAX_DISPLAY_NAME_CHARS]


@dataclass
class KnowledgeChunk:
    """A granular chunk of user knowledge with exact substring coordinates.

    Invariant: source_content[start_char:end_char] == text
    """

    chunk_id: str
    source_id: str
    text: str
    start_char: int
    end_char: int
    token_count: int
    content_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeSource:
    """An ingested user document or pasted text record."""

    source_id: str
    session_id: str
    display_name: str
    source_type: str  # 'pasted', 'txt', 'md'
    content: str
    content_hash: str
    character_count: int
    token_count: int
    created_at: str
    chunks: List[KnowledgeChunk] = field(default_factory=list)

    def to_dict(self, include_chunks: bool = True) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "source_id": self.source_id,
            "session_id": self.session_id,
            "display_name": self.display_name,
            "source_type": self.source_type,
            "content_hash": self.content_hash,
            "character_count": self.character_count,
            "token_count": self.token_count,
            "chunk_count": len(self.chunks),
            "created_at": self.created_at,
        }
        if include_chunks:
            d["chunks"] = [c.to_dict() for c in self.chunks]
        return d


@dataclass
class KnowledgeSession:
    """In-memory session holding ingested sources and their chunk indexes."""

    session_id: str
    sources: Dict[str, KnowledgeSource] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def total_bytes(self) -> int:
        return sum(len(s.content.encode("utf-8")) for s in self.sources.values())

    def get_all_chunks(self) -> List[KnowledgeChunk]:
        chunks: List[KnowledgeChunk] = []
        for s in self.sources.values():
            chunks.extend(s.chunks)
        return chunks


def chunk_document_bpe(
    text: str,
    source_id: str,
    tokenizer: Optional[BPETokenizer] = None,
    target_min_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
    target_max_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
) -> List[KnowledgeChunk]:
    """Segment document text into token-budgeted chunks preserving exact substring offsets.

    Guarantees:
        text[chunk.start_char : chunk.end_char] == chunk.text
    """
    if not text or not text.strip():
        return []

    tok = tokenizer or get_default_tokenizer()

    def count_tokens(s: str) -> int:
        return len(tok.encode(s))

    # 1. Identify sentence/line boundaries while preserving character offsets
    raw_spans: List[Tuple[int, int]] = []
    for m in re.finditer(r"[^.!?\n]+(?:[.!?]+|\n+|\Z)", text):
        raw = m.group(0)
        stripped = raw.strip()
        if stripped:
            lead = len(raw) - len(raw.lstrip())
            trail = len(raw) - len(raw.rstrip())
            s = m.start() + lead
            e = m.end() - trail
            if e > s:
                raw_spans.append((s, e))

    if not raw_spans:
        # Fallback if no sentence boundaries found: take entire stripped content
        s = len(text) - len(text.lstrip())
        e = len(text) - len(text.rstrip())
        if e > s:
            raw_spans = [(s, e)]
        else:
            return []

    # 2. Divide any individual spans that exceed target_max_tokens on word boundaries
    spans: List[Tuple[int, int]] = []
    for s, e in raw_spans:
        span_str = text[s:e]
        tok_count = count_tokens(span_str)
        if tok_count > target_max_tokens:
            for wm in re.finditer(r"\S+", span_str):
                w_start = s + wm.start()
                w_end = s + wm.end()
                spans.append((w_start, w_end))
        else:
            spans.append((s, e))

    if not spans:
        return []

    # 3. Group contiguous spans into chunks targeting 30 to 70 tokens
    chunks: List[KnowledgeChunk] = []
    curr_start = spans[0][0]
    curr_end = spans[0][1]

    for s, e in spans[1:]:
        cand_text = text[curr_start:e]
        cand_tokens = count_tokens(cand_text)
        curr_tokens = count_tokens(text[curr_start:curr_end])

        if cand_tokens <= target_max_tokens:
            curr_end = e
        else:
            if curr_tokens >= target_min_tokens or cand_tokens > (target_max_tokens + 15):
                chunk_str = text[curr_start:curr_end]
                ch_hash = hashlib.sha256(chunk_str.encode("utf-8")).hexdigest()
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=f"{source_id}_chunk_{len(chunks)}",
                        source_id=source_id,
                        text=chunk_str,
                        start_char=curr_start,
                        end_char=curr_end,
                        token_count=curr_tokens,
                        content_hash=ch_hash,
                    )
                )
                curr_start = s
                curr_end = e
            else:
                curr_end = e

    # Emit final trailing chunk
    final_str = text[curr_start:curr_end]
    final_tokens = count_tokens(final_str)
    final_hash = hashlib.sha256(final_str.encode("utf-8")).hexdigest()
    chunks.append(
        KnowledgeChunk(
            chunk_id=f"{source_id}_chunk_{len(chunks)}",
            source_id=source_id,
            text=final_str,
            start_char=curr_start,
            end_char=curr_end,
            token_count=final_tokens,
            content_hash=final_hash,
        )
    )

    return chunks


class KnowledgeStore:
    """Thread-safe, in-memory knowledge repository scoped by session."""

    def __init__(self, tokenizer: Optional[BPETokenizer] = None) -> None:
        self.tokenizer = tokenizer or get_default_tokenizer()
        self._sessions: Dict[str, KnowledgeSession] = {}
        self._lock = threading.RLock()

    def get_or_create_session(self, session_id: str) -> KnowledgeSession:
        with self._lock:
            sid = session_id.strip() if session_id else "default"
            if sid not in self._sessions:
                self._sessions[sid] = KnowledgeSession(session_id=sid)
            return self._sessions[sid]

    def add_source(
        self,
        session_id: str,
        display_name: str,
        content: str,
        source_type: str = "pasted",
    ) -> KnowledgeSource:
        """Validate, chunk, and register a user document into the specified session."""
        with self._lock:
            # 1. Validation: content must be non-empty and non-binary
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Document content cannot be empty or whitespace-only.")
            if "\x00" in content:
                raise ValueError("Document contains invalid binary data / null bytes.")

            content_bytes = content.encode("utf-8")
            doc_size = len(content_bytes)
            if doc_size > MAX_SOURCE_BYTES:
                raise ValueError(
                    f"Document size ({doc_size} bytes) exceeds maximum limit of {MAX_SOURCE_BYTES} bytes (16 KB)."
                )

            session = self.get_or_create_session(session_id)
            if len(session.sources) >= MAX_SOURCES_PER_SESSION:
                raise ValueError(
                    f"Session '{session.session_id}' has reached the maximum limit of {MAX_SOURCES_PER_SESSION} sources."
                )

            current_session_bytes = session.total_bytes()
            if current_session_bytes + doc_size > MAX_SESSION_BYTES:
                raise ValueError(
                    f"Adding document would exceed total session limit of {MAX_SESSION_BYTES} bytes (64 KB). "
                    f"Current usage: {current_session_bytes} bytes."
                )

            # 2. Sanitize display name and source type
            clean_name = sanitize_display_name(display_name)
            st = source_type.lower().strip()
            if st not in ("pasted", "txt", "md"):
                st = "pasted"

            # 3. Create unique source ID and hash
            content_hash = hashlib.sha256(content_bytes).hexdigest()
            source_id = f"src_{hashlib.sha256((session.session_id + clean_name + content_hash[:8]).encode('utf-8')).hexdigest()[:12]}"

            # 4. Compute token count and chunks
            tok_count = len(self.tokenizer.encode(content))
            chunks = chunk_document_bpe(
                text=content,
                source_id=source_id,
                tokenizer=self.tokenizer,
            )

            # Strict verification of chunk substring invariant
            for c in chunks:
                if content[c.start_char : c.end_char] != c.text:
                    raise RuntimeError(
                        f"Chunk substring invariant violation in source '{clean_name}': "
                        f"offset [{c.start_char}:{c.end_char}] does not match chunk text."
                    )

            source = KnowledgeSource(
                source_id=source_id,
                session_id=session.session_id,
                display_name=clean_name,
                source_type=st,
                content=content,
                content_hash=content_hash,
                character_count=len(content),
                token_count=tok_count,
                created_at=datetime.now(timezone.utc).isoformat(),
                chunks=chunks,
            )

            session.sources[source_id] = source
            logger.info(
                f"Ingested knowledge source '{clean_name}' (id: {source_id}) with {len(chunks)} chunks "
                f"into session '{session.session_id}'."
            )
            return source

    def get_source(self, session_id: str, source_id: str) -> Optional[KnowledgeSource]:
        with self._lock:
            session = self.get_or_create_session(session_id)
            return session.sources.get(source_id)

    def list_sources(self, session_id: str) -> List[KnowledgeSource]:
        with self._lock:
            session = self.get_or_create_session(session_id)
            return list(session.sources.values())

    def delete_source(self, session_id: str, source_id: str) -> bool:
        with self._lock:
            session = self.get_or_create_session(session_id)
            if source_id in session.sources:
                del session.sources[source_id]
                return True
            return False

    def clear_session(self, session_id: str) -> bool:
        with self._lock:
            session = self.get_or_create_session(session_id)
            session.sources.clear()
            return True

    def get_all_chunks(self, session_id: str) -> List[KnowledgeChunk]:
        with self._lock:
            session = self.get_or_create_session(session_id)
            return session.get_all_chunks()


# Global in-memory singleton
knowledge_store = KnowledgeStore()
