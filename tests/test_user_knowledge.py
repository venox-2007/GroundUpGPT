"""Unit tests for user knowledge ingestion, chunking, and session store."""

import pytest

from src.user_knowledge import (
    KnowledgeStore,
    chunk_document_bpe,
    sanitize_display_name,
    MAX_SOURCE_BYTES,
    MAX_SOURCES_PER_SESSION,
    MAX_SESSION_BYTES,
)


@pytest.fixture
def store() -> KnowledgeStore:
    return KnowledgeStore()


def test_sanitize_display_name():
    """Test sanitization of filenames and titles."""
    assert sanitize_display_name("notes.txt") == "notes.txt"
    assert sanitize_display_name("../../etc/passwd") == "passwd"
    assert sanitize_display_name("C:\\Users\\Admin\\secret.md") == "secret.md"
    assert sanitize_display_name("hello\x00world.txt") == "helloworld.txt"
    assert sanitize_display_name("") == "Untitled Document"
    assert sanitize_display_name("   ") == "Untitled Document"
    assert sanitize_display_name("..") == "Untitled Document"


def test_chunking_exact_substring_invariant(store: KnowledgeStore):
    """Chunk offsets must match source content exactly across all chunks."""
    sample_text = (
        "GroundUpGPT is a decoder-only Transformer built from scratch.\n\n"
        "It features 27 million parameters trained over 5,949 steps. "
        "The model uses a byte-pair encoding tokenizer with 4,096 vocabulary tokens.\n\n"
        "Supervised fine-tuning was performed using anchor conversational pairs and Dolly examples. "
        "Deterministic tools provide exact arithmetic calculations without hallucination."
    )

    chunks = chunk_document_bpe(sample_text, source_id="test_doc", tokenizer=store.tokenizer)
    assert len(chunks) >= 2

    for c in chunks:
        # Crucial invariant check
        assert sample_text[c.start_char : c.end_char] == c.text
        assert c.token_count > 0
        assert len(c.text.strip()) > 0
        assert c.source_id == "test_doc"


def test_chunking_oversized_text(store: KnowledgeStore):
    """Large sentences must be subdivided into smaller chunks while preserving substring offsets."""
    long_sentence = " ".join([f"item_{i}" for i in range(200)])
    chunks = chunk_document_bpe(long_sentence, source_id="long_doc", tokenizer=store.tokenizer)

    assert len(chunks) > 1
    for c in chunks:
        assert long_sentence[c.start_char : c.end_char] == c.text


def test_store_add_get_delete_source(store: KnowledgeStore):
    """Verify end-to-end ingestion, retrieval, and deletion in a session."""
    sid = "session_test_1"
    doc_content = "This is a verified test document for GroundUpGPT knowledge testing."

    src = store.add_source(
        session_id=sid,
        display_name="test_guide.md",
        content=doc_content,
        source_type="md",
    )

    assert src.display_name == "test_guide.md"
    assert src.source_type == "md"
    assert src.character_count == len(doc_content)
    assert len(src.chunks) >= 1
    assert doc_content[src.chunks[0].start_char : src.chunks[0].end_char] == src.chunks[0].text

    # Retrieve source
    fetched = store.get_source(sid, src.source_id)
    assert fetched is not None
    assert fetched.source_id == src.source_id
    assert fetched.content == doc_content

    # List sources
    sources = store.list_sources(sid)
    assert len(sources) == 1
    assert sources[0].source_id == src.source_id

    # Delete source
    deleted = store.delete_source(sid, src.source_id)
    assert deleted is True
    assert store.get_source(sid, src.source_id) is None
    assert len(store.list_sources(sid)) == 0


def test_store_size_and_count_limits(store: KnowledgeStore):
    """Verify strict rejection of oversized documents, sessions, and counts."""
    sid = "session_limits"

    # 1. Reject binary null byte
    with pytest.raises(ValueError, match="binary"):
        store.add_source(sid, "binary.bin", "hello\x00world")

    # 2. Reject empty content
    with pytest.raises(ValueError, match="empty"):
        store.add_source(sid, "empty.txt", "   \n\t  ")

    # 3. Reject oversized document (> 16 KB)
    big_content = "x" * (MAX_SOURCE_BYTES + 10)
    with pytest.raises(ValueError, match="exceeds maximum limit of 16384 bytes"):
        store.add_source(sid, "oversized.txt", big_content)

    # 4. Enforce max sources limit (10 sources)
    for i in range(MAX_SOURCES_PER_SESSION):
        store.add_source(sid, f"doc_{i}.txt", f"Content for document {i} with some tokens.")

    assert len(store.list_sources(sid)) == MAX_SOURCES_PER_SESSION

    with pytest.raises(ValueError, match="reached the maximum limit of 10 sources"):
        store.add_source(sid, "doc_11.txt", "Eleventh document content.")

    # 5. Clear session
    assert store.clear_session(sid) is True
    assert len(store.list_sources(sid)) == 0
