"""Unit tests for user knowledge BM25 retrieval and classification."""

import pytest

from src.user_knowledge import KnowledgeStore
from src.user_knowledge_retrieval import UserKnowledgeRetriever


@pytest.fixture
def store() -> KnowledgeStore:
    return KnowledgeStore()


def test_retrieval_relevant_passage(store: KnowledgeStore):
    """Accurately match query to specific passage in ingested documents."""
    sid = "retrieval_session_1"
    store.add_source(
        session_id=sid,
        display_name="spaceflight.txt",
        content=(
            "The Apollo program succeeded in landing the first humans on the Moon in 1969. "
            "Neil Armstrong and Buzz Aldrin walked on the lunar surface during Apollo 11.\n\n"
            "The Hubble Space Telescope was launched into low Earth orbit in 1990 and remains in operation."
        ),
    )

    retriever = UserKnowledgeRetriever.from_session(store.get_or_create_session(sid))
    res = retriever.retrieve("When did Apollo land on the Moon?")

    assert res.status == "relevant_passage"
    assert res.chunk is not None
    assert "1969" in res.chunk.text
    assert res.source_display_name == "spaceflight.txt"
    assert res.score >= 1.0
    assert not res.is_ambiguous


def test_retrieval_no_match(store: KnowledgeStore):
    """Return no_match status when query has no matching terms or corpus is empty."""
    sid = "empty_session"
    retriever = UserKnowledgeRetriever.from_session(store.get_or_create_session(sid))
    res = retriever.retrieve("What is quantum computing?")

    assert res.status == "no_match"
    assert res.chunk is None
    assert res.score == 0.0

    # Add unrelated document
    store.add_source(sid, "animals.txt", "Elephants are large mammals of the family Elephantidae.")
    retriever = UserKnowledgeRetriever.from_session(store.get_or_create_session(sid))
    res2 = retriever.retrieve("quantum entanglement")
    assert res2.status == "no_match"
    assert res2.chunk is None


def test_retrieval_ambiguous_evidence(store: KnowledgeStore):
    """Detect ambiguous evidence when two documents contain equally matching passages."""
    sid = "ambiguity_session"
    store.add_source(
        session_id=sid,
        display_name="team_alpha.txt",
        content="Project Orion was completed in 2021 by Team Alpha using Python and PyTorch.",
    )
    store.add_source(
        session_id=sid,
        display_name="team_beta.txt",
        content="Project Orion was completed in 2021 by Team Beta using Python and Rust.",
    )

    retriever = UserKnowledgeRetriever.from_session(store.get_or_create_session(sid))
    res = retriever.retrieve("Who completed Project Orion in 2021?")

    assert res.status == "ambiguous_evidence"
    assert res.is_ambiguous is True
    assert len(res.candidates) >= 2
