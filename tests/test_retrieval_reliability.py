"""Unit tests for retrieval reliability, entity disambiguation, and hallucination guardrails."""

import pytest
from starlette.testclient import TestClient

from backend.app import app
from src.user_knowledge import KnowledgeStore, budget_context_bpe
from src.user_knowledge_retrieval import (
    UserKnowledgeRetriever,
    extract_query_entities,
    detect_contradiction,
)
from src.generation import format_sft_prompt


@pytest.fixture
def store() -> KnowledgeStore:
    return KnowledgeStore()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_entity_extraction_logic():
    """Verify clean extraction of proper nouns and mission/project names."""
    q1 = "What is Project Orion?"
    ents1 = extract_query_entities(q1)
    assert "orion" in ents1

    q2 = "When did Apollo 11 land on the Moon?"
    ents2 = extract_query_entities(q2)
    assert "apollo" in ents2
    assert "moon" in ents2
    assert "when" not in ents2

    q3 = "Explain the difference between Team Alpha and Team Beta."
    ents3 = extract_query_entities(q3)
    assert "alpha" in ents3
    assert "beta" in ents3


def test_entity_disambiguation_orion_vs_aurora(store: KnowledgeStore):
    """BM25 must never confuse Project Orion with Project Aurora even with shared terms."""
    sid = "disambig_session"
    store.add_source(
        session_id=sid,
        display_name="project_aurora.txt",
        content="Project Aurora was created by researchers in 2020 for quantum simulation software framework development.",
    )
    store.add_source(
        session_id=sid,
        display_name="project_orion.txt",
        content="Project Orion was created by aerospace engineers in 2021 for deep space propulsion and exploration.",
    )

    retriever = UserKnowledgeRetriever.from_session(store.get_or_create_session(sid))

    # Query for Orion
    res_orion = retriever.retrieve("What was Project Orion created for?")
    assert res_orion.status == "relevant_passage"
    assert res_orion.source_display_name == "project_orion.txt"
    assert "propulsion" in (res_orion.chunk.text if res_orion.chunk else "")

    # Query for Aurora
    res_aurora = retriever.retrieve("What was Project Aurora created for?")
    assert res_aurora.status == "relevant_passage"
    assert res_aurora.source_display_name == "project_aurora.txt"
    assert "simulation" in (res_aurora.chunk.text if res_aurora.chunk else "")


def test_conflict_detection_conflicting_dates(store: KnowledgeStore):
    """Detect conflicting years across multiple sources for the same entity."""
    sid = "conflict_session_dates"
    store.add_source(
        session_id=sid,
        display_name="source_v1.txt",
        content="Project Chronos officially launched in 2021 as confirmed by the steering committee.",
    )
    store.add_source(
        session_id=sid,
        display_name="source_v2.txt",
        content="Project Chronos officially launched in 2024 according to subsequent program revisions.",
    )

    retriever = UserKnowledgeRetriever.from_session(store.get_or_create_session(sid))
    res = retriever.retrieve("In what year did Project Chronos officially launch?")

    assert res.status == "conflicting_evidence"
    assert res.conflict_detected is True
    assert res.conflict_details is not None
    assert res.conflict_details["type"] == "conflicting_dates"
    assert "2021" in str(res.conflict_details["values_1"])
    assert "2024" in str(res.conflict_details["values_2"])


def test_conflict_detection_conflicting_teams(store: KnowledgeStore):
    """Detect conflicting attribution across multiple sources."""
    sid = "conflict_session_teams"
    store.add_source(
        session_id=sid,
        display_name="report_a.txt",
        content="The neural engine for Project Titan was built entirely by Team Alpha.",
    )
    store.add_source(
        session_id=sid,
        display_name="report_b.txt",
        content="The neural engine for Project Titan was built entirely by Team Gamma.",
    )

    retriever = UserKnowledgeRetriever.from_session(store.get_or_create_session(sid))
    res = retriever.retrieve("Which team built the neural engine for Project Titan?")

    assert res.status == "conflicting_evidence"
    assert res.conflict_detected is True
    assert res.conflict_details["type"] == "conflicting_teams"


def test_budget_context_bpe(store: KnowledgeStore):
    """Ensure context budgeting strictly guarantees <= 190 total prompt tokens at sentence boundaries."""
    prompt = "What are the core capabilities of GroundUpGPT?"
    huge_context = (
        "GroundUpGPT features 27 million parameters trained from scratch. "
        "It supports byte-pair encoding with 4096 vocabulary tokens. "
        "The model contains 8 Transformer layers and 8 attention heads. "
        "Pretraining processed over 50 million clean tokens from Project Gutenberg and Simple English Wikipedia. "
        "Supervised fine-tuning used verified short-answer conversational anchors. "
        "SafeCalculator evaluates deterministic mathematical expressions without neural hallucination. "
        "LocalFactRetriever answers verified geographic and scientific questions. "
        "The system runs entirely locally on an NVIDIA RTX 3050 Laptop GPU with 6GB VRAM. "
    ) * 10

    budgeted = budget_context_bpe(
        prompt=prompt,
        context=huge_context,
        tokenizer=store.tokenizer,
        max_total_prompt_tokens=190,
    )

    formatted = format_sft_prompt(prompt, context=budgeted)
    toks = len(store.tokenizer.encode(formatted))
    assert toks <= 190
    assert len(budgeted) > 100
    # Must end at sentence period or word boundary
    assert budgeted.rstrip().endswith(".") or budgeted.rstrip().endswith("...")


def test_grounding_guardrail_api_halts_on_conflict(client: TestClient):
    """In grounded mode, conflicting evidence must trigger guardrail and halt generation."""
    sid = "api_conflict_session"
    client.post(
        "/api/knowledge/source",
        json={
            "session_id": sid,
            "display_name": "memo1.txt",
            "content": "Project Phoenix was completed in 2020 by lead engineer Davis.",
        },
    )
    client.post(
        "/api/knowledge/source",
        json={
            "session_id": sid,
            "display_name": "memo2.txt",
            "content": "Project Phoenix was completed in 2023 by lead engineer Davis.",
        },
    )

    resp = client.post(
        "/api/assistant/respond",
        json={
            "prompt": "When was Project Phoenix completed?",
            "mode": "assisted",
            "knowledge_mode": "grounded",
            "knowledge_session_id": sid,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["origin"] == "document_excerpt"
    assert data["model_generated"] is False
    assert "Grounding Guardrail: Conflicting Evidence Detected" in data["answer_text"]
    assert data["error_code"] == "conflicting_evidence"
    assert data["details"]["conflict_detected"] is True
