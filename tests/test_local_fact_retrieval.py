"""Unit tests for the local BM25 fact retrieval engine."""

import pytest

from src.local_fact_retrieval import (
    FACT_INVENTORY,
    FactRecord,
    LocalFactRetriever,
    RetrievalResult,
    tokenize_text,
)


def test_tokenization_and_stopwords() -> None:
    """Verify text tokenization strips punctuation, lowercases, and filters stopwords."""
    tokens = tokenize_text("What is the capital of France?")
    assert "capital" in tokens
    assert "france" in tokens
    assert "what" not in tokens
    assert "is" not in tokens
    assert "the" not in tokens
    assert "of" not in tokens


def test_retrieval_hits_on_known_facts() -> None:
    """Verify that known factual questions successfully retrieve corresponding fact records."""
    retriever = LocalFactRetriever()

    # France capital
    res_fr = retriever.retrieve("What is the capital of France?")
    assert isinstance(res_fr, RetrievalResult)
    assert res_fr.status == "match"
    assert res_fr.matched_fact is not None
    assert res_fr.matched_fact.fact_id == "geo_france_capital"
    assert res_fr.matched_fact.verified_answer == "Paris"
    assert res_fr.retrieval_score > 2.0
    assert res_fr.is_tool_computed is True
    assert res_fr.source_provenance == "curated_groundup_facts_v1"

    # Australia capital
    res_au = retriever.retrieve("Which city is Australia's capital?")
    assert res_au.status == "match"
    assert res_au.matched_fact is not None
    assert res_au.matched_fact.verified_answer == "Canberra"

    # Synthetic / fictional entity
    res_syn = retriever.retrieve("What is the capital of Eldoria?")
    assert res_syn.status == "match"
    assert res_syn.matched_fact is not None
    assert res_syn.matched_fact.verified_answer == "Quellix"


def test_retrieval_no_match_on_unknown_queries() -> None:
    """Verify unknown or out-of-domain queries return status='no_match' without hallucinating."""
    retriever = LocalFactRetriever()

    res_unknown = retriever.retrieve("Who was the third emperor of Mars in 2450?")
    assert res_unknown.status == "no_match"
    assert res_unknown.matched_fact is None

    res_empty = retriever.retrieve("??? !!!")
    assert res_empty.status == "no_match"
    assert res_empty.matched_fact is None


def test_ambiguity_detection() -> None:
    """Verify that when multiple distinct facts have near-identical match scores, ambiguity is flagged."""
    # Create two facts sharing identical terms with different entities
    fact_a = FactRecord(
        fact_id="fact_a",
        entity="Entity Alpha",
        category="test",
        search_terms=["sharedterm", "concept"],
        question_forms=["What is sharedterm concept?"],
        verified_answer="Answer A",
        supporting_passage="Sharedterm concept applies to Alpha.",
    )
    fact_b = FactRecord(
        fact_id="fact_b",
        entity="Entity Beta",
        category="test",
        search_terms=["sharedterm", "concept"],
        question_forms=["What is sharedterm concept?"],
        verified_answer="Answer B",
        supporting_passage="Sharedterm concept applies to Beta.",
    )

    retriever = LocalFactRetriever(inventory=[fact_a, fact_b], min_score_threshold=0.5, ambiguity_threshold_ratio=0.20)
    res = retriever.retrieve("What is sharedterm concept?")
    assert res.status == "ambiguous"
    assert res.is_ambiguous is True
    assert res.matched_fact is None
    assert len(res.candidates) >= 2


def test_custom_inventory_support() -> None:
    """Verify custom fact inventory can be instantiated independently."""
    custom_facts = [
        FactRecord(
            fact_id="custom_01",
            entity="GroundUpGPT",
            category="project",
            search_terms=["groundupgpt", "parameters", "27m"],
            question_forms=["How many parameters does GroundUpGPT-27M have?"],
            verified_answer="27,411,456 parameters",
            supporting_passage="GroundUpGPT-27M contains exactly 27,411,456 parameters.",
            source="custom_test_v1",
        )
    ]
    retriever = LocalFactRetriever(inventory=custom_facts)
    res = retriever.retrieve("How many parameters in GroundUpGPT-27M?")
    assert res.status == "match"
    assert res.matched_fact is not None
    assert res.matched_fact.verified_answer == "27,411,456 parameters"
    assert res.source_provenance == "custom_test_v1"
