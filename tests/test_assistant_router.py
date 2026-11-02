"""Unit tests for GroundUpGPT assistant router."""

import pytest

from backend.assistant_router import (
    AssistantRouter,
    AssistantRouteDecision,
    MODE_NATIVE,
    MODE_ASSISTED,
    ORIGIN_NEURAL,
    ORIGIN_SAFE_CALCULATOR,
    ORIGIN_LOCAL_FACT,
    ORIGIN_TOOL_ERROR,
    CURATED_FACTS,
)


@pytest.fixture
def router() -> AssistantRouter:
    return AssistantRouter()


def test_mode_defaults_and_validation(router: AssistantRouter):
    """Native mode always returns neural origin regardless of prompt content."""
    dec_math = router.classify_and_route("What is 7 + 9?", mode=MODE_NATIVE)
    assert dec_math.origin == ORIGIN_NEURAL
    assert dec_math.mode == MODE_NATIVE
    assert dec_math.model_generated is True
    assert dec_math.is_terminal is False

    dec_fact = router.classify_and_route("What is the capital of Japan?", mode=MODE_NATIVE)
    assert dec_fact.origin == ORIGIN_NEURAL
    assert dec_fact.is_terminal is False

    # Invalid mode rejection
    with pytest.raises(ValueError, match="Invalid mode"):
        router.classify_and_route("Hello", mode="unsupported_mode")


def test_arithmetic_routing_success(router: AssistantRouter):
    """Test valid arithmetic requests in assisted mode."""
    test_cases = [
        ("What is 7 + 9?", "16", "7 + 9"),
        ("64 / 8", "8", "64 / 8"),
        ("Calculate (12 + 4) * 3", "48", "(12 + 4) * 3"),
        ("15 minus 7", "8", "15 - 7"),
        ("100 * 2.5", "250", "100 * 2.5"),
    ]
    for query, expected_answer, _ in test_cases:
        dec = router.classify_and_route(query, mode=MODE_ASSISTED)
        assert dec.origin == ORIGIN_SAFE_CALCULATOR
        assert dec.mode == MODE_ASSISTED
        assert dec.model_generated is False
        assert dec.is_terminal is True
        assert dec.answer_text == expected_answer
        assert dec.tool_name == "SafeCalculator"
        assert dec.evaluated_expression is not None


def test_arithmetic_routing_rejections(router: AssistantRouter):
    """Test arithmetic errors such as division by zero, syntax errors, and overflow."""
    # Division by zero
    dec_div0 = router.classify_and_route("What is 10 / 0?", mode=MODE_ASSISTED)
    assert dec_div0.origin == ORIGIN_TOOL_ERROR
    assert dec_div0.tool_name == "SafeCalculator"
    assert dec_div0.error_code == "division_by_zero"
    assert "division by zero" in dec_div0.answer_text.lower()
    assert dec_div0.model_generated is False

    # Consecutive operators / syntax error
    dec_syntax = router.classify_and_route("12 ++ 4", mode=MODE_ASSISTED)
    assert dec_syntax.origin == ORIGIN_TOOL_ERROR
    assert dec_syntax.error_code == "syntax_error"

    # Extreme magnitude overflow limit
    dec_overflow = router.classify_and_route("What is 1000000000000000000000 + 1", mode=MODE_ASSISTED)
    assert dec_overflow.origin == ORIGIN_TOOL_ERROR


def test_synthetic_facts_strictly_excluded(router: AssistantRouter):
    """Fictional testing records must NEVER be exposed as verified real-world knowledge."""
    # Ensure CURATED_FACTS has zero synthetic records
    for f in CURATED_FACTS:
        assert getattr(f, "category", "") != "synthetic"
        assert "eldoria" not in f.entity.lower()
        assert "xylo" not in f.entity.lower()
        assert "boranic" not in f.entity.lower()
        assert "zorblax" not in f.entity.lower()

    # Query for synthetic capital of Eldoria
    dec_eldoria = router.classify_and_route("What is the capital of Eldoria?", mode=MODE_ASSISTED)
    assert dec_eldoria.origin == ORIGIN_LOCAL_FACT
    assert dec_eldoria.fact_id is None
    assert dec_eldoria.error_code == "unsupported_fact"
    assert "Quellix" not in (dec_eldoria.answer_text or "")
    assert "supported countries are" in (dec_eldoria.answer_text or "").lower()

    # Query for other synthetic records
    dec_xylo = router.classify_and_route("What is Xylo-7?", mode=MODE_ASSISTED)
    assert dec_xylo.origin == ORIGIN_NEURAL  # falls back to neural, does not return synthetic fact


def test_geography_capitals_and_provenance(router: AssistantRouter):
    """Test valid capital city lookups with correct answers and provenance."""
    capitals = [
        ("What is the capital of France?", "Paris", "geo_france_capital"),
        ("What is the capital of Germany?", "Berlin", "geo_germany_capital"),
        ("What is the capital of Japan?", "Tokyo", "geo_japan_capital"),
        ("What is the capital of Australia?", "Canberra", "geo_australia_capital"),
        ("What is the capital of Canada?", "Ottawa", "geo_canada_capital"),
        ("What is the capital of Italy?", "Rome", "geo_italy_capital"),
        ("What is the capital of Spain?", "Madrid", "geo_spain_capital"),
        ("Which city is Japan's capital?", "Tokyo", "geo_japan_capital"),
    ]
    for query, expected_answer, expected_id in capitals:
        dec = router.classify_and_route(query, mode=MODE_ASSISTED)
        assert dec.origin == ORIGIN_LOCAL_FACT
        assert dec.fact_id == expected_id
        assert dec.answer_text == expected_answer
        assert dec.provenance == "curated_groundup_facts_v1"
        assert dec.model_generated is False
        assert dec.is_terminal is True


def test_prevent_cross_entity_and_relation_mismatches(router: AssistantRouter):
    """Prevent returning Tokyo for population of Japan or inventing unsupported capitals."""
    # Relation mismatch: Population of Japan must NOT return Tokyo
    dec_pop = router.classify_and_route("What is the population of Japan?", mode=MODE_ASSISTED)
    assert dec_pop.origin == ORIGIN_LOCAL_FACT
    assert dec_pop.fact_id is None
    assert dec_pop.error_code == "unsupported_fact"
    assert "Tokyo" not in (dec_pop.answer_text or "")
    assert "population" in (dec_pop.answer_text or "").lower()

    # Unsupported country capital: Norway
    dec_norway = router.classify_and_route("What is the capital of Norway?", mode=MODE_ASSISTED)
    assert dec_norway.origin == ORIGIN_LOCAL_FACT
    assert dec_norway.fact_id is None
    assert dec_norway.error_code == "unsupported_fact"

    # Unsupported entity capital: Jupiter
    dec_jupiter = router.classify_and_route("What is the capital of Jupiter?", mode=MODE_ASSISTED)
    assert dec_jupiter.origin == ORIGIN_LOCAL_FACT
    assert dec_jupiter.fact_id is None
    assert dec_jupiter.error_code == "unsupported_fact"

    # Ambiguous multi-country query: France and Germany
    dec_multi = router.classify_and_route("What is the capital of France and Germany?", mode=MODE_ASSISTED)
    assert dec_multi.origin == ORIGIN_LOCAL_FACT
    assert dec_multi.fact_id is None
    assert dec_multi.error_code == "ambiguous_query"


def test_science_and_biology_facts(router: AssistantRouter):
    """Test curated science and biology lookups."""
    facts = [
        ("Is water a solid, liquid, or gas at room temperature?", "liquid", "sci_water_room_temp"),
        ("Is wood a conductor or an insulator?", "insulator", "sci_wood_conductivity"),
        ("Is iron a metal or a non-metal?", "metal", "sci_iron_classification"),
        ("What is the chemical formula for water?", "H2O", "sci_chemical_water"),
        ("What is the chemical symbol for gold?", "Au", "sci_gold_symbol"),
        ("What is the speed of light in a vacuum?", "299,792,458 meters per second", "sci_speed_of_light"),
        ("Name 3 primary colors.", "red, blue, and yellow", "sci_primary_colors"),
        ("Is a whale a fish or a mammal?", "mammal", "bio_whale_mammal"),
        ("Which organelle is responsible for photosynthesis?", "chloroplast", "bio_photosynthesis_organelle"),
    ]
    for query, expected_answer, expected_id in facts:
        dec = router.classify_and_route(query, mode=MODE_ASSISTED)
        assert dec.origin == ORIGIN_LOCAL_FACT
        assert dec.fact_id == expected_id
        assert dec.answer_text == expected_answer
        assert dec.model_generated is False
        assert dec.is_terminal is True


def test_unsupported_science_queries(router: AssistantRouter):
    """Unsupported science queries return appropriate limitation messages."""
    dec_silver = router.classify_and_route("What is the chemical symbol for silver?", mode=MODE_ASSISTED)
    assert dec_silver.origin == ORIGIN_LOCAL_FACT
    assert dec_silver.fact_id is None
    assert dec_silver.error_code == "unsupported_fact"
    assert dec_silver.answer_text != "Au"


def test_conversational_and_creative_fallback(router: AssistantRouter):
    """Conversational, greeting, and creative queries route to neural generation."""
    conversational_queries = [
        "Hello!",
        "Hi there, how are you?",
        "Tell me a short story.",
        "Explain why people dream.",
        "Write a poem about the morning sunrise.",
        "Can you help me brainstorm project ideas?",
    ]
    for q in conversational_queries:
        dec = router.classify_and_route(q, mode=MODE_ASSISTED)
        assert dec.origin == ORIGIN_NEURAL
        assert dec.mode == MODE_ASSISTED
        assert dec.is_terminal is False
        assert dec.model_generated is True
