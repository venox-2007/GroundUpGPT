"""Unit tests for the task-specific conversation evaluation engine.

Verifies:
- Exact arithmetic validation (matching operands, operation, calculated result)
- Rejection of modified operands (e.g. 65 / 8 = 8 for 64 / 8)
- Rejection of incorrect arithmetic results (e.g. 7 + 9 = 10)
- Rejection of contradictory extraneous numbers (e.g. 102 4 years old)
- Rejection of prompt echoing without assertion
- Rejection of circular tautological definitions (e.g. 'A mammal is a mammal')
- Minimum count enforcement on N-item lists (1 or 2 items fail when 3 requested)
- Binary classification assertions and rejection of contradictions
- Multi-item classification completeness
- Open-ended qualitative items properly marked as unscored
"""

from __future__ import annotations

import pytest

from src.conversation_evaluation import (
    TaskType,
    evaluate_arithmetic,
    evaluate_binary_classification,
    evaluate_exact_factual,
    evaluate_multi_item_classification,
    evaluate_n_item_list,
    evaluate_scientific_definition,
    evaluate_text_transformation,
    grade_benchmark_item,
)


def test_exact_arithmetic_success() -> None:
    res = evaluate_arithmetic("What is 2 plus 2?", "2 + 2 = 4.")
    assert res.is_objective is True
    assert res.passed is True
    assert "4" in res.reason


def test_arithmetic_operand_modification_rejection() -> None:
    # Model changed 64 to 65 even though 8 is in response
    res = evaluate_arithmetic("What is 64 divided by 8?", "65 divided by 8 is 8.")
    assert res.is_objective is True
    assert res.passed is False
    assert "modified operands" in res.reason.lower()


def test_arithmetic_incorrect_calculation() -> None:
    res = evaluate_arithmetic("What is 7 + 9?", "7 + 9 = 10.")
    assert res.is_objective is True
    assert res.passed is False
    assert "incorrect calculation" in res.reason.lower()


def test_arithmetic_extraneous_conflicting_numbers() -> None:
    res = evaluate_arithmetic("What is 2 + 2?", "102 4 years old.")
    assert res.is_objective is True
    assert res.passed is False
    assert "contradictory extraneous numbers" in res.reason.lower()


def test_exact_factual_echo_and_tautology_rejection() -> None:
    # Prompt echo
    res_echo = evaluate_exact_factual(
        "What is the capital of Japan?",
        "The capital of Japan",
        target_answer="Tokyo",
    )
    assert res_echo.passed is False
    assert "echo" in res_echo.reason.lower()

    # Tautological repetition
    res_taut = evaluate_exact_factual(
        "What is the capital of Japan?",
        "The capital of Japan's capital of Japan.",
        target_answer="Tokyo",
    )
    assert res_taut.passed is False
    assert "tokyo" in res_taut.reason.lower()

    # Valid assertion
    res_valid = evaluate_exact_factual(
        "What is the capital of Japan?",
        "The capital of Japan is Tokyo.",
        target_answer="Tokyo",
    )
    assert res_valid.passed is True


def test_scientific_definition_circular_rejection() -> None:
    # Circular definition
    res_circ = grade_benchmark_item("bench_10", "What is a mammal?", "A mammal.", "definitions", "frozen_35")
    assert res_circ.passed is False
    assert "circular" in res_circ.reason.lower()

    # Valid substantive definition
    res_valid = grade_benchmark_item(
        "bench_10",
        "What is a mammal?",
        "A mammal is a warm-blooded vertebrate animal with fur that produces milk for its young.",
        "definitions",
        "frozen_35",
    )
    assert res_valid.passed is True

    # Missing core concepts
    res_miss = grade_benchmark_item("bench_13", "What is gravity?", "gravity! What is it?", "definitions", "frozen_35")
    assert res_miss.passed is False


def test_n_item_list_count_enforcement() -> None:
    # 1 color when 3 requested
    res_1 = grade_benchmark_item("bench_28", "List three primary colors.", "red.", "instructions", "frozen_35")
    assert res_1.passed is False
    assert "insufficient items" in res_1.reason.lower()

    # 2 colors when 3 requested
    res_2 = grade_benchmark_item("bench_28", "List three primary colors.", "red and blue.", "instructions", "frozen_35")
    assert res_2.passed is False

    # 3 distinct valid colors
    res_3 = grade_benchmark_item("bench_28", "List three primary colors.", "Red, blue, and yellow.", "instructions", "frozen_35")
    assert res_3.passed is True


def test_binary_and_multi_item_classification() -> None:
    # Binary valid
    res_bin_ok = evaluate_binary_classification("Is wood a conductor or an insulator?", "Wood is an insulator.", "insulator", "conductor")
    assert res_bin_ok.passed is True

    # Binary inverted
    res_bin_bad = evaluate_binary_classification("Is wood a conductor or an insulator?", "Wood is a conductor.", "insulator", "conductor")
    assert res_bin_bad.passed is False

    # Multi-item classification (all 4 required)
    item_cats = {
        "apple": "fruit",
        "banana": "fruit",
        "carrot": "vegetable",
        "broccoli": "vegetable",
    }
    # Partial classification fails
    res_part = evaluate_multi_item_classification(
        "Classify as fruit or vegetable: apple, carrot, banana, broccoli.",
        "Apple is a fruit. Carrot is a vegetable.",
        item_cats,
    )
    assert res_part.passed is False
    assert "2/4" in res_part.reason

    # Exhaustive classification passes
    res_full = evaluate_multi_item_classification(
        "Classify as fruit or vegetable: apple, carrot, banana, broccoli.",
        "Fruits: apple, banana. Vegetables: carrot, broccoli.",
        item_cats,
    )
    assert res_full.passed is True


def test_unscored_qualitative_items() -> None:
    # Open greeting
    res_greet = grade_benchmark_item("bench_01", "Hello! How are you today?", "Hello! I am doing well, thank you.", "greetings", "frozen_35")
    assert res_greet.is_objective is False
    assert res_greet.passed is None
    assert res_greet.requires_qualitative_review is True
