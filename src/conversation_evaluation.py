"""Task-specific, objective conversational evaluation engine for GroundUpGPT.

Replaces loose word-matching heuristics with rigorous, category-specific evaluation rubrics:
- Exact factual QA (requires verified entity assertion; rejects echoing and tautology)
- Numeric arithmetic (validates question operands, operation, and computed result; rejects operand modifications)
- Scientific definitions (requires substantive explanatory concepts; rejects circular definitions)
- Binary classification (verifies asserted category and absence of opposing category)
- Multi-item classification (verifies exhaustive placement of all requested items into correct categories)
- Requested N-item lists (requires at least N distinct valid members; rejects partial lists)
- Deterministic text transformations (verifies exact target string transformations)
- Open-ended conversational responses (marked explicitly as requiring qualitative review)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


class TaskType(str, Enum):
    """Categorical classification of evaluation tasks."""
    EXACT_FACTUAL_QA = "exact_factual_qa"
    NUMERIC_ARITHMETIC = "numeric_arithmetic"
    SCIENTIFIC_DEFINITIONS = "scientific_definitions"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTI_ITEM_CLASSIFICATION = "multi_item_classification"
    REQUESTED_N_ITEM_LISTS = "requested_n_item_lists"
    DETERMINISTIC_TEXT_TRANSFORMATION = "deterministic_text_transformation"
    OPEN_ENDED_CONVERSATIONAL = "open_ended_conversational"


@dataclass
class EvaluationResult:
    """Structured evaluation output for an evaluated response."""
    task_type: TaskType
    is_objective: bool
    passed: Optional[bool]
    reason: str
    requires_qualitative_review: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type.value,
            "is_objective": self.is_objective,
            "passed": self.passed,
            "reason": self.reason,
            "requires_qualitative_review": self.requires_qualitative_review,
            "details": self.details,
        }


def _clean_text(text: str) -> str:
    """Strip whitespace and delimiters."""
    cleaned = text.strip()
    for delim in ("<|endoftext|>", "<|assistant|>", "<|user|>"):
        cleaned = cleaned.replace(delim, "").strip()
    return cleaned


def _is_prompt_echo(prompt: str, response: str) -> bool:
    """Detect if response merely echoes the prompt without answering."""
    norm_p = re.sub(r"[^\w\s]", "", prompt.lower()).strip()
    norm_r = re.sub(r"[^\w\s]", "", response.lower()).strip()
    if not norm_r:
        return True
    if norm_r == norm_p:
        return True
    if len(norm_p) > 10 and norm_r.startswith(norm_p) and len(norm_r) < len(norm_p) + 12:
        return True
    if len(norm_r) > 8 and norm_r in norm_p:
        return True
    return False


def _is_circular_tautology(term: str, response: str) -> bool:
    """Detect circular definitions like 'A mammal is a mammal' or 'Tokyo's capital is Tokyo'."""
    norm_r = response.lower().strip()
    norm_t = term.lower().strip()
    # e.g., "the capital of japan's capital of japan"
    if norm_r.count(norm_t) >= 2 and len(norm_r) < len(norm_t) * 4:
        return True
    # e.g., "a mammal." or "mammal"
    words = re.findall(r"\b\w+\b", norm_r)
    if set(words) <= {"a", "an", "the", "is", norm_t}:
        return True
    return False


# ==============================================================================
# 1. Numeric Arithmetic Evaluator
# ==============================================================================

def parse_arithmetic_problem(prompt: str) -> Optional[Tuple[float, float, str, float]]:
    """Parse operands, operator, and expected result from supported arithmetic prompt formats.

    Returns:
        (operand1, operand2, operator_symbol, expected_result) or None if unsupported.
    """
    clean_p = prompt.strip().replace("?", "").replace(",", "")

    # Pattern 1: "What is X plus Y" or "X + Y"
    m_plus = re.search(r"\b(\d+)\s*(?:\+|\bplus\b)\s*(\d+)\b", clean_p, re.IGNORECASE)
    if m_plus:
        op1, op2 = float(m_plus.group(1)), float(m_plus.group(2))
        return (op1, op2, "+", op1 + op2)

    # Pattern 2: "What is X minus Y" or "X - Y"
    m_minus = re.search(r"\b(\d+)\s*(?:-|\bminus\b)\s*(\d+)\b", clean_p, re.IGNORECASE)
    if m_minus:
        op1, op2 = float(m_minus.group(1)), float(m_minus.group(2))
        return (op1, op2, "-", op1 - op2)

    # Pattern 3: "What is X multiplied by Y" or "X times Y" or "X * Y"
    m_mult = re.search(r"\b(\d+)\s*(?:\*|\btimes\b|\bmultiplied by\b)\s*(\d+)\b", clean_p, re.IGNORECASE)
    if m_mult:
        op1, op2 = float(m_mult.group(1)), float(m_mult.group(2))
        return (op1, op2, "*", op1 * op2)

    # Pattern 4: "What is X divided by Y" or "X / Y"
    m_div = re.search(r"\b(\d+)\s*(?:/|\bdivided by\b)\s*(\d+)\b", clean_p, re.IGNORECASE)
    if m_div:
        op1, op2 = float(m_div.group(1)), float(m_div.group(2))
        if op2 != 0:
            return (op1, op2, "/", op1 / op2)

    # Word Problem 1: "train travels at X miles per hour, how far does it travel in Y hours?"
    m_train = re.search(r"(\d+)\s*miles per hour.*in\s*(\d+)\s*hours", clean_p, re.IGNORECASE)
    if m_train:
        op1, op2 = float(m_train.group(1)), float(m_train.group(2))
        return (op1, op2, "*", op1 * op2)

    # Word Problem 2: "have X apples and give away Y, how many do you have left?"
    m_apples = re.search(r"have\s*(\d+)\s*apples.*give away\s*(\d+)", clean_p, re.IGNORECASE)
    if m_apples:
        op1, op2 = float(m_apples.group(1)), float(m_apples.group(2))
        return (op1, op2, "-", op1 - op2)

    return None


def evaluate_arithmetic(prompt: str, response: str) -> EvaluationResult:
    """Strictly evaluate arithmetic expressions and equations.

    Rules:
    - Parses question operands and operator.
    - Computes ground truth deterministically using Python math.
    - If response states an equation (e.g. 'A op B = C' or 'A divided by B is C'),
      operands A and B MUST match prompt operands.
    - If model modified an operand (e.g. '65 divided by 8 is 8' for '64 / 8'), it FAILS immediately.
    - Rejects echoing prompt numbers without answering.
    - Rejects conflicting extraneous numbers.
    """
    clean_r = _clean_text(response)
    if not clean_r:
        return EvaluationResult(
            task_type=TaskType.NUMERIC_ARITHMETIC,
            is_objective=True,
            passed=False,
            reason="Empty or delimiter-only response",
        )

    parsed = parse_arithmetic_problem(prompt)
    if parsed is None:
        return EvaluationResult(
            task_type=TaskType.NUMERIC_ARITHMETIC,
            is_objective=False,
            passed=None,
            reason="Arithmetic problem format beyond parser capabilities (requires qualitative review)",
            requires_qualitative_review=True,
        )

    op1, op2, op_sym, expected_res = parsed
    exp_int = int(expected_res) if expected_res.is_integer() else expected_res
    exp_str = str(exp_int)

    # 1. Check for equation with operands in response: e.g. "X op Y = Z" or "X op Y is Z"
    # Matches: "25 + 37 = 62", "48 divided by 6 is 8", "17 + 15 = 15"
    eq_pattern = r"(\d+)\s*([\+\-\*\/]|plus|minus|times|multiplied by|divided by)\s*(\d+)\s*(?:=|is|equals)\s*(\d+)"
    eq_matches = re.findall(eq_pattern, clean_r, re.IGNORECASE)

    if eq_matches:
        for r_op1_str, r_op_str, r_op2_str, r_ans_str in eq_matches:
            r_op1, r_op2, r_ans = float(r_op1_str), float(r_op2_str), float(r_ans_str)
            # Verify operand matching
            if r_op1 != op1 or r_op2 != op2:
                return EvaluationResult(
                    task_type=TaskType.NUMERIC_ARITHMETIC,
                    is_objective=True,
                    passed=False,
                    reason=f"Model modified operands in equation: asserted ({r_op1_str}, {r_op2_str}) instead of ({int(op1)}, {int(op2)})",
                    details={"prompt_operands": [op1, op2], "response_operands": [r_op1, r_op2]},
                )
            # Verify calculated result
            if r_ans == expected_res:
                return EvaluationResult(
                    task_type=TaskType.NUMERIC_ARITHMETIC,
                    is_objective=True,
                    passed=True,
                    reason=f"Correct arithmetic equation verified: {r_op1_str} {op_sym} {r_op2_str} = {r_ans_str}",
                    details={"equation": f"{r_op1_str} {op_sym} {r_op2_str} = {r_ans_str}"},
                )
            else:
                return EvaluationResult(
                    task_type=TaskType.NUMERIC_ARITHMETIC,
                    is_objective=True,
                    passed=False,
                    reason=f"Incorrect calculation in equation: {r_op1_str} {op_sym} {r_op2_str} = {r_ans_str} (expected {exp_str})",
                    details={"asserted": r_ans, "expected": expected_res},
                )

    # 2. Check standalone concise number or sentence assertion
    # Look for all numbers in response
    num_matches = re.findall(r"\b\d+\b", clean_r)
    prompt_num_set = {str(int(op1)), str(int(op2))}
    if str(int(exp_int)) in prompt_num_set:
        # e.g. "64 divided by 8 is 8" -> 8 is both operand and result
        prompt_num_set = {str(int(op1))}

    # Filter out numbers that were part of the prompt
    new_numbers = [n for n in num_matches if n not in prompt_num_set]

    if not new_numbers:
        # Check word forms (e.g. 'four', 'eight')
        word_digits = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
            "ten": "10", "sixteen": "16", "twenty": "20"
        }
        words = re.findall(r"\b[a-z]+\b", clean_r.lower())
        matched_words = [word_digits[w] for w in words if w in word_digits]
        if exp_str in matched_words:
            return EvaluationResult(
                task_type=TaskType.NUMERIC_ARITHMETIC,
                is_objective=True,
                passed=True,
                reason=f"Correct numeric answer word '{exp_str}' matched",
            )
        return EvaluationResult(
            task_type=TaskType.NUMERIC_ARITHMETIC,
            is_objective=True,
            passed=False,
            reason=f"No answer number asserted in response (expected {exp_str})",
        )

    # If expected result is in new numbers
    if exp_str in new_numbers:
        # Check for contradictory extra numbers
        contradictory = [n for n in new_numbers if n != exp_str]
        if contradictory:
            # e.g., "102 4 years old"
            return EvaluationResult(
                task_type=TaskType.NUMERIC_ARITHMETIC,
                is_objective=True,
                passed=False,
                reason=f"Contradictory extraneous numbers present: {contradictory}",
                details={"numbers": num_matches},
            )
        return EvaluationResult(
            task_type=TaskType.NUMERIC_ARITHMETIC,
            is_objective=True,
            passed=True,
            reason=f"Correct numeric answer '{exp_str}' asserted",
        )

    return EvaluationResult(
        task_type=TaskType.NUMERIC_ARITHMETIC,
        is_objective=True,
        passed=False,
        reason=f"Incorrect answer numbers asserted: {new_numbers} (expected {exp_str})",
        details={"numbers": new_numbers, "expected": exp_str},
    )


# ==============================================================================
# 2. Exact Factual QA Evaluator
# ==============================================================================

def evaluate_exact_factual(
    prompt: str,
    response: str,
    target_answer: str,
    acceptable_equivalents: Optional[List[str]] = None,
    prohibited_answers: Optional[List[str]] = None,
    aliases: Optional[List[str]] = None,
    prohibited: Optional[List[str]] = None,
) -> EvaluationResult:
    """Strict evaluation of single-target factual QA items."""
    clean_r = _clean_text(response)
    if not clean_r:
        return EvaluationResult(
            task_type=TaskType.EXACT_FACTUAL_QA,
            is_objective=True,
            passed=False,
            reason="Empty or delimiter-only response",
        )

    prohib = prohibited or prohibited_answers or []
    equivs = aliases or acceptable_equivalents or []

    # 1. Echo rejection
    if _is_prompt_echo(prompt, clean_r):
        return EvaluationResult(
            task_type=TaskType.EXACT_FACTUAL_QA,
            is_objective=True,
            passed=False,
            reason="Echoed prompt without asserting an answer",
        )

    # 2. Circular tautology rejection (e.g. "The capital of Japan's capital of Japan")
    if _is_circular_tautology(target_answer, clean_r):
        return EvaluationResult(
            task_type=TaskType.EXACT_FACTUAL_QA,
            is_objective=True,
            passed=False,
            reason=f"Circular tautological repetition of '{target_answer}' without factual assertion",
        )

    all_targets = [target_answer] + equivs

    # Check for prohibited answers (e.g. asserting 'Kyoto' or 'Osaka' for Japan capital)
    if prohib:
        for p_ans in prohib:
            if re.search(r"\b" + re.escape(p_ans.lower()) + r"\b", clean_r.lower()):
                return EvaluationResult(
                    task_type=TaskType.EXACT_FACTUAL_QA,
                    is_objective=True,
                    passed=False,
                    reason=f"Asserted incorrect/prohibited entity: '{p_ans}'",
                )

    # Check for correct target with word boundary
    for tgt in all_targets:
        pattern = r"\b" + re.escape(tgt.lower()) + r"\b"
        if re.search(pattern, clean_r.lower()):
            # Rejection of uncertain question-mark outputs: "Tokyo or Kyoto?"
            if " or " in clean_r.lower() and "?" in clean_r:
                return EvaluationResult(
                    task_type=TaskType.EXACT_FACTUAL_QA,
                    is_objective=True,
                    passed=False,
                    reason="Uncertain alternative question output containing target",
                )
            return EvaluationResult(
                task_type=TaskType.EXACT_FACTUAL_QA,
                is_objective=True,
                passed=True,
                reason=f"Correct factual target '{tgt}' matched",
            )

    return EvaluationResult(
        task_type=TaskType.EXACT_FACTUAL_QA,
        is_objective=True,
        passed=False,
        reason=f"Target entity '{target_answer}' not found in response",
        details={"acceptable_targets": all_targets},
    )


# ==============================================================================
# 3. Binary Classification Evaluator
# ==============================================================================

def evaluate_binary_classification(
    prompt: str,
    response: str,
    correct_category: str,
    incorrect_category: str,
) -> EvaluationResult:
    """Evaluate binary classification (e.g. reptile vs amphibian, conductor vs insulator)."""
    clean_r = _clean_text(response)
    if not clean_r:
        return EvaluationResult(
            task_type=TaskType.BINARY_CLASSIFICATION,
            is_objective=True,
            passed=False,
            reason="Empty or delimiter-only response",
        )

    norm_r = clean_r.lower()
    has_correct = bool(re.search(r"\b" + re.escape(correct_category.lower()) + r"\b", norm_r))
    has_incorrect = bool(re.search(r"\b" + re.escape(incorrect_category.lower()) + r"\b", norm_r))

    if has_correct and not has_incorrect:
        return EvaluationResult(
            task_type=TaskType.BINARY_CLASSIFICATION,
            is_objective=True,
            passed=True,
            reason=f"Correctly classified as '{correct_category}'",
        )
    elif has_correct and has_incorrect:
        # Check if model resolved it: e.g. "Wood is an insulator, not a conductor."
        if re.search(r"\bis an?\s+" + re.escape(correct_category.lower()), norm_r) and not re.search(r"\bis an?\s+" + re.escape(incorrect_category.lower()), norm_r):
            return EvaluationResult(
                task_type=TaskType.BINARY_CLASSIFICATION,
                is_objective=True,
                passed=True,
                reason=f"Asserted '{correct_category}' despite mentioning comparison",
            )
        return EvaluationResult(
            task_type=TaskType.BINARY_CLASSIFICATION,
            is_objective=True,
            passed=False,
            reason=f"Ambiguous or contradictory classification mentioning both '{correct_category}' and '{incorrect_category}'",
        )
    elif has_incorrect:
        return EvaluationResult(
            task_type=TaskType.BINARY_CLASSIFICATION,
            is_objective=True,
            passed=False,
            reason=f"Incorrect classification: asserted '{incorrect_category}' instead of '{correct_category}'",
        )
    else:
        return EvaluationResult(
            task_type=TaskType.BINARY_CLASSIFICATION,
            is_objective=True,
            passed=False,
            reason=f"Neither category ('{correct_category}' nor '{incorrect_category}') asserted in response",
        )


# ==============================================================================
# 4. Multi-Item Classification Evaluator
# ==============================================================================

def evaluate_multi_item_classification(
    prompt: str,
    response: str,
    item_categories: Dict[str, str],
) -> EvaluationResult:
    """Evaluate classification where multiple named items must each be placed into correct categories.

    Example:
        item_categories = {"apple": "fruit", "banana": "fruit", "carrot": "vegetable", "broccoli": "vegetable"}
    """
    clean_r = _clean_text(response)
    if not clean_r:
        return EvaluationResult(
            task_type=TaskType.MULTI_ITEM_CLASSIFICATION,
            is_objective=True,
            passed=False,
            reason="Empty response",
        )

    norm_r = clean_r.lower()
    missing_items = []
    misclassified_items = []
    correct_items = []

    for item, expected_cat in item_categories.items():
        item_pat = r"\b" + re.escape(item.lower()) + r"\b"
        cat_base = expected_cat.lower().rstrip("s")
        cat_pat = r"\b" + re.escape(cat_base) + r"s?\b"
        if not re.search(item_pat, norm_r):
            missing_items.append(item)
            continue

        # Look for association: e.g. "fruit: apple" or "apple is a fruit" or "fruits: ... apple"
        # Context window around the item
        idx = norm_r.find(item.lower())
        snippet = norm_r[max(0, idx - 45) : min(len(norm_r), idx + len(item) + 45)]
        if re.search(cat_pat, snippet):
            correct_items.append(item)
        else:
            misclassified_items.append(item)

    total_needed = len(item_categories)
    if len(correct_items) == total_needed:
        return EvaluationResult(
            task_type=TaskType.MULTI_ITEM_CLASSIFICATION,
            is_objective=True,
            passed=True,
            reason=f"All {total_needed} items correctly classified: {correct_items}",
            details={"correct": correct_items},
        )

    return EvaluationResult(
        task_type=TaskType.MULTI_ITEM_CLASSIFICATION,
        is_objective=True,
        passed=False,
        reason=f"Partial or incorrect classification: {len(correct_items)}/{total_needed} correct (missing: {missing_items}, misclassified: {misclassified_items})",
        details={"correct": correct_items, "missing": missing_items, "misclassified": misclassified_items},
    )


# ==============================================================================
# 5. Requested N-Item Lists Evaluator
# ==============================================================================

def evaluate_n_item_list(
    prompt: str,
    response: str,
    min_count: int,
    valid_pool: Set[str],
    mutually_exclusive_sets: Optional[List[Set[str]]] = None,
) -> EvaluationResult:
    """Evaluate lists where at least N distinct valid items from an acceptable set must be present."""
    clean_r = _clean_text(response)
    if not clean_r:
        return EvaluationResult(
            task_type=TaskType.REQUESTED_N_ITEM_LISTS,
            is_objective=True,
            passed=False,
            reason="Empty response",
        )

    norm_r = clean_r.lower()
    matched_items = set()

    for item in valid_pool:
        if re.search(r"\b" + re.escape(item.lower()) + r"\b", norm_r):
            matched_items.add(item.lower())

    # If mutually exclusive sets are specified (e.g., traditional RYB vs additive RGB)
    if mutually_exclusive_sets:
        best_set_match = 0
        for s in mutually_exclusive_sets:
            overlap = len(matched_items & {x.lower() for x in s})
            if overlap > best_set_match:
                best_set_match = overlap
        count_for_eval = best_set_match
    else:
        count_for_eval = len(matched_items)

    if count_for_eval >= min_count:
        return EvaluationResult(
            task_type=TaskType.REQUESTED_N_ITEM_LISTS,
            is_objective=True,
            passed=True,
            reason=f"Successfully listed {count_for_eval} valid items ({list(matched_items)})",
            details={"matched_items": list(matched_items)},
        )

    return EvaluationResult(
        task_type=TaskType.REQUESTED_N_ITEM_LISTS,
        is_objective=True,
        passed=False,
        reason=f"Insufficient items listed: found {count_for_eval} distinct valid items, required {min_count} (matched: {list(matched_items)})",
        details={"matched_items": list(matched_items), "required_min": min_count},
    )


# ==============================================================================
# 6. Scientific Definitions Evaluator
# ==============================================================================

def evaluate_scientific_definition(
    prompt: str,
    response: str,
    defined_term: str,
    required_concept_groups: List[List[str]],
    min_groups_required: int = 1,
) -> EvaluationResult:
    """Evaluate scientific definitions requiring biological/physical concepts.

    Rejects circular definitions (e.g. 'A mammal is a mammal') and requires substantive concept groups.
    """
    clean_r = _clean_text(response)
    if not clean_r:
        return EvaluationResult(
            task_type=TaskType.SCIENTIFIC_DEFINITIONS,
            is_objective=True,
            passed=False,
            reason="Empty response",
        )

    # 1. Circular definition check
    if _is_circular_tautology(defined_term, clean_r):
        return EvaluationResult(
            task_type=TaskType.SCIENTIFIC_DEFINITIONS,
            is_objective=True,
            passed=False,
            reason=f"Circular definition merely repeating '{defined_term}' without explanatory concepts",
        )

    norm_r = clean_r.lower()
    matched_groups = []

    for grp in required_concept_groups:
        for term in grp:
            if re.search(r"\b" + re.escape(term.lower()) + r"\b", norm_r):
                matched_groups.append(term)
                break

    if len(matched_groups) >= min_groups_required:
        return EvaluationResult(
            task_type=TaskType.SCIENTIFIC_DEFINITIONS,
            is_objective=True,
            passed=True,
            reason=f"Matched {len(matched_groups)} substantive scientific concepts: {matched_groups}",
            details={"matched_concepts": matched_groups},
        )

    return EvaluationResult(
        task_type=TaskType.SCIENTIFIC_DEFINITIONS,
        is_objective=True,
        passed=False,
        reason=f"Missing core scientific concepts for '{defined_term}' (matched {len(matched_groups)}/{min_groups_required})",
        details={"matched_concepts": matched_groups, "required_min": min_groups_required},
    )


# ==============================================================================
# 7. Deterministic Text Transformation Evaluator
# ==============================================================================

def evaluate_text_transformation(
    prompt: str,
    response: str,
    expected_substring: str,
    case_sensitive: bool = True,
) -> EvaluationResult:
    """Evaluate deterministic text transformations (e.g. uppercase conversion)."""
    clean_r = _clean_text(response)
    if not clean_r:
        return EvaluationResult(
            task_type=TaskType.DETERMINISTIC_TEXT_TRANSFORMATION,
            is_objective=True,
            passed=False,
            reason="Empty response",
        )

    target = expected_substring if case_sensitive else expected_substring.lower()
    actual = clean_r if case_sensitive else clean_r.lower()

    if target in actual:
        return EvaluationResult(
            task_type=TaskType.DETERMINISTIC_TEXT_TRANSFORMATION,
            is_objective=True,
            passed=True,
            reason=f"Exact target transformation matched: '{expected_substring}'",
        )

    return EvaluationResult(
        task_type=TaskType.DETERMINISTIC_TEXT_TRANSFORMATION,
        is_objective=True,
        passed=False,
        reason=f"Expected transformation '{expected_substring}' not found in output",
    )


# ==============================================================================
# 8. Master Evaluator Router with Versioned Rubric Store
# ==============================================================================

FROZEN_35_RUBRICS: Dict[str, Dict[str, Any]] = {
    # Greetings (Qualitative)
    "bench_01": {"task_type": TaskType.OPEN_ENDED_CONVERSATIONAL, "is_objective": False},
    "bench_02": {"task_type": TaskType.OPEN_ENDED_CONVERSATIONAL, "is_objective": False},
    "bench_03": {"task_type": TaskType.OPEN_ENDED_CONVERSATIONAL, "is_objective": False},
    "bench_04": {"task_type": TaskType.OPEN_ENDED_CONVERSATIONAL, "is_objective": False},

    # Factual QA
    "bench_05": {
        "task_type": TaskType.EXACT_FACTUAL_QA,
        "is_objective": True,
        "target": "Tokyo",
        "aliases": ["Tokyo"],
        "prohibited": ["Kyoto", "Osaka"],
    },
    "bench_06": {
        "task_type": TaskType.EXACT_FACTUAL_QA,
        "is_objective": True,
        "target": "Mercury",
        "aliases": ["Mercury"],
        "prohibited": ["Venus", "Mars", "Earth"],
    },
    "bench_07": {
        "task_type": TaskType.EXACT_FACTUAL_QA,
        "is_objective": True,
        "target": "100",
        "aliases": ["100 degrees", "100 C", "100°C", "100 celsius"],
    },
    "bench_08": {
        "task_type": TaskType.EXACT_FACTUAL_QA,
        "is_objective": True,
        "target": "Shakespeare",
        "aliases": ["William Shakespeare", "Shakespeare"],
    },
    "bench_09": {
        "task_type": TaskType.EXACT_FACTUAL_QA,
        "is_objective": True,
        "target": "366",
        "aliases": ["366 days"],
    },

    # Definitions
    "bench_10": {
        "task_type": TaskType.SCIENTIFIC_DEFINITIONS,
        "is_objective": True,
        "term": "mammal",
        "groups": [
            ["warm-blooded", "endothermic"],
            ["vertebrate", "backbone"],
            ["milk", "mammary"],
            ["hair", "fur"],
        ],
        "min_groups": 2,  # At least two distinct biological traits required!
    },
    "bench_11": {
        "task_type": TaskType.SCIENTIFIC_DEFINITIONS,
        "is_objective": True,
        "term": "photosynthesis",
        "groups": [
            ["plants", "organisms", "chlorophyll"],
            ["sunlight", "light energy"],
            ["glucose", "sugar", "chemical energy", "oxygen"],
        ],
        "min_groups": 2,
    },
    "bench_12": {
        "task_type": TaskType.SCIENTIFIC_DEFINITIONS,
        "is_objective": True,
        "term": "algorithm",
        "groups": [
            ["step-by-step", "sequence of steps"],
            ["procedure", "rules", "instructions", "method"],
            ["solving", "solve", "problem", "computation"],
        ],
        "min_groups": 2,
    },
    "bench_13": {
        "task_type": TaskType.SCIENTIFIC_DEFINITIONS,
        "is_objective": True,
        "term": "gravity",
        "groups": [
            ["force", "fundamental interaction"],
            ["attracts", "attraction", "pull", "pulls"],
            ["mass", "bodies", "objects"],
        ],
        "min_groups": 2,
    },

    # Arithmetic
    "bench_14": {"task_type": TaskType.NUMERIC_ARITHMETIC, "is_objective": True},
    "bench_15": {"task_type": TaskType.NUMERIC_ARITHMETIC, "is_objective": True},
    "bench_16": {"task_type": TaskType.NUMERIC_ARITHMETIC, "is_objective": True},
    "bench_17": {
        "task_type": TaskType.EXACT_FACTUAL_QA,
        "is_objective": True,
        "target": "yes",
        "aliases": ["yes, 29 is a prime", "29 is prime", "is a prime number"],
    },
    "bench_18": {"task_type": TaskType.NUMERIC_ARITHMETIC, "is_objective": True},

    # Explanations (Qualitative Review)
    "bench_19": {"task_type": TaskType.OPEN_ENDED_CONVERSATIONAL, "is_objective": False},
    "bench_20": {"task_type": TaskType.OPEN_ENDED_CONVERSATIONAL, "is_objective": False},
    "bench_21": {"task_type": TaskType.OPEN_ENDED_CONVERSATIONAL, "is_objective": False},
    "bench_22": {"task_type": TaskType.OPEN_ENDED_CONVERSATIONAL, "is_objective": False},

    # Classifications
    "bench_23": {
        "task_type": TaskType.MULTI_ITEM_CLASSIFICATION,
        "is_objective": True,
        "item_categories": {
            "apple": "fruit",
            "banana": "fruit",
            "carrot": "vegetable",
            "broccoli": "vegetable",
        },
    },
    "bench_24": {
        "task_type": TaskType.MULTI_ITEM_CLASSIFICATION,
        "is_objective": True,
        "item_categories": {
            "ice": "solid",
            "water": "liquid",
            "oxygen": "gas",
        },
    },
    "bench_25": {
        "task_type": TaskType.MULTI_ITEM_CLASSIFICATION,
        "is_objective": True,
        "item_categories": {
            "dolphin": "mammal",
            "dog": "mammal",
            "eagle": "bird",
            "penguin": "bird",
        },
    },
    "bench_26": {
        "task_type": TaskType.MULTI_ITEM_CLASSIFICATION,
        "is_objective": True,
        "item_categories": {
            "solar": "renewable",
            "wind": "renewable",
            "coal": "non-renewable",
            "petroleum": "non-renewable",
        },
    },

    # Instructions & Lists
    "bench_27": {"task_type": TaskType.OPEN_ENDED_CONVERSATIONAL, "is_objective": False},
    "bench_28": {
        "task_type": TaskType.REQUESTED_N_ITEM_LISTS,
        "is_objective": True,
        "min_count": 3,
        "valid_pool": {"red", "blue", "yellow", "green"},
        "mutually_exclusive_sets": [
            {"red", "blue", "yellow"},  # Traditional RYB
            {"red", "green", "blue"},   # Additive RGB
        ],
    },
    "bench_29": {
        "task_type": TaskType.DETERMINISTIC_TEXT_TRANSFORMATION,
        "is_objective": True,
        "expected_substring": "GROUNDUP GPT IS FAST AND LIGHTWEIGHT",
        "case_sensitive": True,
    },
    "bench_30": {
        "task_type": TaskType.REQUESTED_N_ITEM_LISTS,
        "is_objective": True,
        "min_count": 3,
        "valid_pool": {
            "python", "javascript", "c++", "java", "c", "rust", "go",
            "c#", "ruby", "php", "swift", "kotlin", "typescript"
        },
    },
    "bench_31": {"task_type": TaskType.OPEN_ENDED_CONVERSATIONAL, "is_objective": False},

    # Everyday Questions
    "bench_32": {"task_type": TaskType.OPEN_ENDED_CONVERSATIONAL, "is_objective": False},
    "bench_33": {
        "task_type": TaskType.EXACT_FACTUAL_QA,
        "is_objective": True,
        "target": "rain",
        "aliases": ["protection from rain", "rain or sun", "protect from rain", "keep dry", "shelter from rain"],
    },
    "bench_34": {"task_type": TaskType.OPEN_ENDED_CONVERSATIONAL, "is_objective": False},
    "bench_35": {
        "task_type": TaskType.EXACT_FACTUAL_QA,
        "is_objective": True,
        "target": "twice",
        "aliases": ["twice a day", "2 times", "two times a day", "twice daily"],
    },
}


def grade_benchmark_item(
    item_id: str,
    prompt: str,
    response: str,
    category: str,
    suite: str,
    expected_guidance: str = "",
) -> EvaluationResult:
    """Unified grading entry point for any benchmark item."""
    # 1. Check if frozen 35 item with registered rubric
    if suite == "frozen_35" and item_id in FROZEN_35_RUBRICS:
        rubric = FROZEN_35_RUBRICS[item_id]
        ttype = rubric["task_type"]
        if not rubric["is_objective"]:
            return EvaluationResult(
                task_type=ttype,
                is_objective=False,
                passed=None,
                reason="Unscored (requires qualitative review)",
                requires_qualitative_review=True,
            )
        if ttype == TaskType.NUMERIC_ARITHMETIC:
            return evaluate_arithmetic(prompt, response)
        elif ttype == TaskType.EXACT_FACTUAL_QA:
            return evaluate_exact_factual(prompt, response, rubric["target"], rubric.get("aliases"), rubric.get("prohibited"))
        elif ttype == TaskType.SCIENTIFIC_DEFINITIONS:
            return evaluate_scientific_definition(prompt, response, rubric["term"], rubric["groups"], rubric.get("min_groups", 2))
        elif ttype == TaskType.MULTI_ITEM_CLASSIFICATION:
            return evaluate_multi_item_classification(prompt, response, rubric["item_categories"])
        elif ttype == TaskType.REQUESTED_N_ITEM_LISTS:
            return evaluate_n_item_list(prompt, response, rubric["min_count"], rubric["valid_pool"], rubric.get("mutually_exclusive_sets"))
        elif ttype == TaskType.DETERMINISTIC_TEXT_TRANSFORMATION:
            return evaluate_text_transformation(prompt, response, rubric["expected_substring"], rubric.get("case_sensitive", True))

    # 2. Check general category-based routing for transfer test & seen train
    if category == "arithmetic":
        return evaluate_arithmetic(prompt, response)

    elif category == "classification":
        # Determine binary categories from prompt
        m_or = re.search(r"is\s+(?:an?\s+)?([\w\-]+)\s+(?:a\s+|an\s+)?([\w\-]+)\s+or\s+(?:a\s+|an\s+)?([\w\-]+)", prompt, re.IGNORECASE)
        if m_or:
            subj, opt1, opt2 = m_or.group(1), m_or.group(2), m_or.group(3)
            # Find which option is in expected guidance
            norm_exp = expected_guidance.lower()
            if opt1.lower() in norm_exp and opt2.lower() not in norm_exp:
                return evaluate_binary_classification(prompt, response, opt1, opt2)
            elif opt2.lower() in norm_exp and opt1.lower() not in norm_exp:
                return evaluate_binary_classification(prompt, response, opt2, opt1)
        # Fallback binary classification
        if "conductor or an insulator" in prompt.lower():
            return evaluate_binary_classification(prompt, response, "insulator", "conductor")
        elif "reptile or an amphibian" in prompt.lower():
            return evaluate_binary_classification(prompt, response, "amphibian", "reptile")
        elif "diurnal or nocturnal" in prompt.lower():
            return evaluate_binary_classification(prompt, response, "nocturnal", "diurnal")
        elif "gas, liquid, or solid" in prompt.lower():
            return evaluate_binary_classification(prompt, response, "gas", "solid")
        elif "magnetic or non-magnetic" in prompt.lower():
            return evaluate_binary_classification(prompt, response, "magnetic", "non-magnetic")
        elif "fruit or a vegetable" in prompt.lower():
            return evaluate_binary_classification(prompt, response, "vegetable", "fruit")
        elif "fish or a mammal" in prompt.lower():
            return evaluate_binary_classification(prompt, response, "mammal", "fish")

        return EvaluationResult(
            task_type=TaskType.BINARY_CLASSIFICATION,
            is_objective=False,
            passed=None,
            reason="Classification pattern not deterministically parsed (requires qualitative review)",
            requires_qualitative_review=True,
        )

    elif category == "everyday_factual":
        # Extract target from expected_guidance
        if "The capital of " in expected_guidance:
            cap = expected_guidance.split("is ")[-1].rstrip(".")
            return evaluate_exact_factual(prompt, response, cap, [cap])
        elif "sides" in expected_guidance or "hours" in expected_guidance or "days" in expected_guidance or "inches" in expected_guidance:
            nums = re.findall(r"\b\d+\b", expected_guidance)
            if nums:
                return evaluate_exact_factual(prompt, response, nums[-1], [nums[-1]])
        elif "0 degrees" in expected_guidance or "0°C" in expected_guidance or "0 Celsius" in expected_guidance:
            return evaluate_exact_factual(prompt, response, "0", ["0 degrees", "0°C", "0 C", "0 Celsius"])

        return EvaluationResult(
            task_type=TaskType.EXACT_FACTUAL_QA,
            is_objective=False,
            passed=None,
            reason="Factual question requires qualitative review",
            requires_qualitative_review=True,
        )

    elif category == "basic_science":
        # Potassium -> K
        if "potassium" in prompt.lower() and "symbol" in prompt.lower():
            return evaluate_exact_factual(prompt, response, "K", ["K"], prohibited=["Na", "P", "Ca"])
        elif "largest internal organ" in prompt.lower():
            return evaluate_exact_factual(prompt, response, "liver", ["liver", "the liver"])
        elif "negative electrical charge" in prompt.lower():
            return evaluate_exact_factual(prompt, response, "electron", ["electron", "electrons"], prohibited=["proton", "neutron"])
        elif "pumps blood" in prompt.lower():
            return evaluate_exact_factual(prompt, response, "heart", ["heart", "the heart"])
        elif "center of an atom" in prompt.lower():
            return evaluate_exact_factual(prompt, response, "nucleus", ["nucleus"])
        elif "red planet" in prompt.lower():
            return evaluate_exact_factual(prompt, response, "Mars", ["Mars"])
        elif "chemical formula for water" in prompt.lower():
            return evaluate_exact_factual(prompt, response, "H2O", ["H2O", "H 2 O", "H20"])
        elif "humans need to breathe" in prompt.lower():
            return evaluate_exact_factual(prompt, response, "oxygen", ["oxygen", "O2"])

        return EvaluationResult(
            task_type=TaskType.EXACT_FACTUAL_QA,
            is_objective=False,
            passed=None,
            reason="Science question requires qualitative review",
            requires_qualitative_review=True,
        )

    elif category == "definitions":
        if "constellation" in prompt.lower():
            return evaluate_scientific_definition(prompt, response, "constellation", [["stars", "star"], ["pattern", "group", "configuration"]], min_groups_required=2)
        elif "kinetic energy" in prompt.lower():
            return evaluate_scientific_definition(prompt, response, "kinetic energy", [["motion", "movement", "moving"], ["energy"]], min_groups_required=2)
        elif "insulator" in prompt.lower():
            return evaluate_scientific_definition(prompt, response, "insulator", [["resists", "blocks", "prevents", "does not conduct"], ["current", "electricity", "flow"]], min_groups_required=2)
        elif "carnivore" in prompt.lower():
            return evaluate_scientific_definition(prompt, response, "carnivore", [["meat", "flesh", "animals"], ["eats", "consumes", "feed"]], min_groups_required=2)
        elif "habitat" in prompt.lower():
            return evaluate_scientific_definition(prompt, response, "habitat", [["environment", "home", "place", "area"], ["lives", "living", "organism", "species"]], min_groups_required=2)
        elif "atom" in prompt.lower():
            return evaluate_scientific_definition(prompt, response, "atom", [["unit", "particle", "building block"], ["element", "matter", "nucleus", "electrons"]], min_groups_required=2)
        elif "velocity" in prompt.lower():
            return evaluate_scientific_definition(prompt, response, "velocity", [["speed", "rate"], ["direction"]], min_groups_required=2)

        return EvaluationResult(
            task_type=TaskType.SCIENTIFIC_DEFINITIONS,
            is_objective=False,
            passed=None,
            reason="Definition requires qualitative review",
            requires_qualitative_review=True,
        )

    elif category == "short_lists":
        if "states of matter" in prompt.lower():
            return evaluate_n_item_list(prompt, response, 3, {"solid", "liquid", "gas"})
        elif "fruits" in prompt.lower():
            return evaluate_n_item_list(prompt, response, 3, {"apple", "orange", "banana", "grape", "berry", "pear", "mango"})
        elif "continents" in prompt.lower():
            return evaluate_n_item_list(prompt, response, 2, {"asia", "africa", "north america", "south america", "antarctica", "europe", "australia"})
        elif "household pets" in prompt.lower():
            return evaluate_n_item_list(prompt, response, 2, {"dog", "cat", "fish", "bird", "hamster", "rabbit"})
        elif "ocean animals" in prompt.lower():
            return evaluate_n_item_list(prompt, response, 2, {"whale", "dolphin", "shark", "octopus", "seal", "fish"})
        elif "primary colors" in prompt.lower():
            return evaluate_n_item_list(prompt, response, 2, {"red", "blue", "yellow", "green"})

        return EvaluationResult(
            task_type=TaskType.REQUESTED_N_ITEM_LISTS,
            is_objective=False,
            passed=None,
            reason="List item requires qualitative review",
            requires_qualitative_review=True,
        )

    elif category == "simple_instructions":
        if "spell" in prompt.lower() and "'cat'" in prompt.lower():
            return evaluate_text_transformation(prompt, response, "c-a-t", case_sensitive=False)
        elif "french" in prompt.lower() and "'hello'" in prompt.lower():
            return evaluate_exact_factual(prompt, response, "bonjour", ["bonjour", "'bonjour'"])
        elif "spanish" in prompt.lower() and "'thank you'" in prompt.lower():
            return evaluate_exact_factual(prompt, response, "gracias", ["gracias", "'gracias'"])
        elif "spanish" in prompt.lower() and "'hello'" in prompt.lower():
            return evaluate_exact_factual(prompt, response, "hola", ["hola", "'hola'"])
        elif "opposite of 'hot'" in prompt.lower():
            return evaluate_exact_factual(prompt, response, "cold", ["cold", "'cold'"])
        elif "antonym for 'fast'" in prompt.lower():
            return evaluate_exact_factual(prompt, response, "slow", ["slow", "'slow'"])

        return EvaluationResult(
            task_type=TaskType.DETERMINISTIC_TEXT_TRANSFORMATION,
            is_objective=False,
            passed=None,
            reason="Instruction requires qualitative review",
            requires_qualitative_review=True,
        )

    # Conversational conventions: greetings, thanks, farewells, assistance
    return EvaluationResult(
        task_type=TaskType.OPEN_ENDED_CONVERSATIONAL,
        is_objective=False,
        passed=None,
        reason="Open-ended conversational prompt (requires qualitative review)",
        requires_qualitative_review=True,
    )
