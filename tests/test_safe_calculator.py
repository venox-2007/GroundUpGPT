"""Unit tests for the deterministic safe arithmetic calculator tool."""

from decimal import Decimal
import pytest

from src.safe_calculator import (
    CalculationResult,
    MAX_EXPRESSION_LENGTH,
    MAX_NUMERIC_MAGNITUDE,
    SafeASTEvaluator,
    SecurityError,
    calculate,
    extract_arithmetic_expression,
)


def test_basic_arithmetic_operations() -> None:
    """Verify standard arithmetic operations evaluate to exact results."""
    # Addition
    res_add = calculate("What is 7 + 9?")
    assert res_add.is_valid is True
    assert res_add.result == 16
    assert res_add.result_str == "16"
    assert res_add.is_tool_computed is True

    # Subtraction
    res_sub = calculate("What is 15 - 7?")
    assert res_sub.is_valid is True
    assert res_sub.result == 8
    assert res_sub.result_str == "8"

    # Multiplication
    res_mul = calculate("Compute 12 * 4")
    assert res_mul.is_valid is True
    assert res_mul.result == 48
    assert res_mul.result_str == "48"

    # Division (integer)
    res_div = calculate("What is 64 / 8?")
    assert res_div.is_valid is True
    assert res_div.result == 8
    assert res_div.result_str == "8"

    # Division (fractional decimal)
    res_dec = calculate("7 / 2")
    assert res_dec.is_valid is True
    assert res_dec.result_str == "3.5"


def test_natural_language_phrasing() -> None:
    """Verify natural language mathematical phrasing converts properly."""
    res_div_nl = calculate("What is 64 divided by 8?")
    assert res_div_nl.is_valid is True
    assert res_div_nl.result == 8

    res_mul_nl = calculate("Calculate 25 times 4")
    assert res_mul_nl.is_valid is True
    assert res_mul_nl.result == 100

    res_add_nl = calculate("How much is 10 plus 25?")
    assert res_add_nl.is_valid is True
    assert res_add_nl.result == 35

    res_sub_nl = calculate("Solve 50 minus 18")
    assert res_sub_nl.is_valid is True
    assert res_sub_nl.result == 32


def test_parentheses_and_operator_precedence() -> None:
    """Verify standard operator precedence and parenthesized expressions."""
    # Precedence: multiplication before addition
    res_prec = calculate("10 + 5 * 2")
    assert res_prec.is_valid is True
    assert res_prec.result == 20

    # Parentheses override precedence
    res_paren = calculate("(10 + 5) * 2")
    assert res_paren.is_valid is True
    assert res_paren.result == 30

    # Nested parentheses
    res_nested = calculate("(100 - 20) / (2 + 2)")
    assert res_nested.is_valid is True
    assert res_nested.result == 20


def test_division_by_zero_handling() -> None:
    """Verify division by zero is safely caught and returns explicit error without crashing."""
    res_zero = calculate("10 / 0")
    assert res_zero.is_valid is False
    assert res_zero.error_type == "division_by_zero"
    assert "Division by zero" in res_zero.error_message

    res_zero_paren = calculate("50 / (5 - 5)")
    assert res_zero_paren.is_valid is False
    assert res_zero_paren.error_type == "division_by_zero"


def test_security_and_code_injection_rejection() -> None:
    """Verify code execution, imports, and arbitrary variable names are rejected."""
    # Import attempt
    res_imp = calculate("__import__('os').system('dir')")
    assert res_imp.is_valid is False

    # Function call attempt
    res_call = calculate("eval('2+2')")
    assert res_call.is_valid is False

    # Attribute access
    res_attr = calculate("(1).__class__.__bases__[0]")
    assert res_attr.is_valid is False

    # AST evaluator directly raises SecurityError on disallowed AST nodes
    import ast
    evaluator = SafeASTEvaluator()
    bad_ast = ast.parse("len([1, 2, 3])", mode="eval")
    with pytest.raises(SecurityError):
        evaluator.evaluate(bad_ast)


def test_expression_limits_and_overflow() -> None:
    """Verify maximum length, operation complexity, and numeric magnitude limits."""
    # Expression length limit
    long_expr = "1 + " * (MAX_EXPRESSION_LENGTH // 4 + 5) + "1"
    res_long = calculate(long_expr)
    assert res_long.is_valid is False
    assert res_long.error_type == "length_limit_exceeded"

    # Numeric magnitude limit in evaluator
    import ast
    evaluator = SafeASTEvaluator()
    huge_ast = ast.parse("1000000000000000000000", mode="eval")
    with pytest.raises(ValueError, match="exceeds magnitude limit"):
        evaluator.evaluate(huge_ast)


def test_non_arithmetic_queries_rejection() -> None:
    """Verify non-arithmetic and ambiguous queries are rejected without guessing."""
    res_text = calculate("What is the capital of France?")
    assert res_text.is_valid is False
    assert res_text.is_arithmetic_query is False

    res_date = calculate("What is the date 2024-05-12?")
    assert res_date.is_valid is False
    assert res_date.is_arithmetic_query is False
