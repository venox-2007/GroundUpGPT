"""Deterministic, safe arithmetic calculator tool for GroundUpGPT.

Evaluates basic mathematical expressions deterministically without using Python
eval(), exec(), shell calls, or unrestricted AST evaluation. Distinguishes tool-computed
results from neural language model generations.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from decimal import Decimal, DivisionByZero, InvalidOperation, Overflow
import re
from typing import Optional, Union

# Safety constraints
MAX_EXPRESSION_LENGTH: int = 120
MAX_NUMERIC_MAGNITUDE: Decimal = Decimal("1e15")
MAX_OPERATIONS_COUNT: int = 25

# Normalization patterns for natural language arithmetic queries
NL_ARITHMETIC_PREFIX_RE = re.compile(
    r"^(?:what\s+is|calculate|compute|solve|how\s+much\s+is|evaluate|find)\s+",
    re.IGNORECASE,
)
TRAILING_PUNCT_RE = re.compile(r"[?=.\s]+$")


@dataclass
class CalculationResult:
    """Structured result of a safe calculation attempt.

    Attributes:
        raw_query: The input prompt or expression string.
        extracted_expression: The isolated mathematical expression string, if extracted.
        is_arithmetic_query: Whether the query was identified as arithmetic.
        parser_confidence: Confidence score of expression extraction (0.0 to 1.0).
        is_valid: Whether the expression was safely parsed and evaluated.
        result: Evaluated numeric result (Decimal or int), or None if evaluation failed.
        result_str: Human-readable string representation of the result.
        error_type: Error category code if failed, otherwise None.
        error_message: Detailed explanatory error message if failed.
        is_tool_computed: Flag explicitly indicating this is a deterministic tool output.
    """

    raw_query: str
    extracted_expression: Optional[str]
    is_arithmetic_query: bool
    parser_confidence: float
    is_valid: bool
    result: Optional[Union[int, Decimal]] = None
    result_str: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    is_tool_computed: bool = True


class SafeASTEvaluator:
    """Restricted AST visitor that evaluates only safe elementary arithmetic."""

    ALLOWED_BIN_OPS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
    }

    ALLOWED_UNARY_OPS = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }

    def __init__(self, max_magnitude: Decimal = MAX_NUMERIC_MAGNITUDE, max_ops: int = MAX_OPERATIONS_COUNT) -> None:
        self.max_magnitude = max_magnitude
        self.max_ops = max_ops
        self.op_count = 0

    def evaluate(self, node: ast.AST) -> Decimal:
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Expression):
            return self.evaluate(node.body)

        elif isinstance(node, ast.Constant):
            # Only numeric literals are permitted
            if isinstance(node.value, (int, float)):
                val = Decimal(str(node.value))
                if abs(val) > self.max_magnitude:
                    raise ValueError(f"Numeric literal {val} exceeds magnitude limit {self.max_magnitude}")
                return val
            raise TypeError(f"Disallowed constant type in arithmetic expression: {type(node.value)}")

        elif isinstance(node, ast.UnaryOp):
            self.op_count += 1
            if self.op_count > self.max_ops:
                raise ValueError("Expression exceeds maximum allowed operation complexity")

            op_type = type(node.op)
            if op_type not in self.ALLOWED_UNARY_OPS:
                raise TypeError(f"Disallowed unary operator: {op_type.__name__}")

            operand = self.evaluate(node.operand)
            val = self.ALLOWED_UNARY_OPS[op_type](operand)
            if abs(val) > self.max_magnitude:
                raise ValueError(f"Intermediate result {val} exceeds magnitude limit")
            return val

        elif isinstance(node, ast.BinOp):
            self.op_count += 1
            if self.op_count > self.max_ops:
                raise ValueError("Expression exceeds maximum allowed operation complexity")

            op_type = type(node.op)
            if op_type not in self.ALLOWED_BIN_OPS:
                raise TypeError(f"Disallowed binary operator: {op_type.__name__}")

            left = self.evaluate(node.left)
            right = self.evaluate(node.right)

            # Check division by zero before executing
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ZeroDivisionError("Division by zero")

            try:
                val = self.ALLOWED_BIN_OPS[op_type](left, right)
            except (DivisionByZero, ZeroDivisionError):
                raise ZeroDivisionError("Division by zero")
            except (InvalidOperation, Overflow) as exc:
                raise ValueError(f"Arithmetic computation overflow or invalid operation: {exc}")

            if abs(val) > self.max_magnitude:
                raise ValueError(f"Intermediate result {val} exceeds magnitude limit")
            return val

        else:
            # Reject any function calls, variable names, attribute lookups, lambdas, etc.
            raise SecurityError(f"Disallowed AST node type '{type(node).__name__}' in safe arithmetic calculator")


class SecurityError(Exception):
    """Raised when an expression attempts unsafe operations, lookups, or injections."""


def extract_arithmetic_expression(query: str) -> tuple[Optional[str], float]:
    """Extract and normalize a candidate arithmetic expression from natural language query.

    Args:
        query: User input query (e.g. "What is 7 + 9?", "Compute 64 divided by 8").

    Returns:
        tuple[Optional[str], float]: (isolated_expression, confidence).
    """
    cleaned = query.strip()
    if not cleaned:
        return None, 0.0

    # Strip natural language preamble
    stripped = NL_ARITHMETIC_PREFIX_RE.sub("", cleaned)
    stripped = TRAILING_PUNCT_RE.sub("", stripped).strip()

    # Normalize natural language math operators
    replacements = [
        (r"\bdivided\s+by\b", "/"),
        (r"\bmultiplied\s+by\b", "*"),
        (r"\btimes\b", "*"),
        (r"\bplus\b", "+"),
        (r"\bminus\b", "-"),
        (r"\bover\b", "/"),
        (r"\bmod\b", "%"),
        (r"\bmodulo\b", "%"),
        (r"×", "*"),
        (r"÷", "/"),
    ]
    normalized = stripped
    for pattern, repl in replacements:
        normalized = re.sub(pattern, repl, normalized, flags=re.IGNORECASE)

    # Sanity check: must contain at least one digit and at least one operator
    has_digit = bool(re.search(r"\d", normalized))
    has_operator = bool(re.search(r"[+\-*/%]", normalized))

    if not has_digit:
        return None, 0.0

    # Pure expression check (only allowed characters: digits, ., +, -, *, /, %, (, ), space)
    if re.fullmatch(r"[\d\s.+\-*/%()]+", normalized):
        confidence = 1.0 if has_operator else 0.5
        return normalized.strip(), confidence

    # If it still contains arbitrary alphabetical words, it's ambiguous or non-arithmetic
    if re.search(r"[a-zA-Z]", normalized):
        return None, 0.0

    return None, 0.0


def calculate(query: str) -> CalculationResult:
    """Deterministically parse and evaluate an arithmetic query.

    Args:
        query: Arithmetic expression or natural language question.

    Returns:
        CalculationResult containing success status, value, or detailed error reason.
    """
    raw_query = query.strip()
    if len(raw_query) > MAX_EXPRESSION_LENGTH:
        return CalculationResult(
            raw_query=raw_query,
            extracted_expression=None,
            is_arithmetic_query=False,
            parser_confidence=0.0,
            is_valid=False,
            error_type="length_limit_exceeded",
            error_message=f"Expression length ({len(raw_query)}) exceeds maximum limit ({MAX_EXPRESSION_LENGTH})",
        )

    expr_str, confidence = extract_arithmetic_expression(raw_query)
    if expr_str is None or confidence < 0.4:
        return CalculationResult(
            raw_query=raw_query,
            extracted_expression=expr_str,
            is_arithmetic_query=False,
            parser_confidence=confidence,
            is_valid=False,
            error_type="not_arithmetic_query",
            error_message="Query could not be parsed as an unambiguous arithmetic expression",
        )

    # Check for invalid consecutive operators (e.g. '12 ++ 4' or '15 */ 3')
    if re.search(r"[+\-*/%]{2,}", expr_str):
        return CalculationResult(
            raw_query=raw_query,
            extracted_expression=expr_str,
            is_arithmetic_query=True,
            parser_confidence=confidence,
            is_valid=False,
            error_type="syntax_error",
            error_message="Invalid consecutive operators in expression",
        )

    # Parse with restricted Python AST in eval mode
    try:
        parsed_ast = ast.parse(expr_str, mode="eval")
    except SyntaxError as syn_err:
        return CalculationResult(
            raw_query=raw_query,
            extracted_expression=expr_str,
            is_arithmetic_query=True,
            parser_confidence=confidence,
            is_valid=False,
            error_type="syntax_error",
            error_message=f"Syntax error in arithmetic expression: {syn_err}",
        )

    evaluator = SafeASTEvaluator()
    try:
        val = evaluator.evaluate(parsed_ast)

        # Normalize whole numbers to exact int representation
        if val == val.to_integral_value():
            int_val = int(val)
            return CalculationResult(
                raw_query=raw_query,
                extracted_expression=expr_str,
                is_arithmetic_query=True,
                parser_confidence=confidence,
                is_valid=True,
                result=int_val,
                result_str=str(int_val),
            )
        else:
            # Format decimal cleanly (e.g., 2.5)
            # Normalize trailing zeros
            dec_str = f"{val.normalize():f}"
            return CalculationResult(
                raw_query=raw_query,
                extracted_expression=expr_str,
                is_arithmetic_query=True,
                parser_confidence=confidence,
                is_valid=True,
                result=val,
                result_str=dec_str,
            )

    except ZeroDivisionError:
        return CalculationResult(
            raw_query=raw_query,
            extracted_expression=expr_str,
            is_arithmetic_query=True,
            parser_confidence=confidence,
            is_valid=False,
            error_type="division_by_zero",
            error_message="Division by zero is undefined",
        )
    except SecurityError as sec_err:
        return CalculationResult(
            raw_query=raw_query,
            extracted_expression=expr_str,
            is_arithmetic_query=True,
            parser_confidence=confidence,
            is_valid=False,
            error_type="security_violation",
            error_message=str(sec_err),
        )
    except Exception as exc:
        return CalculationResult(
            raw_query=raw_query,
            extracted_expression=expr_str,
            is_arithmetic_query=True,
            parser_confidence=confidence,
            is_valid=False,
            error_type="evaluation_error",
            error_message=str(exc),
        )
