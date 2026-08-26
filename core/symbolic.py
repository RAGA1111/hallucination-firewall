"""
Symbolic logic checker for numerical and date constraints in claims.

Extracts ONLY explicit arithmetic / date relationships from a claim via an
LLM call, then evaluates them safely using a restricted AST walker.

Important:
- Independent dates/numbers are NOT treated as a mathematical relationship.
- The LLM must identify a relationship explicitly stated by the claim.
- No eval() or exec() is used.

Public API:
    check_symbolic_logic(claim) -> dict
"""

from __future__ import annotations

import ast
import logging
import re

from core.call_llm import call_llm, is_ollama_running

logger = logging.getLogger(__name__)

__all__ = ["check_symbolic_logic"]


# ── Prompt ─────────────────────────────────────────────────────────────────────

# NEW: The prompt is intentionally strict about semantic relationships.
# Previously, the LLM could see several unrelated numbers/dates and invent
# an arithmetic relationship between them (for example, 1932 + 18 == 1955).
SYMBOLIC_PROMPT = """Extract ONLY an explicit numerical or date relationship that the claim itself states.

Write it as a single Python assert.

IMPORTANT RULES:
- Do NOT create a relationship merely because multiple numbers or dates appear.
- Do NOT add, subtract, compare, multiply, divide, or otherwise calculate
  between independent dates.
- Do NOT infer an age, duration, elapsed years, or date difference unless
  the claim explicitly states that relationship.
- A birth year and death year are NOT an arithmetic constraint unless the
  claim explicitly gives an age or duration.
- An emigration year and death year are NOT an arithmetic constraint unless
  the claim explicitly says something like "X years later" or gives an
  explicit duration.
- Use ONLY relationships explicitly expressed in the wording of the claim.
- If there is no explicit numerical/date relationship, return NONE.

Examples:

Claim: "Born 1990, died 20 years later in 2010."
-> assert 1990 + 20 == 2010

Claim: "Born in 1990 and died in 2010."
-> NONE

Claim: "He was 20 years old when he died in 2010."
-> NONE

Claim: "He was born in 1990 and died at age 20."
-> NONE

Claim: "The event lasted 5 years and ended in 2010 after starting in 2005."
-> assert 2005 + 5 == 2010

Claim: "Einstein emigrated to the United States in 1932 and died on April 18, 1955."
-> NONE

Claim: "The tower is in Paris."
-> NONE

Return ONLY the assert statement or NONE — no markdown, no explanation.

Claim: "{claim}"

Output:"""


# ── AST allowlist ──────────────────────────────────────────────────────────────

# Only numeric literals, arithmetic operators, and comparison operators are
# permitted. No names, no calls, no attribute access.
_ALLOWED_NODES = (
    # Structural
    ast.Expression,

    # Boolean logic
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,

    # Comparisons
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,

    # Arithmetic
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.UAdd,
    ast.USub,

    # Literals only — no ast.Name, ast.Attribute, ast.Call, etc.
    ast.Constant,
)


# ── Safe AST validator ─────────────────────────────────────────────────────────

def _validate_ast(tree: ast.AST) -> None:
    """
    Raise ValueError if the tree contains any node outside the allowlist.

    This is a security pre-flight before _safe_eval runs.
    """
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError("Disallowed AST node: %s" % type(node).__name__)


# ── Grounding helpers ──────────────────────────────────────────────────────────

# A literal is only "grounded" if it appears as a standalone numeric token in
# the claim text — not merely as a substring of a larger number.
def _claim_numbers(claim: str) -> set[str]:
    """Return the set of standalone numeric tokens actually written in the claim."""
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", claim))


def _is_grounded(value: object, claim: str) -> bool:
    """True if value matches a whole numeric token in the claim."""
    if not isinstance(value, (int, float)):
        return True

    claim_nums = _claim_numbers(claim)
    candidates = {str(value)}

    if isinstance(value, float) and value.is_integer():
        candidates.add(str(int(value)))

    if isinstance(value, int):
        candidates.add(f"{value}.0")

    return bool(candidates & claim_nums)


# ── Safe evaluator ─────────────────────────────────────────────────────────────

def _safe_eval(node: ast.AST) -> object:
    """Recursively evaluate a pre-validated AST node."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)

        ops = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.FloorDiv: lambda a, b: a // b,
            ast.Mod: lambda a, b: a % b,
            ast.Pow: lambda a, b: a ** b,
        }

        handler = ops.get(type(node.op))
        if handler is None:
            raise ValueError(
                "Unsupported binary operator: %s" % type(node.op).__name__
            )

        return handler(left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)

        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.Not):
            return not operand

        raise ValueError(
            "Unsupported unary operator: %s" % type(node.op).__name__
        )

    if isinstance(node, ast.BoolOp):
        values = [_safe_eval(v) for v in node.values]

        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)

        raise ValueError(
            "Unsupported boolean operator: %s" % type(node.op).__name__
        )

    if isinstance(node, ast.Compare):
        left = _safe_eval(node.left)

        cmp_ops = {
            ast.Eq: lambda a, b: a == b,
            ast.NotEq: lambda a, b: a != b,
            ast.Lt: lambda a, b: a < b,
            ast.LtE: lambda a, b: a <= b,
            ast.Gt: lambda a, b: a > b,
            ast.GtE: lambda a, b: a >= b,
        }

        for op, comparator in zip(node.ops, node.comparators):
            right = _safe_eval(comparator)
            handler = cmp_ops.get(type(op))

            if handler is None:
                raise ValueError(
                    "Unsupported comparison operator: %s" % type(op).__name__
                )

            if not handler(left, right):
                return False

            left = right

        return True

    raise ValueError("Unsupported AST node: %s" % type(node).__name__)


# ── Public API ─────────────────────────────────────────────────────────────────

_NO_LOGIC = {
    "has_logic": False,
    "passed": True,
    "note": "No symbolic logic detected.",
}


def check_symbolic_logic(claim: str) -> dict:
    """
    Evaluate an explicit numerical/date constraint embedded in claim.

    Returns:
        has_logic : True only when an explicit relationship was extracted
                    and evaluated.
        passed    : True when the relationship holds or no logic was found.
        note      : human-readable explanation.

    Never raises. Errors are treated as no symbolic logic so a weak symbolic
    extraction cannot independently turn a claim into a hallucination.
    """
    if not claim or not claim.strip():
        return _NO_LOGIC

    # Skip the LLM call entirely when Ollama is offline.
    if not is_ollama_running():
        logger.debug(
            "check_symbolic_logic | Ollama offline — skipping symbolic check"
        )
        return _NO_LOGIC

    prompt = SYMBOLIC_PROMPT.format(claim=claim)
    raw = call_llm(prompt, temperature=0.0)
    python_logic = (raw or "").strip()

    if (
        not python_logic
        or python_logic.upper() == "NONE"
        or not python_logic.startswith("assert")
    ):
        return _NO_LOGIC

    logger.info("check_symbolic_logic | logic=%r", python_logic)

    try:
        condition = python_logic[len("assert"):].strip()
        tree = ast.parse(condition, mode="eval")
        _validate_ast(tree)

        # ── Grounding check ────────────────────────────────────────────────
        # This remains a secondary safety check. It ensures every generated
        # number actually occurs in the claim, but it cannot by itself prove
        # that the LLM's relationship is semantically valid.
        ungrounded = [
            node.value
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, (int, float))
                and not _is_grounded(node.value, claim)
            )
        ]

        if ungrounded:
            logger.warning(
                "check_symbolic_logic | rejected ungrounded literal(s) %s "
                "not found in claim",
                ungrounded,
            )
            return {
                "has_logic": False,
                "passed": True,
                "note": (
                    "Skipped: assert contained ungrounded literal(s) %s "
                    "not present in claim." % ungrounded
                ),
            }

        result = _safe_eval(tree)

        if result is True:
            return {
                "has_logic": True,
                "passed": True,
                "note": "Logic passed: %s" % condition,
            }

        return {
            "has_logic": True,
            "passed": False,
            "note": "Logic failed: %s" % condition,
        }

    except Exception as exc:
        logger.warning(
            "check_symbolic_logic | eval failed safely: %s",
            exc,
        )
        return {
            "has_logic": False,
            "passed": True,
            "note": "Could not evaluate: %s" % exc,
        }