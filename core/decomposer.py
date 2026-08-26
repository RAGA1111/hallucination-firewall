"""
Claim decomposer — splits an LLM response into verifiable atomic claims.

Uses Ollama (llama3.2:1b) when available; falls back to a pure-Python
heuristic sentence splitter when Ollama is offline.

Public API:
    decompose_claims(text, ...)        -> DecompositionResult   (sync)
    decompose_claims_async(text, ...)  -> DecompositionResult   (coroutine)
    decompose_batch(texts)             -> list[list[str]]
    decompose_without_llm(text)        -> DecompositionResult
    ClaimDecompositionError
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

from core.call_llm import call_llm, call_llm_async, is_ollama_running

logger = logging.getLogger(__name__)

__all__ = [
    "decompose_claims",
    "decompose_claims_async",
    "decompose_batch",
    "decompose_without_llm",
    "DecompositionResult",
    "ClaimDecompositionError",
]

# ── Constants ──────────────────────────────────────────────────────────────────

MIN_CLAIM_LENGTH: int = 10          # minimum chars for a parsed claim to be kept
MIN_FACTUAL_SENTENCE_LENGTH: int = 15  # minimum chars for heuristic sentence filter


# ── Prompt Templates ───────────────────────────────────────────────────────────

_DECOMPOSE_PROMPT = """Extract verifiable factual claims from the text below.

Rules:
- Each output item must contain ONE independently verifiable factual claim.
- Split compound sentences into separate claims when they contain multiple
  independently verifiable facts.
- In particular, split facts joined by "and", "but", "while", "also", etc.
  when each part can be verified independently.
- Do NOT combine multiple events, dates, people, places, or facts into one claim.
- Preserve the original meaning and factual wording.
- Do not invent facts or add information that is not present in the text.
- Keep dates, numbers, names, places, and other important details exactly as stated.
- Skip opinions, vague statements, and hedged statements.
- Return ONLY a JSON array of strings.

Example 1:

Text:
"Einstein emigrated to the United States in 1932 and died on April 18, 1955."

Output:
[
  "Einstein emigrated to the United States in 1932.",
  "Einstein died on April 18, 1955."
]

Example 2:

Text:
"Python was created by Guido van Rossum and first released in 1991."

Output:
[
  "Python was created by Guido van Rossum.",
  "Python was first released in 1991."
]

Example 3:

Text:
"Einstein was born on March 14, 1879, in Ulm, Germany."

Output:
[
  "Einstein was born on March 14, 1879, in Ulm, Germany."
]

Example 4:

Text:
"Python is popular and easy to learn."

Output:
[]

Text:
{text}

JSON array:"""

_RETRY_FALLBACK_PROMPT = """Extract atomic, independently verifiable factual claims
from the text as a JSON array of strings.

Rules:
- One independently verifiable fact per claim.
- Split compound factual statements joined by "and", "but", "while", etc.
- Do not combine separate events or facts into one claim.
- Preserve names, dates, numbers, places, and factual wording exactly.
- Do not invent or add facts.
- Return ONLY the JSON array.

Example:

Text:
"Einstein emigrated to the United States in 1932 and died on April 18, 1955."

Output:
[
  "Einstein emigrated to the United States in 1932.",
  "Einstein died on April 18, 1955."
]

Text:
{text}

JSON array:"""

_FILTER_PROMPT = """Filter the claims below.

Rules:
- Keep only factual, independently verifiable claims.
- Each claim must contain ONE independently verifiable fact.
- If a claim contains multiple independently verifiable facts joined together,
  split it into separate claims.
- Do not merge separate claims.
- Preserve the original factual meaning.
- Do not invent or add facts.
- Keep names, dates, numbers, and places exactly as stated.
- Remove opinions, vague statements, and unsupported speculation.

Return ONLY a JSON array of strings.

Claims:
{claims}

JSON array:"""


# ── Types ──────────────────────────────────────────────────────────────────────

@dataclass
class DecompositionResult:
    claims: list[str]
    success: bool
    error: str | None
    raw_response: str


class ClaimDecompositionError(Exception):
    def __init__(self, message: str, raw_response: str = "") -> None:
        super().__init__(message)
        self.raw_response = raw_response


# ── Heuristic signals (compiled once) ─────────────────────────────────────────

_OPINION_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\b(i think|i believe|in my opinion|it seems|perhaps|maybe|could be|might be)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(yes|no|sure|okay|well|so|actually|basically|generally|overall)\.?$",
        re.IGNORECASE,
    ),
]

_FACTUAL_SIGNALS: re.Pattern = re.compile(
    r"\b(\d{4}|born|died|founded|created|released|published|invented|won|awarded|"
    r"located in|headquartered|president|prime minister|capital|population|"
    r"university|institute|january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT: re.Pattern = re.compile(r"(?<=[.!?])\s+")


# ── Markdown fence stripping ─────────────────────────────────────────────────

def _strip_markdown_fences(text: str) -> str:
    """
    Strip ```json / ``` code fences some LLMs wrap output in.

    Small local models frequently wrap JSON output in markdown code fences
    (```json ... ``` or ``` ... ```) even when explicitly told not to. This
    breaks json.loads() (leading/trailing backticks are not valid JSON) and,
    without this stripping step, causes the entire fenced blob to fall
    through to the sentence-splitter as one giant unparseable "claim".
    """
    if not text:
        return text
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"```\s*$", "", s)
    return s.strip()


# ── Parse helpers ──────────────────────────────────────────────────────────────

def parse_json_claims(text: str) -> list[str]:
    """Parse a JSON array or object containing claim strings."""
    text = _strip_markdown_fences(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    if isinstance(data, list):
        return [
            str(item).strip()
            for item in data
            if isinstance(item, str) and len(item.strip()) > MIN_CLAIM_LENGTH
        ]

    if isinstance(data, dict):
        for key in ("claims", "items", "facts", "data"):
            if key in data and isinstance(data[key], list):
                return [
                    str(item).strip()
                    for item in data[key]
                    if isinstance(item, str) and len(item.strip()) > MIN_CLAIM_LENGTH
                ]

    return []


def parse_numbered_list(text: str) -> list[str]:
    """Parse numbered lists like `1.` `1)` `1:` from LLM output."""
    claims = []
    for line in text.strip().split("\n"):
        match = re.match(r"^\d+[\.\)\:\-]\s*(.+)$", line.strip())
        if match:
            claim = match.group(1).strip()
            if len(claim) > MIN_CLAIM_LENGTH:
                claims.append(claim)
    return claims


def parse_bullet_list(text: str) -> list[str]:
    """Parse bullet-style lists from LLM output."""
    claims = []
    for line in text.strip().split("\n"):
        match = re.match(r"^[\-\*\u2022]\s*(.+)$", line.strip())
        if match:
            claim = match.group(1).strip()
            if len(claim) > MIN_CLAIM_LENGTH:
                claims.append(claim)
    return claims


def parse_sentences(text: str) -> list[str]:
    """Last-resort fallback: split on sentence boundaries."""
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [
        s.strip(" \n\t-–—")
        for s in sentences
        if len(s.strip()) >= 20 and any(c.isalnum() for c in s)
    ]


# ── Malformed-structure guard (Bug 8 fix, extended for fenced blobs) ──────────

def _looks_like_malformed_structure(claim: str) -> bool:
    """
    Return True if a parsed 'claim' string is actually leftover structured
    data (e.g. a JSON object/array the LLM emitted, possibly wrapped in
    markdown fences) rather than a genuine natural-language claim.

    This guards against two related failure modes where the LLM ignores
    the "JSON array of strings" instruction:
        1. Returns a JSON *object* (dict) instead of an array.
        2. Wraps its JSON output in markdown code fences (``` ... ```),
           which breaks json.loads() and lets the whole blob fall through
           to the sentence-splitter as one giant unparseable "claim".
    Either way, such a blob typically contains no sentence-ending
    punctuation that would let parse_sentences() split it correctly, so it
    can slip through as a single atomic "claim" — silently bundling
    multiple facts (some possibly false) into one string that verification
    can't meaningfully check.
    """
    s = claim.strip()
    if not s:
        return False
    # Strip markdown fences first so a fenced-but-otherwise-malformed blob
    # is still correctly identified as structured data underneath.
    s = _strip_markdown_fences(s)
    if s.startswith("{") or s.startswith("["):
        return True
    # Heuristic: multiple `"key": "value"`-style pairs strongly suggests
    # leftover JSON that wasn't caught by the JSON parser (e.g. because it
    # was embedded inside a larger string rather than being valid JSON on
    # its own).
    if s.count('": "') >= 2 or s.count("':'") >= 2:
        return True
    return False


def _filter_malformed(claims: list[str]) -> list[str]:
    """Drop any claim that looks like leftover structured data."""
    kept = [c for c in claims if not _looks_like_malformed_structure(c)]
    dropped = len(claims) - len(kept)
    if dropped:
        logger.warning(
            "[Decomposer] dropped %d malformed-structure claim(s) "
            "(leftover JSON/dict rather than a sentence)",
            dropped,
        )
    return kept


def _extract_claims(raw_output: str) -> list[str]:
    """
    Try parsers in priority order until claims are extracted.

    Each parser's output is passed through _filter_malformed() before being
    accepted — if a parser's result is entirely leftover structured data
    (e.g. a JSON object/fenced-array dumped as one "sentence"), it is
    treated as if that parser found nothing, and the next parser in the
    chain is tried instead.
    """
    for parser in (parse_json_claims, parse_numbered_list, parse_bullet_list, parse_sentences):
        claims = parser(raw_output)
        if not claims:
            continue
        claims = _filter_malformed(claims)
        if claims:
            return claims
    return []



# ── Deterministic atomic-claim splitter ───────────────────────────────────────

# NEW: The small local LLM is not trusted to be the only component that
# separates compound factual statements. This deterministic layer also runs
# when the LLM returns malformed JSON and the Python fallback is used.
#
# NOTE on _PREDICATE_START coverage: this list must include verbs that start
# a subjective/opinion clause (e.g. "considered", "regarded", "believed"),
# not just neutral factual verbs. Without them, a sentence like
# "X was known to play the violin and considered music his greatest passion"
# never gets split — the right-hand clause doesn't match any known predicate
# start, so the whole sentence survives as ONE merged claim. That merged
# claim then hits the opinion-filtering pass (_FILTER_PROMPT) as a single
# unit; the filter model sees the opinion half ("considered... greatest
# passion") and drops the *entire* claim, silently discarding the genuinely
# verifiable half ("played the violin") along with it. Splitting first lets
# each half be judged/filtered independently.
_PREDICATE_START = re.compile(
    r"^(?:was|were|is|are|has|have|had|won|died|emigrated|immigrated|"
    r"moved|joined|left|created|founded|invented|developed|published|"
    r"released|awarded|received|became|served|worked|studied|"
    r"considered|regarded|described|viewed|known|believed|thought|"
    r"played|composed|wrote|painted|discovered|proposed|introduced|"
    r"first\s+released|later\s+released)\b",
    re.IGNORECASE,
)

_SUBJECT_VERB = re.compile(
    r"^(?P<subject>.+?)\s+"
    r"(?:was|were|is|are|has|have|had|won|died|emigrated|immigrated|"
    r"moved|joined|left|created|founded|invented|developed|published|"
    r"released|awarded|received|became|served|worked|studied)\b",
    re.IGNORECASE,
)


def _split_compound_claim(claim: str) -> list[str]:
    """Split one sentence when a conjunction joins two factual predicates."""
    body = claim.strip().strip(" \n\t-–—").rstrip(".!?").strip()
    if not body:
        return []

    for conjunction in (" and ", " but ", " while "):
        parts = re.split(re.escape(conjunction), body, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) != 2:
            continue

        left, right = parts[0].strip(), parts[1].strip()

        # Do not blindly split every "and". The second part must look like
        # the beginning of a new factual predicate.
        if not _PREDICATE_START.match(right):
            continue

        match = _SUBJECT_VERB.match(left)
        if not match:
            continue

        subject = match.group("subject").strip()
        if not subject:
            continue

        first = left + "."
        second = f"{subject} {right}."

        if len(first) >= MIN_CLAIM_LENGTH and len(second) >= MIN_CLAIM_LENGTH:
            logger.info(
                "[Decomposer] atomic split | %r -> %r + %r",
                claim, first, second
            )
            return [first, second]

    return [body + "."]


def _split_claims_atomically(claims: list[str]) -> list[str]:
    """Apply deterministic atomic splitting and remove duplicates."""
    result: list[str] = []

    for claim in claims:
        result.extend(_split_compound_claim(claim))

    seen: set[str] = set()
    unique: list[str] = []

    for claim in result:
        normalized = re.sub(r"\s+", " ", claim.strip()).lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(claim.strip())

    return unique


# ── Heuristic sentence filter ──────────────────────────────────────────────────

def _is_factual_sentence(sentence: str) -> bool:
    """Return True if the sentence looks like a verifiable factual claim."""
    s = sentence.strip()
    if len(s) < MIN_FACTUAL_SENTENCE_LENGTH:
        return False
    for pattern in _OPINION_PATTERNS:
        if pattern.search(s):
            return False
    if _FACTUAL_SIGNALS.search(s):
        return True
    words = s.split()
    has_proper = any(w[0].isupper() for w in words[1:] if w and w[0].isalpha())
    has_number = bool(re.search(r"\d", s))
    return has_proper or has_number


# ── Input validation ───────────────────────────────────────────────────────────

def _validate_input(text: str) -> None:
    if not text or not text.strip():
        raise ClaimDecompositionError("Empty input provided to decomposer.", raw_response=text or "")


# ── Shared LLM parse logic (used by both sync and async paths) ─────────────────

def _parse_llm_output(raw_output: str) -> list[str]:
    """
    Extract claims from a raw LLM response string and enforce atomicity.
    """
    return _split_claims_atomically(_extract_claims(raw_output))


def _apply_filter(claims: list[str], filter_response: str) -> list[str]:
    """Apply filter output and enforce atomic claims."""
    filtered = _extract_claims(filter_response)
    return _split_claims_atomically(filtered if filtered else claims)


# ── Sync decomposition ─────────────────────────────────────────────────────────

def _decompose_via_llm(text: str, *, filter_claims: bool) -> DecompositionResult:
    start = time.time()
    logger.info("[Decomposer] START | input_chars=%d", len(text))

    raw_output = call_llm(_DECOMPOSE_PROMPT.format(text=text), temperature=0.0)

    if not raw_output:
        logger.warning("[Decomposer] empty LLM response | elapsed=%.2fs", time.time() - start)
        return decompose_without_llm(text)

    claims = _parse_llm_output(raw_output)
    logger.info("[Decomposer] parsed %d raw claims", len(claims))

    if not claims:
        logger.warning("[Decomposer] retry — initial parse returned no claims")
        retry_output = call_llm(_RETRY_FALLBACK_PROMPT.format(text=text), temperature=0.0)
        if retry_output:
            claims = _parse_llm_output(retry_output)
            logger.info("[Decomposer] retry parsed %d claims", len(claims))

    if not claims:
        logger.warning("[Decomposer] FAILURE | no claims parsed | elapsed=%.2fs", time.time() - start)
        return decompose_without_llm(text)

    if filter_claims:
        before = len(claims)
        filter_response = call_llm(
            _FILTER_PROMPT.format(claims="\n".join(claims)), temperature=0.0
        )
        if filter_response:
            claims = _apply_filter(claims, filter_response)
        logger.info("[Decomposer] filter pass: %d -> %d claims", before, len(claims))

    logger.info("[Decomposer] SUCCESS | claims=%d | elapsed=%.2fs", len(claims), time.time() - start)
    return DecompositionResult(claims=claims, success=True, error=None, raw_response=raw_output)


# ── Async decomposition ────────────────────────────────────────────────────────

async def _decompose_via_llm_async(text: str, *, filter_claims: bool) -> DecompositionResult:
    start = time.time()
    logger.info("[Decomposer] START (async) | input_chars=%d", len(text))

    raw_output = await call_llm_async(_DECOMPOSE_PROMPT.format(text=text), temperature=0.0)

    if not raw_output:
        logger.warning("[Decomposer] empty LLM response (async) | elapsed=%.2fs", time.time() - start)
        return decompose_without_llm(text)

    claims = _parse_llm_output(raw_output)
    logger.info("[Decomposer] parsed %d raw claims (async)", len(claims))

    if not claims:
        logger.warning("[Decomposer] retry (async) — initial parse returned no claims")
        retry_output = await call_llm_async(
            _RETRY_FALLBACK_PROMPT.format(text=text), temperature=0.0
        )
        if retry_output:
            claims = _parse_llm_output(retry_output)
            logger.info("[Decomposer] retry (async) parsed %d claims", len(claims))

    if not claims:
        logger.warning(
            "[Decomposer] FAILURE (async) | no claims parsed | elapsed=%.2fs", time.time() - start
        )
        return decompose_without_llm(text)

    if filter_claims:
        before = len(claims)
        filter_response = await call_llm_async(
            _FILTER_PROMPT.format(claims="\n".join(claims)), temperature=0.0
        )
        if filter_response:
            claims = _apply_filter(claims, filter_response)
        logger.info("[Decomposer] filter pass (async): %d -> %d claims", before, len(claims))

    logger.info(
        "[Decomposer] SUCCESS (async) | claims=%d | elapsed=%.2fs", len(claims), time.time() - start
    )
    return DecompositionResult(claims=claims, success=True, error=None, raw_response=raw_output)


# ── Public API ─────────────────────────────────────────────────────────────────

def decompose_without_llm(text: str) -> DecompositionResult:
    """
    Pure-Python fallback decomposer — no Ollama required.

    Splits input into sentences, splits compound factual sentences into atomic
    claims, then filters for verifiable factual signals (dates, named entities, numbers).
    Used automatically when Ollama is not running.
    """
    start = time.time()
    logger.info("[Decomposer] START (fallback) | input_chars=%d", len(text or ""))

    _validate_input(text)

    raw_sentences = _SENTENCE_SPLIT.split(text.strip())
    claims = []

    for sent in raw_sentences:
        candidate = sent.strip(" \n\t-–—").rstrip(".")
        if not candidate:
            continue
        candidate += "."

        # NEW: The fallback must also split compound factual sentences.
        # Otherwise an LLM parsing failure brings back the exact bug we are
        # trying to eliminate.
        for atomic_claim in _split_compound_claim(candidate):
            if _is_factual_sentence(atomic_claim):
                claims.append(atomic_claim)

    claims = _split_claims_atomically(claims)

    if not claims:
        claims = [
            s.strip()
            for s in raw_sentences
            if len(s.strip()) >= MIN_FACTUAL_SENTENCE_LENGTH
        ]

    logger.info(
        "[Decomposer] SUCCESS (fallback) | claims=%d | sentences=%d | elapsed=%.2fs",
        len(claims), len(raw_sentences), time.time() - start,
    )
    return DecompositionResult(claims=claims, success=True, error=None, raw_response=text)


def decompose_claims(text: str, filter_claims: bool = True) -> DecompositionResult:
    """
    Sync decomposition.

    Uses Ollama (llama3.2:1b) when available; falls back to the pure-Python
    sentence-split decomposer when Ollama is not running.
    """
    _validate_input(text)

    if not is_ollama_running():
        logger.warning("[Decomposer] Ollama not running — using Python fallback")
        return decompose_without_llm(text)

    return _decompose_via_llm(text, filter_claims=filter_claims)


async def decompose_claims_async(text: str, filter_claims: bool = True) -> DecompositionResult:
    """
    Async decomposition.

    Uses Ollama when available; falls back to the pure-Python decomposer
    when Ollama is not running.
    """
    _validate_input(text)

    if not is_ollama_running():
        logger.warning("[Decomposer] Ollama not running — using Python fallback (async)")
        return decompose_without_llm(text)

    return await _decompose_via_llm_async(text, filter_claims=filter_claims)


def decompose_batch(texts: list[str]) -> list[list[str]]:
    """
    Decompose multiple LLM responses.

    Args:
        texts : list of response strings

    Returns:
        list of claim lists, one per input text (same order)
    """
    results = []
    for i, text in enumerate(texts):
        logger.info("[Decomposer] batch item %d/%d", i + 1, len(texts))
        results.append(decompose_claims(text).claims)
    return results


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import logging as _logging
    _logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not is_ollama_running():
        print("❌ Ollama is not running. Run: ollama serve")
        sys.exit(1)
    print("✅ Ollama is running\n")

    print("=== Test 1: Clean factual paragraph ===")
    text1 = (
        "Albert Einstein was born on March 14, 1879, in Ulm, Germany. "
        "He developed the theory of relativity. "
        "Einstein won the Nobel Prize in Physics in 1921. "
        "He emigrated to the United States in 1933 and died on April 18, 1955."
    )
    r1 = decompose_claims(text1)
    print(f"Extracted {len(r1.claims)} claims:")
    for i, c in enumerate(r1.claims, 1):
        print(f"  {i}. {c}")

    print("\n=== Test 2: Mixed factual + opinion ===")
    text2 = (
        "I think Python is the best language. "
        "Python was created by Guido van Rossum and first released in 1991. "
        "Python 3.0 was released in December 2008."
    )
    r2 = decompose_claims(text2)
    print(f"Extracted {len(r2.claims)} claims (opinions filtered):")
    for i, c in enumerate(r2.claims, 1):
        print(f"  {i}. {c}")
    print("\n=== Test 3b: Deterministic splitter works without LLM ===")
    fallback_compound = (
        "Einstein emigrated to the United States in 1932 "
        "and died on April 18, 1955."
    )
    fallback_parts = _split_compound_claim(fallback_compound)
    print(f"Deterministic split -> {len(fallback_parts)} claims:")
    for i, c in enumerate(fallback_parts, 1):
        print(f"  {i}. {c}")

    assert len(fallback_parts) == 2
    assert fallback_parts[0] == "Einstein emigrated to the United States in 1932."
    assert fallback_parts[1] == "Einstein died on April 18, 1955."

    python_compound = (
        "Python programming language was created by Guido van Rossum "
        "and first released in 1991."
    )
    python_parts = _split_compound_claim(python_compound)
    assert len(python_parts) == 2
    assert any("created by Guido van Rossum" in c for c in python_parts)
    assert any("first released in 1991" in c for c in python_parts)

    print("\n=== Test 3c: Factual+opinion compound splits so the factual half survives ===")
    violin_compound = (
        "Einstein was also known to play the violin and considered music "
        "his greatest passion in life."
    )
    violin_parts = _split_compound_claim(violin_compound)
    print(f"Deterministic split -> {len(violin_parts)} claims:")
    for i, c in enumerate(violin_parts, 1):
        print(f"  {i}. {c}")
    assert len(violin_parts) == 2
    assert any("play the violin" in c for c in violin_parts)
    assert any("greatest passion" in c for c in violin_parts)

    print("\n=== Test 3: Compound factual claim is split into atomic claims ===")
    compound = (
        "Einstein emigrated to the United States in 1932 "
        "and died on April 18, 1955."
    )

    r_compound = decompose_claims(compound)

    print(f"Extracted {len(r_compound.claims)} claims:")
    for i, c in enumerate(r_compound.claims, 1):
        print(f"  {i}. {c}")

    assert len(r_compound.claims) == 2
    assert any(
        "emigrated" in c.lower() and "1932" in c
        for c in r_compound.claims
    )
    assert any(
        "died" in c.lower() and "1955" in c
        for c in r_compound.claims
    )

    print("\n=== Test 3: Empty input raises ClaimDecompositionError ===")
    try:
        decompose_claims("")
        print("ERROR — expected exception not raised")
        sys.exit(1)
    except ClaimDecompositionError as exc:
        print(f"ClaimDecompositionError raised as expected: {exc}")

    print("\n=== Test 4: Malformed JSON-object output is rejected, not treated as one claim ===")
    malformed = _looks_like_malformed_structure(
        '{\n  "March 14, 1879": "Paris, France",\n  "1921": "Nobel Prize in Chemistry"\n}'
    )
    print(f"_looks_like_malformed_structure() on JSON-object blob -> {malformed} (expected True)")
    assert malformed is True

    normal = _looks_like_malformed_structure(
        "Einstein was born in Ulm, Germany, on March 14, 1879."
    )
    print(f"_looks_like_malformed_structure() on normal sentence -> {normal} (expected False)")
    assert normal is False

    print("\n=== Test 5: Markdown-fenced JSON array parses correctly (not treated as 1 blob) ===")
    fenced = '```json\n["Marie Curie was born on November 7, 1867, in Warsaw, Poland.", "JavaScript was created by Brendan Eich and first released in 1995."]\n```'
    fenced_claims = parse_json_claims(fenced)
    print(f"parse_json_claims() on fenced array -> {len(fenced_claims)} claims (expected 2)")
    for i, c in enumerate(fenced_claims, 1):
        print(f"  {i}. {c}")
    assert len(fenced_claims) == 2
    assert not any(c.startswith("```") or c.startswith("[") for c in fenced_claims)

    fenced_no_lang = '```\n["Fact one is true.", "Fact two is also true."]\n```'
    fenced_no_lang_claims = parse_json_claims(fenced_no_lang)
    print(f"parse_json_claims() on fenced array (no lang tag) -> {len(fenced_no_lang_claims)} claims (expected 2)")
    assert len(fenced_no_lang_claims) == 2

    fenced_malformed = _looks_like_malformed_structure(
        '```\n{"1879": "Paris, France"}\n```'
    )
    print(f"_looks_like_malformed_structure() on fenced JSON object -> {fenced_malformed} (expected True)")
    assert fenced_malformed is True

    print("\n=== All decomposer tests complete ✅ ===")
    sys.exit(0)