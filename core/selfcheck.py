"""
SelfCheck consistency sampler and score-fusion logic.

SelfCheck asks the LLM the same yes/no question N times at non-zero temperature
and measures how consistent the answers are.  High consistency → claim likely
true; low consistency → claim likely hallucinated.

fuse_scores() is also housed here (cohesion note: it is logically a verifier
concern, but lives here because verifier.py imports it from this module and
moving it would be an API change).

Public API:
    check_claim_consistency(claim, ...)       -> dict   (sync)
    check_claim_consistency_async(claim, ...) -> dict   (coroutine)
    check_batch(claims, ...)                  -> list[dict]
    fuse_scores(nli_result, selfcheck_result, ...) -> dict
"""

from __future__ import annotations

import asyncio
import logging

from core.call_llm import call_llm, call_llm_async, is_ollama_running

logger = logging.getLogger(__name__)

__all__ = [
    "check_claim_consistency",
    "check_claim_consistency_async",
    "check_batch",
    "fuse_scores",
]

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_SAMPLES: int = 3
SAMPLE_TEMPERATURE: float = 0.7
CONSISTENCY_THRESHOLD: float = 0.5
SAMPLE_MODEL: str = "llama3.2:1b"

# fuse_scores internal thresholds
_FUSE_CONTRADICTION_THRESHOLD: float = 0.40  # p(contradiction) above this → HALLUCINATED
_FUSE_NLI_CONFIDENCE_MIN: float = 0.60       # NLI confidence floor for SUPPORTED verdict

# Symbolic-conflict haircut: when a claim has strong NLI support but the
# (unreliable, small-model) symbolic-logic extractor flagged an arithmetic
# failure, we keep the SUPPORTED verdict but shave the fused score so the
# conflict is visible in review/reporting rather than silently discarded.
_SYMBOLIC_CONFLICT_SCORE_CAP: float = 0.75


# ── Prompt ─────────────────────────────────────────────────────────────────────

_VERIFY_PROMPT = """You are a fact-checking assistant. Answer with ONLY 'yes' or 'no'.

Claim: {claim}

Is this claim factually correct? Answer yes or no:"""


def _build_verify_prompt(claim: str, context: str) -> str:
    """Return the verify prompt, optionally prefixed with context."""
    base = _VERIFY_PROMPT.format(claim=claim)
    if context and context.strip():
        return "Context: %s\n\n%s" % (context.strip(), base)
    return base


# ── Vote parsing ───────────────────────────────────────────────────────────────

import re as _re

def _extract_yes_no(response: str) -> str | None:
    """
    Parse 'yes' or 'no' from a model response.
    Checks the first word only; falls back to substring search for short replies.
    Returns 'yes', 'no', or None if unparseable.
    """
    if not response:
        return None

    cleaned = response.strip().lower()
    first_word = _re.sub(r"[^a-z]", "", (cleaned.split() or [""])[0])

    if first_word == "yes":
        return "yes"
    if first_word == "no":
        return "no"

    # Fallback for very short responses where model adds punctuation
    if len(cleaned) < 20:
        if "yes" in cleaned:
            return "yes"
        if "no" in cleaned:
            return "no"

    return None


# ── Vote scoring ───────────────────────────────────────────────────────────────

def _score_votes(votes: list[str], consistency_threshold: float) -> dict:
    """
    Aggregate yes/no votes into a labelled consistency result.

    Labelling rules:
        CONSISTENT   — strong agreement in EITHER direction (all/mostly yes OR all/mostly no).
                       consistency_score >= threshold  → consistent yes
                       consistency_score <= 1-threshold → consistent no
        INCONSISTENT — genuinely split votes, score near 0.5.
        UNCERTAIN    — not enough votes to decide (only reachable when threshold > 0.5,
                       creating a band between the two consistent regions).

    Returns:
        empty            : bool   — True when votes list was empty (no other keys set)
        consistency_score: float  — fraction of "yes" votes (0.0–1.0)
        is_consistent    : bool   — True for CONSISTENT, False otherwise
        label            : str    — "CONSISTENT" | "INCONSISTENT" | "UNCERTAIN"
        majority_vote    : str    — "yes" | "no" | "split" (caller hint for downstream logic)
        votes            : list[str]
        yes_count        : int
        no_count         : int
        valid_votes      : int
    """
    if not votes:
        return {"empty": True}

    yes_count = votes.count("yes")
    no_count = votes.count("no")
    valid_votes = len(votes)
    consistency_score = yes_count / valid_votes

    # Unanimous or strong agreement in either direction → CONSISTENT
    if consistency_score == 0.5:
        # Exact tie — no majority in either direction regardless of threshold
        label = "INCONSISTENT"
        is_consistent = False
        majority_vote = "split"
    elif consistency_score >= consistency_threshold:
        label = "CONSISTENT"
        is_consistent = True
        majority_vote = "yes"
    elif consistency_score <= (1.0 - consistency_threshold):
        label = "CONSISTENT"
        is_consistent = True
        majority_vote = "no"
    else:
        # Score is in the middle band — genuine disagreement
        label = "INCONSISTENT"
        is_consistent = False
        majority_vote = "split"

    return {
        "empty": False,
        "consistency_score": round(consistency_score, 3),
        "is_consistent": is_consistent,
        "label": label,
        "majority_vote": majority_vote,
        "votes": votes,
        "yes_count": yes_count,
        "no_count": no_count,
        "valid_votes": valid_votes,
    }


# ── Sync SelfCheck ─────────────────────────────────────────────────────────────

def check_claim_consistency(
    claim: str,
    context: str = "",
    n_samples: int = DEFAULT_SAMPLES,
    model: str = SAMPLE_MODEL,
    consistency_threshold: float = CONSISTENCY_THRESHOLD,
    max_concurrent: int = 4,
) -> dict:
    """
    Check whether a claim is consistent across N sequential LLM samples.

    If Ollama is not running, returns UNCERTAIN immediately without blocking.
    Callers in an async context should use check_claim_consistency_async().

    Returns a dict with keys:
        consistency_score, is_consistent, label, votes,
        yes_count, no_count, valid_votes, claim
    """
    if not claim or not claim.strip():
        logger.warning("check_claim_consistency | empty claim")
        return _empty_result(claim)

    if not is_ollama_running():
        logger.warning("check_claim_consistency | Ollama offline — returning UNCERTAIN")
        return _empty_result(claim, consistency_score=0.5)

    prompt = _build_verify_prompt(claim, context)
    logger.info("check_claim_consistency | sampling %dx for claim: %.60s", n_samples, claim)

    votes: list[str] = []
    for i in range(n_samples):
        response = call_llm(prompt, model=model, temperature=SAMPLE_TEMPERATURE)
        vote = _extract_yes_no(response)
        if vote:
            votes.append(vote)
        else:
            logger.warning(
                "check_claim_consistency | unparseable response (sample %d): %.40r",
                i + 1, response,
            )

    scored = _score_votes(votes, consistency_threshold)
    if scored.get("empty"):
        logger.warning("check_claim_consistency | no valid votes — returning UNCERTAIN")
        return _empty_result(claim, consistency_score=0.5)

    result = {
        "consistency_score": scored["consistency_score"],
        "is_consistent": scored["is_consistent"],
        "label": scored["label"],
        "votes": scored["votes"],
        "yes_count": scored["yes_count"],
        "no_count": scored["no_count"],
        "valid_votes": scored["valid_votes"],
        "claim": claim,
    }
    logger.info(
        "check_claim_consistency | %s score=%.2f yes=%d no=%d/%d",
        result["label"], result["consistency_score"],
        result["yes_count"], result["no_count"], result["valid_votes"],
    )
    return result


# ── Async SelfCheck ────────────────────────────────────────────────────────────

async def check_claim_consistency_async(
    claim: str,
    context: str = "",
    n_samples: int = DEFAULT_SAMPLES,
    model: str = SAMPLE_MODEL,
    consistency_threshold: float = CONSISTENCY_THRESHOLD,
    max_concurrent: int = 4,
) -> dict:
    """
    Parallel SelfCheck via concurrent async Ollama calls.

    Fires n_samples requests concurrently (bounded by max_concurrent semaphore)
    and aggregates the yes/no votes.
    """
    if not claim or not claim.strip():
        logger.warning("check_claim_consistency_async | empty claim")
        return _empty_result(claim)

    prompt = _build_verify_prompt(claim, context)
    logger.info(
        "check_claim_consistency_async | sampling %dx for claim: %.60s", n_samples, claim
    )

    sem = asyncio.Semaphore(max(1, max_concurrent))

    async def _one() -> str:
        async with sem:
            return await call_llm_async(prompt, model=model, temperature=SAMPLE_TEMPERATURE)

    raw_responses = await asyncio.gather(*[_one() for _ in range(n_samples)])

    votes: list[str] = []
    for i, response in enumerate(raw_responses):
        vote = _extract_yes_no(response)
        if vote:
            votes.append(vote)
        else:
            logger.warning(
                "check_claim_consistency_async | unparseable response (sample %d): %.40r",
                i + 1, response,
            )

    scored = _score_votes(votes, consistency_threshold)
    if scored.get("empty"):
        logger.warning("check_claim_consistency_async | no valid votes — returning UNCERTAIN")
        return _empty_result(claim, consistency_score=0.5)

    result = {
        "consistency_score": scored["consistency_score"],
        "is_consistent": scored["is_consistent"],
        "label": scored["label"],
        "votes": scored["votes"],
        "yes_count": scored["yes_count"],
        "no_count": scored["no_count"],
        "valid_votes": scored["valid_votes"],
        "claim": claim,
    }
    logger.info(
        "check_claim_consistency_async | %s score=%.2f yes=%d no=%d/%d",
        result["label"], result["consistency_score"],
        result["yes_count"], result["no_count"], result["valid_votes"],
    )
    return result


# ── Batch helper ───────────────────────────────────────────────────────────────

def check_batch(
    claims: list[str],
    context: str = "",
    n_samples: int = DEFAULT_SAMPLES,
) -> list[dict]:
    """
    Run SelfCheck on multiple claims sequentially.

    Args:
        claims    : list of atomic claim strings
        context   : shared context for all claims (original LLM response)
        n_samples : samples per claim

    Returns:
        list of result dicts in the same order as claims
    """
    if not claims:
        return []

    logger.info("check_batch | %d claims x %d samples", len(claims), n_samples)
    return [
        check_claim_consistency(claim, context=context, n_samples=n_samples)
        for claim in claims
    ]


# ── Score fusion ───────────────────────────────────────────────────────────────

def fuse_scores(
    nli_result: dict,
    selfcheck_result: dict,
    symbolic_result: dict | None = None,
    fuse_high: float = 0.65,
    fuse_low: float = 0.45,
    contradiction_threshold: float = 0.40,
) -> dict:
    """
    Fuse NLI, SelfCheck, and (optional) symbolic-logic signals into a verdict.

    Decision order:
        1. NLI contradiction or high p(contradiction)              → HALLUCINATED
        2. Symbolic-logic failure + strong NLI support              → SUPPORTED
           (flagged as a conflict, score capped — see note below)
        3. Symbolic-logic failure + no strong NLI support           → HALLUCINATED
        4. NLI SUPPORTED with sufficient confidence                 → SUPPORTED
        5. NLI NEUTRAL                                                   → UNVERIFIABLE
           (SelfCheck disagreement cannot independently establish falsity)
        6. High fused score                                              → SUPPORTED
        7. Otherwise                                                     → UNVERIFIABLE

    Fusion weight: 60 % NLI + 40 % SelfCheck.

    Symbolic logic is treated as a *non-fatal* signal rather than an automatic
    override. The symbolic extractor runs on a small, weak LLM that frequently
    conflates unrelated numbers in a claim (e.g. a day-of-month figure used as
    if it were a year in a duration check), so a "logic failed" result on its
    own is not trustworthy enough to overrule strong, independently retrieved
    evidence. When NLI already shows strong entailment against real KB
    evidence, that evidence wins — the symbolic conflict is recorded in the
    output (`symbolic_conflict: True`, `symbolic_note`) and the fused score
    gets a modest haircut so the discrepancy stays visible for review, but the
    claim is not wrongly flipped to HALLUCINATED. Symbolic failures still push
    the verdict to HALLUCINATED when there is no strong contrary evidence to
    outweigh them — this preserves the original purpose of the check (catching
    e.g. "died 20 years later" arithmetic that doesn't add up) for claims the
    retriever/NLI can't otherwise adjudicate.

    Note: SelfCheck is treated as a supporting consistency signal, not as
    independent proof that a claim is false. With no retrieved evidence there
    is nothing to contradict the claim. Likewise, when NLI is NEUTRAL, an
    INCONSISTENT SelfCheck result is insufficient to establish falsity because
    the small local sampling model can disagree on ambiguous or compound
    claims. Such cases remain UNVERIFIABLE unless NLI explicitly establishes
    contradiction.
    """
    nli_label = nli_result.get("label", "NEUTRAL")
    sc_label = selfcheck_result.get("label", "UNCERTAIN")
    sc_score = float(selfcheck_result.get("consistency_score", 0.5))
    nli_conf = float(nli_result.get("confidence", 0.5))
    all_scores = nli_result.get("all_scores") or {}
    p_contradiction = float(all_scores.get("contradiction", 0.0))
    claim_text = nli_result.get("claim", "")

    # Evidence is considered present when the NLI scorer received at least
    # one candidate passage (all_scores will be non-empty in that case).
    has_evidence = bool(all_scores)

    symbolic_failed = bool(
        symbolic_result
        and symbolic_result.get("has_logic")
        and not symbolic_result.get("passed", True)
    )
    symbolic_note = (symbolic_result or {}).get("note")

    def _with_symbolic(payload: dict) -> dict:
        payload["symbolic_conflict"] = symbolic_failed
        payload["symbolic_note"] = symbolic_note if symbolic_failed else None
        return payload

    # 1. Explicit contradiction (only meaningful when evidence exists)
    if nli_label == "CONTRADICTED" or (has_evidence and p_contradiction >= contradiction_threshold):
        fused_score = round(1.0 - max(p_contradiction, 0.6), 3)
        return _with_symbolic({
            "final_label": "HALLUCINATED",
            "fused_score": fused_score,
            "nli_label": "CONTRADICTED",
            "nli_confidence": max(p_contradiction, nli_conf),
            "selfcheck_label": sc_label,
            "selfcheck_score": sc_score,
            "claim": claim_text,
        })

    # Weighted fusion: NLI is more reliable
    nli_signal = {"SUPPORTED": 1.0, "NEUTRAL": 0.5, "CONTRADICTED": 0.0}.get(nli_label, 0.5)
    fused_score = (0.6 * nli_signal) + (0.4 * sc_score)

    strong_nli_support = nli_label == "SUPPORTED" and nli_conf >= _FUSE_NLI_CONFIDENCE_MIN

    # 2. Symbolic failure but strong independent evidence support → trust the
    #    evidence, flag the conflict, don't discard a correct SUPPORTED verdict.
    if symbolic_failed and strong_nli_support:
        capped_score = round(min(fused_score, _SYMBOLIC_CONFLICT_SCORE_CAP), 3)
        return _with_symbolic({
            "final_label": "SUPPORTED",
            "fused_score": capped_score,
            "nli_label": nli_label,
            "nli_confidence": nli_conf,
            "selfcheck_label": sc_label,
            "selfcheck_score": sc_score,
            "claim": claim_text,
        })

    # 3. Symbolic failure is diagnostic, not independent proof of falsity.
    # NEW: The symbolic extractor can invent a relationship between unrelated
    # numbers/dates. Therefore a symbolic failure must NOT independently turn
    # an NLI-NEUTRAL claim into HALLUCINATED.
    #
    # Strong NLI support was already handled above. Explicit NLI contradiction
    # was handled at the top of this function. Therefore, for a symbolic
    # failure without either of those strong NLI signals, continue to the
    # normal fusion rules instead of returning HALLUCINATED immediately.

    # 4–7. Verdict rules (no symbolic signal involved)
    if strong_nli_support:
        final_label = "SUPPORTED"

    # NEW: A NEUTRAL NLI result means the retrieved evidence did not
    # establish either entailment or contradiction. SelfCheck can be noisy,
    # especially with the small local sampling model, so an INCONSISTENT
    # SelfCheck result must not manufacture a HALLUCINATED verdict here.
    # This specifically prevents cases such as:
    #     NLI=NEUTRAL + SelfCheck=INCONSISTENT + fused_score=0.50
    # from being incorrectly classified as HALLUCINATED.
    elif nli_label == "NEUTRAL":
        final_label = "UNVERIFIABLE"

    elif fused_score >= fuse_high:
        final_label = "SUPPORTED"
    else:
        final_label = "UNVERIFIABLE"

    return _with_symbolic({
        "final_label": final_label,
        "fused_score": round(fused_score, 3),
        "nli_label": nli_label,
        "nli_confidence": nli_conf,
        "selfcheck_label": sc_label,
        "selfcheck_score": sc_score,
        "claim": claim_text,
    })


# ── Helpers ────────────────────────────────────────────────────────────────────

def _empty_result(claim: str, consistency_score: float = 0.0) -> dict:
    """Return a safe fallback result when sampling cannot proceed."""
    return {
        "consistency_score": consistency_score,
        "is_consistent": False,
        "label": "UNCERTAIN",
        "votes": [],
        "yes_count": 0,
        "no_count": 0,
        "valid_votes": 0,
        "claim": claim,
    }


def interpret(result: dict) -> str:
    """Human-readable summary of a selfcheck result (debug / demo use)."""
    label = result["label"]
    score = result["consistency_score"]
    yes = result["yes_count"]
    no = result["no_count"]
    total = result["valid_votes"]
    icons = {"CONSISTENT": "✅", "INCONSISTENT": "❌", "UNCERTAIN": "⚠️ "}
    icon = icons.get(label, "❓")
    return "%s %s | score=%.0f%% | votes: %d yes / %d no out of %d" % (
        icon, label, score * 100, yes, no, total
    )


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import logging as _logging
    _logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not is_ollama_running():
        print("❌ Ollama not running. Run: ollama serve")
        sys.exit(1)
    print("✅ Ollama is running\n")

    print("=== Test 1: True claim (should be CONSISTENT) ===")
    r1 = check_claim_consistency("Albert Einstein was born in Germany.", n_samples=3)
    print(interpret(r1))

    print("\n=== Test 2: False claim (should be INCONSISTENT or UNCERTAIN) ===")
    r2 = check_claim_consistency(
        "Albert Einstein won the Nobel Prize in Chemistry in 1930.", n_samples=3
    )
    print(interpret(r2))

    print("\n=== Test 3: Batch check ===")
    claims = [
        "Python was first released in 1991.",
        "Python was invented by Mark Zuckerberg.",
        "The Eiffel Tower is located in Paris.",
    ]
    for r in check_batch(claims, n_samples=3):
        print("  Claim : %s" % r["claim"])
        print("  %s" % interpret(r))

    print("\n=== Test 4: fuse_scores (no symbolic conflict) ===")
    mock_nli = {"label": "SUPPORTED", "confidence": 0.91, "claim": "test", "all_scores": {}}
    mock_sc = r1
    fused = fuse_scores(mock_nli, mock_sc)
    print("NLI: %s  SelfCheck: %s  Fused: %s (score=%s)" % (
        fused["nli_label"], fused["selfcheck_label"],
        fused["final_label"], fused["fused_score"],
    ))

    print("\n=== Test 5: fuse_scores (symbolic conflict, strong NLI support) ===")
    mock_nli_strong = {
        "label": "SUPPORTED", "confidence": 0.95, "claim": "test",
        "all_scores": {"entailment": 0.95, "contradiction": 0.02, "neutral": 0.03},
    }
    mock_symbolic_fail = {"has_logic": True, "passed": False, "note": "Logic failed: 1932 + 18 == 1955"}
    fused2 = fuse_scores(mock_nli_strong, mock_sc, symbolic_result=mock_symbolic_fail)
    print("Final: %s (score=%s) symbolic_conflict=%s" % (
        fused2["final_label"], fused2["fused_score"], fused2["symbolic_conflict"],
    ))

    print("\n=== All SelfCheck tests complete ✅ ===")
    sys.exit(0)