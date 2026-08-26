"""
Self-correction module — rewrites hallucinated claims using evidence.

Uses a 3-step Chain-of-Verification (CoVe) approach:
    1. Generate verification questions for the incorrect claim
    2. Answer those questions using only the retrieved evidence
    3. Rewrite the claim from the verified answers

Note: each correction attempt makes 3 sequential LLM calls. With
MAX_ATTEMPTS=2 this is up to 6 LLM calls per hallucinated claim.
There is no per-call timeout beyond call_llm's own timeout parameter.

Public API:
    Regenerator   — instantiate once, call .correct() or .correct_batch()
"""

from __future__ import annotations

import logging

from core.call_llm import call_llm, is_ollama_running
from core.config import DEFAULT_CONFIG, PipelineConfig
from core.nli_scorer import NLIScorer

logger = logging.getLogger(__name__)

__all__ = ["Regenerator"]

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_ATTEMPTS: int = 2
CORRECTION_CONFIDENCE_THRESHOLD: float = 0.65

# Labels that are already fine — skip correction
SKIP_LABELS: frozenset[str] = frozenset({"SUPPORTED"})


# ── Prompt Templates ───────────────────────────────────────────────────────────

_CORRECTION_PROMPT = """You are a fact-correction assistant.

A language model made a factual error. Your job is to rewrite the incorrect claim
using ONLY the information in the evidence provided below.

Rules:
- Use ONLY facts from the evidence — do not add anything else
- Keep the corrected claim as one sentence
- Match the style and length of the original claim
- Do NOT say "According to the evidence" or add any preamble
- Return ONLY the corrected claim sentence, nothing else

Original (incorrect) claim: {claim}
Evidence: {evidence}

Corrected claim:"""

_CANNOT_CORRECT_PROMPT = """You are a fact-checking assistant.

The following claim cannot be verified or corrected because there is no
relevant evidence available.

Write a single sentence that:
- Acknowledges the claim cannot be verified
- Does NOT make up any new facts
- Is neutral and honest

Claim: {claim}

Response:"""


# ── Regenerator ────────────────────────────────────────────────────────────────

class Regenerator:
    """
    Self-correction module — rewrites hallucinated claims using evidence.

    Usage:
        regen = Regenerator()
        result = regen.correct(verify_result)
        results = regen.correct_batch(verify_results)
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        logger.info("Initializing Regenerator...")
        self.config = config or DEFAULT_CONFIG
        self.nli = NLIScorer(
            entailment_threshold=self.config.nli_entailment_threshold,
            contradiction_threshold=self.config.nli_contradiction_threshold,
        )
        logger.info("Regenerator ready")

    # ── Single claim correction ────────────────────────────────────────────────

    def correct(self, verify_result: dict) -> dict:
        """
        Attempt to correct a hallucinated claim.

        Args:
            verify_result : Output dict from verifier.verify_claim()

        Returns:
            Copy of verify_result with added keys:
                corrected_claim    : str
                correction_status  : "CORRECTED" | "UNCHANGED" | "UNVERIFIABLE" | "FAILED"
                correction_attempt : int
                correction_note    : str
        """
        claim = verify_result.get("claim", "")
        final_label = verify_result.get("final_label", "UNVERIFIABLE")
        evidence = verify_result.get("nli_evidence", "")

        # Already supported — nothing to do
        if final_label in SKIP_LABELS:
            logger.info("correct | SUPPORTED — skipping: %.50s", claim)
            return self._attach(
                verify_result, claim, "UNCHANGED", 0,
                "Claim is already supported — no correction needed.",
            )

        # No evidence — cannot correct regardless of label.
        # HALLUCINATED claims with empty evidence arise from the symbolic-logic
        # short-circuit path (no retrieval was attempted) or from retrieval
        # falling below threshold.  Proceeding to _generate_correction() with
        # blank evidence would waste up to 6 LLM calls and always fail NLI
        # validation (NLIScorer.score returns NEUTRAL for empty evidence).
        # We use "UNVERIFIABLE" status because the claim cannot be corrected
        # without a grounding passage — the label reflects epistemic state,
        # not that the claim is necessarily true.
        if not evidence:
            logger.info(
                "correct | no evidence available — skipping correction for %s claim: %.50s",
                final_label, claim,
            )
            return self._attach(
                verify_result, claim, "UNVERIFIABLE", 0,
                "No evidence available — correction skipped to avoid wasted LLM calls.",
            )

        # Ollama offline — flag but skip LLM rewriting
        if not is_ollama_running():
            logger.warning("correct | Ollama offline — skipping LLM rewriting")
            if evidence:
                note = (
                    "Claim detected as hallucinated by NLI cross-encoder. "
                    "LLM correction unavailable (Ollama not running). "
                    "Evidence: %.200s..." % evidence
                )
            else:
                note = "Claim detected as hallucinated. No correction applied (Ollama offline)."
            return self._attach(verify_result, claim, "FAILED", 0, note)

        # Attempt correction
        logger.info("correct | attempting correction: %.60s", claim)

        for attempt in range(1, MAX_ATTEMPTS + 1):
            logger.info("correct | attempt %d/%d", attempt, MAX_ATTEMPTS)

            corrected = self._generate_correction(claim, evidence)

            if not corrected:
                logger.warning("correct | attempt %d: empty correction returned", attempt)
                continue

            if corrected.lower() == claim.lower():
                logger.warning("correct | attempt %d: correction identical to original", attempt)
                continue

            # Validate the correction is actually supported by evidence
            validation = self.nli.score(corrected, evidence)
            val_label = validation["label"]
            val_conf = validation["confidence"]

            logger.info("correct | attempt %d validation: %s (%.0f%%)", attempt, val_label, val_conf * 100)

            if val_label == "SUPPORTED" and val_conf >= CORRECTION_CONFIDENCE_THRESHOLD:
                logger.info("correct | accepted on attempt %d", attempt)
                return self._attach(
                    verify_result, corrected, "CORRECTED", attempt,
                    "Corrected using evidence after %d attempt(s). Validation: %s (%.0f%%)."
                    % (attempt, val_label, val_conf * 100),
                )

            logger.warning(
                "correct | attempt %d not validated (%s %.0f%%) — retrying",
                attempt, val_label, val_conf * 100,
            )

        logger.warning("correct | all %d attempts failed: %.50s", MAX_ATTEMPTS, claim)
        return self._attach(
            verify_result, claim, "FAILED", MAX_ATTEMPTS,
            "Could not produce a validated correction after %d attempts." % MAX_ATTEMPTS,
        )

    # ── Batch correction ───────────────────────────────────────────────────────

    def correct_batch(self, verify_results: list[dict]) -> list[dict]:
        """
        Correct all hallucinated claims in a batch of verification results.

        Args:
            verify_results : List of dicts from verifier.verify_batch()

        Returns:
            Same list with correction fields added to each dict.
        """
        if not verify_results:
            return []

        hallucinated = sum(
            1 for r in verify_results if r.get("final_label") == "HALLUCINATED"
        )
        logger.info(
            "correct_batch | %d claims, %d hallucinated",
            len(verify_results), hallucinated,
        )

        results = []
        for i, result in enumerate(verify_results):
            label = result.get("final_label", "")
            logger.info(
                "correct_batch | [%d/%d] %s: %.50s",
                i + 1, len(verify_results), label, result.get("claim", ""),
            )
            results.append(self.correct(result))

        return results

    # ── Demo / CLI helper ──────────────────────────────────────────────────────

    def print_corrections(self, results: list[dict]) -> None:
        # Demo / CLI helper — uses print() intentionally for human-readable output.
        icons = {
            "CORRECTED": "🔧",
            "UNCHANGED": "✅",
            "UNVERIFIABLE": "⚠️ ",
            "FAILED": "❌",
        }

        corrected_count = sum(1 for r in results if r.get("correction_status") == "CORRECTED")
        failed_count = sum(1 for r in results if r.get("correction_status") == "FAILED")
        unchanged_count = sum(1 for r in results if r.get("correction_status") == "UNCHANGED")
        unverifiable_count = sum(1 for r in results if r.get("correction_status") == "UNVERIFIABLE")

        print("\n" + "=" * 60)
        print("SELF-CORRECTION REPORT")
        print("=" * 60)
        print("Corrected    : %d  🔧" % corrected_count)
        print("Unchanged    : %d  ✅" % unchanged_count)
        print("Unverifiable : %d  ⚠️" % unverifiable_count)
        print("Failed       : %d  ❌" % failed_count)
        print("=" * 60)

        for i, r in enumerate(results, 1):
            status = r.get("correction_status", "UNKNOWN")
            icon = icons.get(status, "❓")
            attempts = r.get("correction_attempt", 0)

            print("\n[%d] %s %s" % (i, icon, status))
            print("     Original  : %s" % r.get("claim", ""))

            corrected = r.get("corrected_claim", "")
            if corrected and corrected != r.get("claim", ""):
                print("     Corrected : %s" % corrected)

            print("     Note      : %s" % r.get("correction_note", ""))
            if attempts > 0:
                print("     Attempts  : %d" % attempts)

        print("\n" + "=" * 60)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _generate_correction(self, claim: str, evidence: str) -> str:
        """
        Generate a corrected version using Chain-of-Verification (CoVe).

        Makes 3 sequential LLM calls:
            1. Generate verification questions
            2. Answer questions using only the evidence
            3. Rewrite claim from verified answers
        """
        # Step 1: Generate verification questions
        plan_prompt = (
            'Based on this incorrect claim: "%s"\n'
            "What 2 specific questions must be answered to correct this claim?\n"
            "Provide only the questions." % claim
        )
        questions = call_llm(plan_prompt, temperature=0.0).strip()

        # Step 2: Answer questions using evidence only
        answer_prompt = (
            "Evidence: %s\nQuestions: %s\n"
            "Answer the questions using ONLY the evidence." % (evidence, questions)
        )
        answers = call_llm(answer_prompt, temperature=0.0).strip()

        # Step 3: Rewrite claim from verified answers
        cove_prompt = (
            "You are a fact-correction assistant.\n"
            'Incorrect Claim: "%s"\n'
            "Verified Answers: %s\n"
            "Evidence: %s\n\n"
            "Rewrite the incorrect claim into a single, accurate sentence "
            "using ONLY the facts above.\n"
            "Corrected claim:" % (claim, answers, evidence)
        )
        corrected = call_llm(cove_prompt, temperature=0.0).strip().strip('"').strip()

        if len(corrected) < 10 or len(corrected) > len(claim) * 3:
            return ""

        return corrected

    def _generate_caveat(self, claim: str) -> str:
        """Generate an honest caveat when a claim cannot be corrected."""
        response = call_llm(_CANNOT_CORRECT_PROMPT.format(claim=claim), temperature=0.0)
        if response and len(response.strip()) > 10:
            return response.strip()
        return "[Unverified] This claim could not be verified: '%s'" % claim

    def _attach(
        self,
        result: dict,
        corrected_claim: str,
        status: str,
        attempts: int,
        note: str,
    ) -> dict:
        """Return a copy of result with correction fields attached."""
        result = result.copy()
        result["corrected_claim"] = corrected_claim
        result["correction_status"] = status
        result["correction_attempt"] = attempts
        result["correction_note"] = note
        return result


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import logging as _logging
    _logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    from core.call_llm import is_ollama_running
    from core.knowledge_base import KnowledgeBase
    from core.verifier import Verifier

    if not is_ollama_running():
        print("❌ Ollama not running. Run: ollama serve")
        sys.exit(1)
    print("✅ Ollama is running\n")

    passages = [
        "Albert Einstein was born on March 14, 1879, in Ulm, Germany.",
        "Einstein received the Nobel Prize in Physics in 1921 for the photoelectric effect.",
        "Einstein emigrated to the United States in December 1932.",
        "Einstein died on April 18, 1955, at Princeton Hospital in New Jersey.",
        "The Eiffel Tower was completed in 1889 and stands 330 metres tall in Paris, France.",
        "Python was created by Guido van Rossum and first released in 1991.",
    ]

    kb = KnowledgeBase()
    kb.build(passages)
    verifier = Verifier(kb, use_selfcheck=True)

    test_claims = [
        "Einstein was born in France in 1885.",
        "Einstein won the Nobel Prize in Physics.",
        "Einstein moved to Canada after the war.",
        "The Eiffel Tower is located in London.",
    ]

    verify_results = verifier.verify_batch(test_claims, context="Test context.")
    regen = Regenerator()
    corrected = regen.correct_batch(verify_results)
    regen.print_corrections(corrected)
    sys.exit(0)
