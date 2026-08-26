"""
NLI-based claim scorer using DeBERTa-v3-small.

The NLI model is loaded once on first NLIScorer instantiation and reused
for the lifetime of the process (singleton via NLIModelRegistry).

Public API:
    NLIScorer   — wraps the pipeline; call .score() or .score_batch()
"""

from __future__ import annotations

import logging

from transformers import pipeline

logger = logging.getLogger(__name__)

__all__ = ["NLIScorer"]

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_NAME: str = "cross-encoder/nli-deberta-v3-small"

# Default thresholds — the authoritative values live in PipelineConfig.
# These are fallbacks used when NLIScorer is instantiated without a config.
ENTAILMENT_THRESHOLD: float = 0.70
CONTRADICTION_THRESHOLD: float = 0.70

# NLI label strings returned by DeBERTa
LABEL_ENTAILMENT: str = "entailment"
LABEL_CONTRADICTION: str = "contradiction"
LABEL_NEUTRAL: str = "neutral"


# ── Model registry (process-wide singleton) ───────────────────────────────────

class _NLIModelRegistry:
    """
    Holds a single loaded HuggingFace pipeline for the NLI model.

    The model is loaded on the first call to get_pipeline() and reused
    on every subsequent call. Requesting a different model_name after the
    first load logs a warning and returns the already-loaded pipeline —
    this is intentional: loading a second model would double memory usage.
    """

    _pipeline = None
    _loaded_model_name: str | None = None

    @classmethod
    def get_pipeline(cls, model_name: str = MODEL_NAME):
        if cls._pipeline is None:
            logger.info("Loading NLI pipeline: %s", model_name)
            cls._pipeline = pipeline(
                "text-classification",
                model=model_name,
                top_k=None,
            )
            cls._loaded_model_name = model_name
            logger.info("NLI pipeline loaded")
        elif cls._loaded_model_name != model_name:
            logger.warning(
                "NLI pipeline already loaded as %r; ignoring request for %r",
                cls._loaded_model_name,
                model_name,
            )
        return cls._pipeline


# ── Scorer ─────────────────────────────────────────────────────────────────────

class NLIScorer:
    """
    Wraps the DeBERTa-v3-small NLI model for claim verification.

    Usage:
        scorer = NLIScorer()
        result = scorer.score(claim, evidence)
        results = scorer.score_batch([(claim1, ev1), (claim2, ev2)])
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        entailment_threshold: float | None = None,
        contradiction_threshold: float | None = None,
    ) -> None:
        self.model = _NLIModelRegistry.get_pipeline(model_name)
        self.entailment_threshold = (
            entailment_threshold
            if entailment_threshold is not None
            else ENTAILMENT_THRESHOLD
        )
        self.contradiction_threshold = (
            contradiction_threshold
            if contradiction_threshold is not None
            else CONTRADICTION_THRESHOLD
        )

    # ── Single scoring ─────────────────────────────────────────────────────────

    def score(self, claim: str, evidence: str) -> dict:
        """
        Score whether *evidence* supports, contradicts, or is neutral to *claim*.

        DeBERTa NLI input format:  "premise [SEP] hypothesis"
            premise    = evidence  (what we know)
            hypothesis = claim     (what we are checking)

        Returns:
            label       : "SUPPORTED" | "CONTRADICTED" | "NEUTRAL"
            raw_label   : "entailment" | "contradiction" | "neutral"
            confidence  : float 0.0–1.0
            all_scores  : dict of all three label probabilities
            claim       : original claim string
            evidence    : original evidence string
        """
        if not claim or not evidence:
            logger.warning("NLIScorer.score | empty claim or evidence")
            return self._empty_result(claim, evidence)

        input_text = f"{evidence} [SEP] {claim}"

        try:
            raw_results = self.model(input_text)[0]

            all_scores = {
                item["label"].lower(): round(item["score"], 4)
                for item in raw_results
            }

            entailment_score = all_scores.get(LABEL_ENTAILMENT, 0.0)
            contradiction_score = all_scores.get(LABEL_CONTRADICTION, 0.0)
            neutral_score = all_scores.get(LABEL_NEUTRAL, 0.0)

            if entailment_score >= self.entailment_threshold:
                label = "SUPPORTED"
                confidence = entailment_score
                raw_label = LABEL_ENTAILMENT
            elif contradiction_score >= self.contradiction_threshold:
                label = "CONTRADICTED"
                confidence = contradiction_score
                raw_label = LABEL_CONTRADICTION
            else:
                label = "NEUTRAL"
                confidence = neutral_score
                raw_label = LABEL_NEUTRAL

            return {
                "label": label,
                "raw_label": raw_label,
                "confidence": confidence,
                "all_scores": all_scores,
                "claim": claim,
                "evidence": evidence,
            }

        except Exception as exc:
            logger.error("NLIScorer.score | failed: %s", exc)
            return self._empty_result(claim, evidence)

    # ── Batch scoring ──────────────────────────────────────────────────────────

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """
        Score multiple (claim, evidence) pairs.

        Args:
            pairs : list of (claim, evidence) tuples

        Returns:
            list of result dicts in the same order as *pairs*
        """
        if not pairs:
            return []

        logger.info("NLIScorer.score_batch | %d pairs", len(pairs))
        return [self.score(claim, evidence) for claim, evidence in pairs]

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _empty_result(self, claim: str, evidence: str) -> dict:
        """Safe fallback result when scoring cannot proceed."""
        return {
            "label": "NEUTRAL",
            "raw_label": LABEL_NEUTRAL,
            "confidence": 0.0,
            "all_scores": {
                LABEL_ENTAILMENT: 0.0,
                LABEL_CONTRADICTION: 0.0,
                LABEL_NEUTRAL: 1.0,
            },
            "claim": claim,
            "evidence": evidence,
        }

    def interpret(self, result: dict) -> str:
        """Human-readable one-liner for a scoring result (debug / demo use)."""
        label = result["label"]
        conf = result["confidence"]
        messages = {
            "SUPPORTED": f"✅ SUPPORTED ({conf:.0%}) — Evidence backs the claim",
            "CONTRADICTED": f"❌ CONTRADICTED ({conf:.0%}) — Evidence conflicts with claim",
            "NEUTRAL": f"⚠️  NEUTRAL ({conf:.0%}) — Evidence neither supports nor contradicts",
        }
        return messages.get(label, f"❓ UNKNOWN label={label!r}")


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    print("\n=== Loading NLI scorer ===")
    scorer = NLIScorer()

    print("\n=== Test 1: Clear ENTAILMENT ===")
    claim = "Einstein won the Nobel Prize in Physics in 1921."
    evidence = "Albert Einstein received the Nobel Prize in Physics in 1921 for his discovery of the photoelectric effect."
    result = scorer.score(claim, evidence)
    print(scorer.interpret(result))
    print(f"Scores: {result['all_scores']}")

    print("\n=== Test 2: Clear CONTRADICTION ===")
    claim2 = "Einstein was born in France."
    evidence2 = "Albert Einstein was born on March 14, 1879, in Ulm, Germany."
    result2 = scorer.score(claim2, evidence2)
    print(scorer.interpret(result2))
    print(f"Scores: {result2['all_scores']}")

    print("\n=== Test 3: NEUTRAL — unrelated evidence ===")
    claim3 = "Einstein invented the telephone."
    evidence3 = "The Eiffel Tower stands 330 metres tall and is located in Paris, France."
    result3 = scorer.score(claim3, evidence3)
    print(scorer.interpret(result3))

    print("\n=== Test 4: Batch scoring ===")
    pairs = [
        ("Einstein died in 1955.", "Einstein died on April 18, 1955, at Princeton Hospital."),
        ("Einstein was born in 1900.", "Albert Einstein was born on March 14, 1879."),
        ("Python was created by Guido van Rossum.", "Python was created by Guido van Rossum and first released in 1991."),
    ]
    for r in scorer.score_batch(pairs):
        print(f"  {scorer.interpret(r)}")

    print("\n=== Test 5: Edge cases ===")
    print(f"Empty claim    → {scorer.score('', 'some evidence')['label']}")
    print(f"Empty evidence → {scorer.score('some claim', '')['label']}")

    print("\n=== All NLI scorer tests complete ✅ ===")
    sys.exit(0)
