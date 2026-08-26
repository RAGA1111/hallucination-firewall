"""
Rule-based hallucination type from verifier signals (no extra LLM call).
"""

from __future__ import annotations


def classify_hallucination_type(verify_result: dict) -> str:
    """
    Return a coarse type string for demos / downstream analytics.

    Types:
        symbolic_logic_error, contradicts_evidence, inconsistent_sampling,
        ungrounded, neutral_or_mixed, supported, unknown
    """
    expl = (verify_result.get("explanation") or "").lower()
    if "symbolic" in expl or "logic failure" in expl:
        return "symbolic_logic_error"

    nli = verify_result.get("nli_label", "")
    sc = verify_result.get("selfcheck_label", "")
    retr = float(verify_result.get("retrieval_score") or 0.0)
    final = verify_result.get("final_label", "")

    if final == "SUPPORTED":
        return "supported"

    if nli == "CONTRADICTED":
        return "contradicts_evidence"

    if sc == "INCONSISTENT":
        return "inconsistent_sampling"

    if retr <= 0.0 or "no relevant evidence" in expl or "no strong evidence" in expl:
        return "ungrounded"

    if final == "UNVERIFIABLE":
        return "neutral_or_mixed"

    if final == "HALLUCINATED":
        return "inconsistent_sampling"

    return "unknown"
