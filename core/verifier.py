"""
Claim verifier — orchestrates KB retrieval, NLI scoring, SelfCheck sampling,
symbolic logic checking, and score fusion into a single verdict per claim.

Public API:
    Verifier   — instantiate once, call verify_claim() / verify_batch()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict

from core.call_llm import is_ollama_running
from core.config import DEFAULT_CONFIG, PipelineConfig
from core.hallucination_type import classify_hallucination_type
from core.knowledge_base import KnowledgeBase
from core.nli_scorer import NLIScorer
from core.selfcheck import (
    SAMPLE_MODEL,
    check_claim_consistency,
    check_claim_consistency_async,
    fuse_scores,
)
from core.symbolic import check_symbolic_logic

logger = logging.getLogger(__name__)

__all__ = ["Verifier"]

# ── Constants ──────────────────────────────────────────────────────────────────

_UNKNOWN_HALLUCINATION_TYPE: str = "unknown"


# ── LRU cache ──────────────────────────────────────────────────────────────────

class _LRUCache:
    """
    Simple LRU cache with optional TTL.

    Values are deep-copied via json.loads(json.dumps(...)) on both read and
    write.  This ensures callers cannot mutate a cached dict and corrupt future
    cache hits.  All stored values must be JSON-serialisable.
    """

    def __init__(self, maxsize: int, ttl_seconds: float = 0.0) -> None:
        self.maxsize = max(1, int(maxsize))
        self.ttl_seconds = float(ttl_seconds)
        self._store: OrderedDict[str, tuple[dict, float]] = OrderedDict()

    def get(self, key: str) -> dict | None:
        if key not in self._store:
            return None
        value, ts = self._store[key]
        if self.ttl_seconds > 0 and (time.time() - ts) > self.ttl_seconds:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return json.loads(json.dumps(value))

    def set(self, key: str, value: dict) -> None:
        self._store[key] = (json.loads(json.dumps(value)), time.time())
        self._store.move_to_end(key)
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)


# ── NLI aggregation helpers ────────────────────────────────────────────────────

def _pick_best_nli(
    scored: list[tuple[dict, dict]],
    nli_entailment_threshold: float,
    nli_contradiction_threshold: float,
) -> tuple[dict, dict]:
    """
    Choose the best (nli_result, retrieval_row) pair from a scored list.

    Waterfall by retrieval relevance, most relevant passage first:
        - If the most relevant passage gives a decisive verdict (entailment
          or contradiction above its threshold), use it immediately.
        - If it's NEUTRAL (inconclusive), fall through to the next-most
          relevant passage and repeat.
        - If nothing in the whole candidate list is decisive, fall back to
          the single most relevant passage's own (inconclusive) result.

    This deliberately does NOT compare raw NLI scores across all retrieved
    passages regardless of relevance (an earlier approach did, and later a
    fixed similarity-margin filter was tried as a guard). Both were fragile:
    a low-relevance, tangentially related passage can still produce a very
    high raw contradiction score (e.g. flagging a date mismatch against a
    passage about a different sub-topic entirely), and a similarity margin
    can be defeated by small wording changes to the claim that shift the
    top passage's score down and a tangential passage's score up just
    enough to both land inside the margin — which is exactly what happened
    when claim decomposition started producing shorter, more focused atomic
    claims (e.g. "Python programming language first released in 1991."
    scored lower similarity against the exact-match KB sentence than the
    original longer claim did, while a tangential "Python implementation
    began in December 1989." passage scored correspondingly higher,
    collapsing the gap between them).

    Evaluating strictly in relevance order and only advancing past a
    passage when it is genuinely inconclusive removes the need for any
    such threshold entirely: the most relevant passage always gets first
    say, and a less relevant passage can only ever contribute when nothing
    more relevant had an opinion.
    """
    if not scored:
        raise ValueError("_pick_best_nli called with an empty scored list")

    ordered = sorted(scored, key=lambda t: t[1]["score"], reverse=True)

    for nli, r in ordered:
        all_scores = nli.get("all_scores", {}) or {}
        ent = float(all_scores.get("entailment", 0.0))
        con = float(all_scores.get("contradiction", 0.0))

        if con >= nli_contradiction_threshold and con >= ent:
            return nli, r
        if ent >= nli_entailment_threshold:
            return nli, r
        # Inconclusive on this passage — fall through to the next-most
        # relevant candidate rather than letting it be overridden by raw
        # score comparisons against less relevant passages.

    # Nothing in the candidate list was decisive; fall back to the single
    # most relevant passage's own (inconclusive) result rather than an
    # arbitrary one.
    return ordered[0]


# ── Verifier ───────────────────────────────────────────────────────────────────

class Verifier:
    """
    Unified claim verifier — orchestrates KB retrieval, NLI, and SelfCheck.

    Instantiate once and reuse across many claims; the NLI model and KB are
    loaded at construction time.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        use_selfcheck: bool = True,
        config: PipelineConfig | None = None,
    ) -> None:
        logger.info("Initializing Verifier...")
        self.kb = kb
        self.use_selfcheck = use_selfcheck
        self.config = config or DEFAULT_CONFIG
        self.nli = NLIScorer(
            entailment_threshold=self.config.nli_entailment_threshold,
            contradiction_threshold=self.config.nli_contradiction_threshold,
        )
        self._claim_cache = _LRUCache(
            self.config.claim_cache_max_entries,
            self.config.claim_cache_ttl_seconds,
        )
        logger.info("Verifier ready")

    # ── Cache key ──────────────────────────────────────────────────────────────

    def _cache_key(self, claim: str, use_selfcheck: bool) -> str:
        payload = json.dumps(
            {
                "c": claim.strip().lower(),
                "sc": use_selfcheck,
                "fp": list(self.config.cache_fingerprint()),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def _retrieve(self, claim: str) -> list[dict]:
        return self.kb.retrieve(
            claim,
            top_k=self.config.top_k_retrieval,
            min_score=self.config.kb_retrieve_min_score,
            wiki_min_similarity=self.config.wiki_min_similarity,
        )

    # ── NLI aggregation ────────────────────────────────────────────────────────

    def _build_nli_meta(self, nli: dict, r: dict) -> dict:
        return {
            "retrieval_score": r["score"],
            "source": r.get("source", ""),
            "nli_label": nli.get("label"),
            "nli_confidence": nli.get("confidence"),
            "all_scores": nli.get("all_scores", {}),
            "passage_preview": (r.get("passage") or "")[:240],
        }

    def _aggregate_nli(
        self,
        claim: str,
        retrieved: list[dict],
    ) -> tuple[dict, str, float, list[dict]]:
        """Run NLI on each retrieved passage and return the best verdict."""
        candidates = [r for r in retrieved if r["score"] >= self.config.min_retrieval_score]

        if not candidates:
            return self._neutral_nli(claim), "", 0.0, []

        scored: list[tuple[dict, dict]] = []
        meta: list[dict] = []
        for r in candidates:
            nli = self.nli.score(claim, r["passage"])
            scored.append((nli, r))
            meta.append(self._build_nli_meta(nli, r))

        chosen_nli, chosen_ret = _pick_best_nli(
            scored,
            self.config.nli_entailment_threshold,
            self.config.nli_contradiction_threshold,
        )
        return chosen_nli, chosen_ret["passage"], float(chosen_ret["score"]), meta

    async def _aggregate_nli_async(
        self,
        claim: str,
        retrieved: list[dict],
    ) -> tuple[dict, str, float, list[dict]]:
        """Async version: scores all candidates concurrently then picks best."""
        candidates = [r for r in retrieved if r["score"] >= self.config.min_retrieval_score]

        if not candidates:
            return self._neutral_nli(claim), "", 0.0, []

        async def _score_one(r: dict) -> tuple[dict, dict]:
            nli = await asyncio.to_thread(self.nli.score, claim, r["passage"])
            return nli, r

        scored = list(await asyncio.gather(*[_score_one(r) for r in candidates]))
        meta = [self._build_nli_meta(nli, r) for nli, r in scored]

        chosen_nli, chosen_ret = _pick_best_nli(
            scored,
            self.config.nli_entailment_threshold,
            self.config.nli_contradiction_threshold,
        )
        return chosen_nli, chosen_ret["passage"], float(chosen_ret["score"]), meta

    @staticmethod
    def _neutral_nli(claim: str) -> dict:
        return {
            "label": "NEUTRAL",
            "confidence": 0.5,
            "all_scores": {},
            "claim": claim,
            "evidence": "",
        }

    # ── Result builders ────────────────────────────────────────────────────────

    def _base_result(
        self,
        claim: str,
        final_label: str,
        fused_score: float,
        nli_result: dict,
        selfcheck_result: dict,
        best_evidence: str,
        retrieval_score: float,
        explanation: str,
        nli_evidence_meta: list[dict],
        status_override: str | None = None,
        symbolic_result: dict | None = None,
        symbolic_conflict: bool = False,
    ) -> dict:
        status = status_override or (
            "VERIFIED" if final_label == "SUPPORTED"
            else "HALLUCINATED" if final_label == "HALLUCINATED"
            else "UNVERIFIABLE"
        )
        out = {
            "claim": claim,
            "final_label": final_label,
            "status": status,
            "fused_score": fused_score,
            "claim_confidence": fused_score,
            "nli_label": nli_result["label"],
            "nli_confidence": nli_result["confidence"],
            "nli_all_scores": nli_result.get("all_scores", {}),
            "nli_evidence": best_evidence,
            "nli_evidence_meta": nli_evidence_meta,
            "retrieval_score": retrieval_score,
            "selfcheck_label": selfcheck_result["label"],
            "selfcheck_score": selfcheck_result["consistency_score"],
            "selfcheck_votes": selfcheck_result.get("votes", []),
            "symbolic_has_logic": bool(symbolic_result and symbolic_result.get("has_logic")),
            "symbolic_passed": (symbolic_result or {}).get("passed", True),
            "symbolic_note": (symbolic_result or {}).get("note", ""),
            "symbolic_conflict": symbolic_conflict,
            "explanation": explanation,
        }
        out["hallucination_type"] = classify_hallucination_type(out)
        return out

    def _empty_result(self, claim: str) -> dict:
        return {
            "claim": claim,
            "final_label": "UNVERIFIABLE",
            "status": "ERROR",
            "fused_score": 0.5,
            "claim_confidence": 0.5,
            "nli_label": "NEUTRAL",
            "nli_confidence": 0.0,
            "nli_all_scores": {},
            "nli_evidence": "",
            "nli_evidence_meta": [],
            "retrieval_score": 0.0,
            "selfcheck_label": "UNCERTAIN",
            "selfcheck_score": 0.5,
            "selfcheck_votes": [],
            "symbolic_has_logic": False,
            "symbolic_passed": True,
            "symbolic_note": "",
            "symbolic_conflict": False,
            "explanation": "Empty claim passed to verifier.",
            "hallucination_type": _UNKNOWN_HALLUCINATION_TYPE,
        }

    def _timeout_result(self, claim: str) -> dict:
        return {
            "claim": claim,
            "final_label": "UNVERIFIABLE",
            "status": "TIMEOUT",
            "fused_score": 0.0,
            "claim_confidence": 0.0,
            "nli_label": "NEUTRAL",
            "nli_confidence": 0.0,
            "nli_all_scores": {},
            "nli_evidence": "",
            "nli_evidence_meta": [],
            "retrieval_score": 0.0,
            "selfcheck_label": "UNCERTAIN",
            "selfcheck_score": 0.0,
            "selfcheck_votes": [],
            "symbolic_has_logic": False,
            "symbolic_passed": True,
            "symbolic_note": "",
            "symbolic_conflict": False,
            "explanation": "Claim verification timed out before completion.",
            "hallucination_type": _UNKNOWN_HALLUCINATION_TYPE,
        }

    # ── Skipped selfcheck placeholder ──────────────────────────────────────────

    @staticmethod
    def _skipped_selfcheck(claim: str) -> dict:
        return {
            "label": "UNCERTAIN",
            "consistency_score": 0.5,
            "votes": [],
            "yes_count": 0,
            "no_count": 0,
            "valid_votes": 0,
            "claim": claim,
        }

    # ── Core verification (sync) ───────────────────────────────────────────────

    def verify_claim(
        self,
        claim: str,
        context: str = "",
        use_selfcheck: bool | None = None,
    ) -> dict:
        """
        Verify a single claim synchronously.

        Steps: retrieval → symbolic check → NLI → SelfCheck → fusion.

        Note: the symbolic-logic check runs early (it's cheap and doesn't
        depend on NLI/SelfCheck), but its result is no longer used to
        short-circuit the pipeline. A symbolic failure is passed into
        fuse_scores() as one signal among several — see fuse_scores'
        docstring for why an automatic override was unsafe (the small
        extraction model frequently misreads which numbers in a claim are
        actually related, producing false arithmetic failures on claims
        that are otherwise fully supported by evidence).
        """
        use_selfcheck = self.use_selfcheck if use_selfcheck is None else use_selfcheck

        if not claim or not claim.strip():
            logger.warning("verify_claim | empty claim")
            return self._empty_result(claim)

        ck = self._cache_key(claim, use_selfcheck)
        cached = self._claim_cache.get(ck)
        if cached is not None:
            logger.debug("verify_claim | cache hit")
            return cached

        logger.info("verify_claim | %.70s", claim)

        retrieved = self._retrieve(claim)

        # Symbolic check — informational only; see docstring note above.
        sym = check_symbolic_logic(claim)
        if sym["has_logic"] and not sym["passed"]:
            logger.info("verify_claim | symbolic check flagged (non-fatal): %s", sym["note"])

        # NLI
        nli_result, best_evidence, retrieval_score, nli_meta = self._aggregate_nli(
            claim, retrieved
        )
        logger.info("verify_claim | NLI %s (%.0f%%)", nli_result["label"], nli_result["confidence"] * 100)

        # SelfCheck
        if use_selfcheck:
            selfcheck_result = check_claim_consistency(
                claim,
                context=context,
                n_samples=self.config.selfcheck_samples,
                consistency_threshold=self.config.selfcheck_consistency_threshold,
                max_concurrent=self.config.selfcheck_concurrency,
            )
            logger.info(
                "verify_claim | SelfCheck %s (%.0f%%)",
                selfcheck_result["label"], selfcheck_result["consistency_score"] * 100,
            )
        else:
            selfcheck_result = self._skipped_selfcheck(claim)

        fused = fuse_scores(
            nli_result, selfcheck_result, symbolic_result=sym,
            fuse_high=self.config.fuse_high,
            fuse_low=self.config.fuse_low,
            contradiction_threshold=self.config.nli_contradiction_threshold,
        )
        explanation = self._build_explanation(
            fused["final_label"], nli_result, selfcheck_result,
            best_evidence, retrieval_score, multi=len(nli_meta) > 1,
            symbolic_conflict=fused.get("symbolic_conflict", False),
            symbolic_note=fused.get("symbolic_note"),
        )
        result = self._base_result(
            claim, fused["final_label"], fused["fused_score"],
            nli_result, selfcheck_result, best_evidence,
            retrieval_score, explanation, nli_meta,
            symbolic_result=sym,
            symbolic_conflict=fused.get("symbolic_conflict", False),
        )
        logger.info(
            "verify_claim | verdict=%s score=%.3f", fused["final_label"], fused["fused_score"]
        )
        self._claim_cache.set(ck, result)
        return result

    # ── Core verification (async) ──────────────────────────────────────────────

    async def verify_claim_async(
        self,
        claim: str,
        context: str = "",
        use_selfcheck: bool | None = None,
    ) -> dict:
        """Async version of verify_claim. See verify_claim's docstring for the
        note on why symbolic-logic failures are non-fatal."""
        use_selfcheck = self.use_selfcheck if use_selfcheck is None else use_selfcheck

        if not claim or not claim.strip():
            logger.warning("verify_claim_async | empty claim")
            return self._empty_result(claim)

        ck = self._cache_key(claim, use_selfcheck)
        cached = self._claim_cache.get(ck)
        if cached is not None:
            logger.debug("verify_claim_async | cache hit")
            return cached

        logger.info("verify_claim_async | %.70s", claim)

        retrieved = await asyncio.to_thread(self._retrieve, claim)

        # Symbolic check — informational only; see verify_claim's docstring.
        sym = await asyncio.to_thread(check_symbolic_logic, claim)
        if sym["has_logic"] and not sym["passed"]:
            logger.info("verify_claim_async | symbolic check flagged (non-fatal): %s", sym["note"])

        # NLI
        nli_result, best_evidence, retrieval_score, nli_meta = await self._aggregate_nli_async(
            claim, retrieved
        )

        # SelfCheck
        if use_selfcheck:
            selfcheck_result = await check_claim_consistency_async(
                claim,
                context=context,
                n_samples=self.config.selfcheck_samples,
                model=SAMPLE_MODEL,
                consistency_threshold=self.config.selfcheck_consistency_threshold,
                max_concurrent=self.config.selfcheck_concurrency,
            )
        else:
            selfcheck_result = self._skipped_selfcheck(claim)

        fused = fuse_scores(
            nli_result, selfcheck_result, symbolic_result=sym,
            fuse_high=self.config.fuse_high,
            fuse_low=self.config.fuse_low,
            contradiction_threshold=self.config.nli_contradiction_threshold,
        )
        explanation = self._build_explanation(
            fused["final_label"], nli_result, selfcheck_result,
            best_evidence, retrieval_score, multi=len(nli_meta) > 1,
            symbolic_conflict=fused.get("symbolic_conflict", False),
            symbolic_note=fused.get("symbolic_note"),
        )
        result = self._base_result(
            claim, fused["final_label"], fused["fused_score"],
            nli_result, selfcheck_result, best_evidence,
            retrieval_score, explanation, nli_meta,
            symbolic_result=sym,
            symbolic_conflict=fused.get("symbolic_conflict", False),
        )
        self._claim_cache.set(ck, result)
        return result

    # ── Batch verification ─────────────────────────────────────────────────────

    def verify_batch(
        self,
        claims: list[str],
        context: str = "",
        use_selfcheck: bool | None = None,
    ) -> list[dict]:
        if not claims:
            return []
        logger.info("verify_batch | %d claims", len(claims))
        results = []
        for i, claim in enumerate(claims):
            logger.info("verify_batch | [%d/%d]", i + 1, len(claims))
            results.append(self.verify_claim(claim, context=context, use_selfcheck=use_selfcheck))
        return results

    async def verify_batch_async(
        self,
        claims: list[str],
        context: str = "",
        use_selfcheck: bool | None = None,
    ) -> list[dict]:
        if not claims:
            return []
        logger.info("verify_batch_async | %d claims", len(claims))

        timeout = float(self.config.verify_claim_timeout_seconds)

        async def _one(claim: str) -> dict:
            try:
                return await asyncio.wait_for(
                    self.verify_claim_async(claim, context, use_selfcheck),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("verify_batch_async | timeout: %.64s", claim)
                return self._timeout_result(claim)
            except Exception as exc:
                logger.error("verify_batch_async | error for %.64s: %s", claim, exc)
                return self._empty_result(claim)

        return list(await asyncio.gather(*[_one(c) for c in claims]))

    # ── Summary / reporting (demo / CLI helpers) ───────────────────────────────

    def summarize(self, results: list[dict]) -> dict:
        supported = [r for r in results if r["final_label"] == "SUPPORTED"]
        hallucinated = [r for r in results if r["final_label"] == "HALLUCINATED"]
        unverifiable = [r for r in results if r["final_label"] == "UNVERIFIABLE"]
        return {
            "total": len(results),
            "supported": len(supported),
            "hallucinated": len(hallucinated),
            "unverifiable": len(unverifiable),
            "hallucination_rate": round(len(hallucinated) / len(results), 3) if results else 0,
            "supported_claims": [r["claim"] for r in supported],
            "hallucinated_claims": [r["claim"] for r in hallucinated],
            "unverifiable_claims": [r["claim"] for r in unverifiable],
        }

    def print_report(self, results: list[dict]) -> None:
        # Demo / CLI helper — uses print() intentionally for human-readable output.
        summary = self.summarize(results)
        print("\n" + "=" * 60)
        print("HALLUCINATION DETECTION REPORT")
        print("=" * 60)
        print("Total claims   : %d" % summary["total"])
        print("Supported      : %d  ✅" % summary["supported"])
        print("Hallucinated   : %d  ❌" % summary["hallucinated"])
        print("Unverifiable   : %d  ⚠️" % summary["unverifiable"])
        print("Hallucination rate: %.0f%%" % (summary["hallucination_rate"] * 100))
        print("=" * 60)
        icons = {"SUPPORTED": "✅", "HALLUCINATED": "❌", "UNVERIFIABLE": "⚠️ "}
        for i, r in enumerate(results, 1):
            icon = icons.get(r["final_label"], "❓")
            print("\n[%d] %s %s" % (i, icon, r["final_label"]))
            print("     Claim      : %s" % r["claim"])
            print("     NLI        : %s (%.0f%%)" % (r["nli_label"], r["nli_confidence"] * 100))
            print("     SelfCheck  : %s (%.0f%%)" % (r["selfcheck_label"], r["selfcheck_score"] * 100))
            print("     Fused score: %s" % r["fused_score"])
            if r.get("symbolic_conflict"):
                print("     Symbolic   : ⚠ conflict — %s" % r.get("symbolic_note", ""))
            if r["nli_evidence"]:
                print("     Evidence   : %s..." % r["nli_evidence"][:80])
            print("     Explanation: %s" % r["explanation"])
        print("\n" + "=" * 60)

    # ── Explanation builder ────────────────────────────────────────────────────

    def _build_explanation(
        self,
        final_label: str,
        nli_result: dict,
        selfcheck_result: dict,
        evidence: str,
        retrieval_score: float,
        multi: bool = False,
        symbolic_conflict: bool = False,
        symbolic_note: str | None = None,
    ) -> str:
        nli = nli_result["label"]
        sc = selfcheck_result["label"]
        sc_score = selfcheck_result["consistency_score"]
        yes = selfcheck_result.get("yes_count", 0)
        no = selfcheck_result.get("no_count", 0)
        sampled = selfcheck_result.get("valid_votes", 0) > 0

        multi_note = "Compared top retrieved passages with NLI. " if multi else ""
        symbolic_note_str = (
            "Note: symbolic logic check flagged a possible arithmetic issue (%s), "
            "but strong evidence support was judged more reliable. " % symbolic_note
            if symbolic_conflict else ""
        )

        if final_label == "SUPPORTED":
            sample_note = (
                "Model sampled %d times: %d yes / %d no." % (yes + no, yes, no)
                if sampled else "SelfCheck not sampled."
            )
            return (
                "%s%sEvidence (relevance=%.2f) supports this claim via NLI (%s). %s"
                % (multi_note, symbolic_note_str, retrieval_score, nli, sample_note)
            )

        if final_label == "HALLUCINATED":
            if nli == "CONTRADICTED":
                sample_note = (
                    "SelfCheck votes: %d yes / %d no." % (yes, no)
                    if sampled else "SelfCheck not sampled."
                )
                return (
                    "%sEvidence directly contradicts this claim. "
                    "Retrieved: '%.60s...' %s"
                    % (multi_note, evidence, sample_note)
                )
            if symbolic_conflict:
                return (
                    "%sSymbolic logic check failed (%s) and no strong evidence "
                    "was found to support the claim instead. NLI: %s."
                    % (multi_note, symbolic_note, nli)
                )
            sample_note = (
                "Model inconsistent across %d samples (%d yes / %d no = %.0f%% consistency)."
                % (yes + no, yes, no, sc_score * 100)
                if sampled else "SelfCheck not sampled."
            )
            return "%s%s NLI: %s. Likely hallucinated." % (multi_note, sample_note, nli)

        if not evidence:
            return (
                "%sNo relevant evidence found in knowledge base "
                "(best retrieval score below threshold). "
                "Cannot verify — treat with caution."
                % multi_note
            )
        return (
            "%sEvidence found but neither strongly supports nor contradicts. "
            "NLI: %s, SelfCheck: %s (%.0f%%). Insufficient signal to verify."
            % (multi_note, nli, sc, sc_score * 100)
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

    kb = KnowledgeBase()
    passages = [
        "Albert Einstein was born on March 14, 1879, in Ulm, Germany.",
        "Einstein received the Nobel Prize in Physics in 1921 for the photoelectric effect.",
        "Einstein emigrated to the United States in December 1932.",
        "Einstein died on April 18, 1955, at Princeton Hospital.",
        "Python was created by Guido van Rossum and first released in 1991.",
        "The Eiffel Tower was completed in 1889 and stands 330 metres tall in Paris.",
    ]
    kb.build(passages)

    verifier = Verifier(kb, use_selfcheck=True)
    test_claims = [
        "Einstein was born in Germany in 1879.",
        "Einstein won the Nobel Prize in Chemistry.",
        "Einstein moved to the United States in 1932.",
        "Einstein's favorite color was blue.",
        "The Eiffel Tower is located in London.",
    ]
    context = "A hallucinated LLM response about Einstein."
    results = verifier.verify_batch(test_claims, context=context)
    verifier.print_report(results)
    sys.exit(0)