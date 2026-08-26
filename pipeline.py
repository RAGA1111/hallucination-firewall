"""
Hallucination Firewall — master orchestrator.

Runs the full pipeline for a single LLM response:
    Decompose → (Wikipedia augment) → Verify → Correct → Finalize

Public API:
    HallucinationPipeline

Status constants (exported for API / callers):
    PIPELINE_STATUS_OK
    PIPELINE_STATUS_DECOMPOSITION_FAILED
    PIPELINE_STATUS_VERIFICATION_FAILED
    PIPELINE_STATUS_TIMEOUT
    PIPELINE_STATUS_INVALID_INPUT
    PIPELINE_STATUS_UNVERIFIED
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import html
import json
import logging
import time
from collections import OrderedDict

from core.call_llm import call_llm, call_llm_async, is_ollama_running
from core.chunking import chunk_passages_to_sentences, split_into_sentences
from core.config import DEFAULT_CONFIG, PipelineConfig
from core.decomposer import ClaimDecompositionError, decompose_claims, decompose_claims_async
from core.knowledge_base import KnowledgeBase, load_passages_from_file
from core.regenerator import Regenerator
from core.run_logger import append_pipeline_run
from core.tracking import append_tracking_row
from core.verifier import Verifier
from core.wiki_ingest import (
    fetch_passages_for_claim,
    fetch_wikipedia_passages,
    queries_from_question_and_claims,
)

logger = logging.getLogger(__name__)

__all__ = [
    "HallucinationPipeline",
    "PIPELINE_STATUS_OK",
    "PIPELINE_STATUS_DECOMPOSITION_FAILED",
    "PIPELINE_STATUS_VERIFICATION_FAILED",
    "PIPELINE_STATUS_TIMEOUT",
    "PIPELINE_STATUS_INVALID_INPUT",
    "PIPELINE_STATUS_UNVERIFIED",
]

# ── Status constants ───────────────────────────────────────────────────────────

PIPELINE_STATUS_OK = "OK"
PIPELINE_STATUS_DECOMPOSITION_FAILED = "DECOMPOSITION_FAILED"
PIPELINE_STATUS_VERIFICATION_FAILED = "VERIFICATION_FAILED"
PIPELINE_STATUS_TIMEOUT = "TIMEOUT"
PIPELINE_STATUS_INVALID_INPUT = "INVALID_INPUT"
PIPELINE_STATUS_UNVERIFIED = "UNVERIFIED"

# ── Prompt ─────────────────────────────────────────────────────────────────────

_ASK_PROMPT = """Answer the following question in 3-5 sentences with specific facts.
Include dates, names, and numbers where relevant.

Question: {question}

Answer:"""


# ── Response-level LRU cache ───────────────────────────────────────────────────

class _LRUCache:
    """
    Process-local LRU cache for full pipeline responses.

    Values are deep-copied via json.loads(json.dumps(...)) on read and write
    to prevent callers from mutating cached dicts.
    All stored values must be JSON-serialisable.
    """

    def __init__(self, maxsize: int) -> None:
        self.maxsize = max(1, int(maxsize))
        self._store: OrderedDict[str, dict] = OrderedDict()

    def get(self, key: str) -> dict | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return json.loads(json.dumps(self._store[key]))

    def set(self, key: str, value: dict) -> None:
        self._store[key] = json.loads(json.dumps(value))
        self._store.move_to_end(key)
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)


# ── Pipeline ───────────────────────────────────────────────────────────────────

class HallucinationPipeline:
    """
    Master orchestrator — runs the full hallucination detection pipeline.

    Usage:
        pipeline = HallucinationPipeline()
        pipeline.load_kb(passages)
        result = pipeline.run(llm_response)
    """

    def __init__(
        self,
        use_selfcheck: bool = True,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = PipelineConfig.from_env(config)
        self.kb = KnowledgeBase()
        self.verifier: Verifier | None = None
        self.regenerator = Regenerator()
        self.use_selfcheck = use_selfcheck
        self.kb_loaded = False
        self._seed_passages: list[str] = []
        self._response_cache = _LRUCache(
            maxsize=max(1, int(self.config.pipeline_response_cache_max))
        )
        logger.info("HallucinationPipeline initialized")

    # ── KB management ──────────────────────────────────────────────────────────

    def _rebuild_kb_from_seed(self) -> None:
        passages = [p for p in self._seed_passages if str(p).strip()]
        if not passages:
            raise ValueError("Cannot rebuild KB: seed passage list is empty after filtering.")
        if self.config.use_sentence_chunks:
            to_index = chunk_passages_to_sentences(passages)
        else:
            to_index = passages
        logger.info("Indexing KB: %d chunks from %d seed passages", len(to_index), len(passages))
        self.kb.build(to_index)
        self.verifier = Verifier(
            self.kb,
            use_selfcheck=self.use_selfcheck,
            config=self.config,
        )

    def load_kb(self, passages: list[str]) -> None:
        """Build the KB from a list of passage strings. Must be called before run()."""
        logger.info("Loading KB with %d passages", len(passages))
        self._seed_passages = [str(p).strip() for p in passages if str(p).strip()]
        self._rebuild_kb_from_seed()
        self.kb_loaded = True
        logger.info("KB loaded: %d passages", len(self._seed_passages))

    def load_kb_from_file(self, filepath: str) -> None:
        """Load KB from a plain text file (paragraphs separated by blank lines)."""
        passages = load_passages_from_file(filepath)
        self.load_kb(passages)

    def load_kb_from_disk(self) -> bool:
        """Load a previously saved KB from disk. Returns True if successful."""
        loaded = self.kb.load()
        if loaded:
            self._seed_passages = list(self.kb.passages)
            self.verifier = Verifier(
                self.kb,
                use_selfcheck=self.use_selfcheck,
                config=self.config,
            )
            self.kb_loaded = True
        return loaded

    def load_kb_auto(self, fallback_passages: list[str] | None = None) -> None:
        """
        Load KB from disk if available; otherwise build from fallback_passages.
        Raises RuntimeError if neither source is available.
        """
        if self.load_kb_from_disk():
            logger.info("Loaded KB from disk: %d passages", len(self.kb.passages))
            return
        if fallback_passages:
            logger.info("Disk KB not found — building from %d fallback passages", len(fallback_passages))
            self.load_kb(fallback_passages)
        else:
            raise RuntimeError("No saved KB found on disk and no fallback passages provided.")

    # ── Wikipedia augmentation ─────────────────────────────────────────────────

    def _augment_kb_with_wikipedia(self, question: str, claims: list[str]) -> None:
        if not self.config.dynamic_wikipedia:
            return
        queries = queries_from_question_and_claims(
            question, claims, max_queries=self.config.wiki_max_queries,
        )
        if not queries:
            return
        try:
            extra = fetch_wikipedia_passages(
                queries, max_pages=self.config.wiki_max_pages_per_run,
            )
        except Exception as exc:
            logger.warning("Dynamic Wikipedia ingest failed: %s", exc)
            return
        if not extra:
            return
        seen: set[str] = {s.lower() for s in self._seed_passages if s}
        merged: list[str] = list(self._seed_passages)
        for p in extra:
            k = str(p).strip().lower()
            if k and k not in seen:
                seen.add(k)
                merged.append(str(p).strip())
        self._seed_passages = merged
        self._rebuild_kb_from_seed()
        logger.info("Wikipedia KB merged — total seed passages: %d", len(self._seed_passages))

    def _maybe_active_kb_expand(
        self,
        claims: list[str],
        verify_results: list[dict],
        context: str,
        use_selfcheck: bool | None,
    ) -> list[dict]:
        if not self.config.active_kb_expansion or self.verifier is None:
            return verify_results
        additions: list[str] = []
        for r in verify_results:
            if r.get("final_label") != "UNVERIFIABLE":
                continue
            if float(r.get("retrieval_score") or 0.0) > self.config.active_kb_min_retrieval:
                continue
            c = r.get("claim") or ""
            if not c.strip():
                continue
            try:
                additions.extend(
                    fetch_passages_for_claim(c, max_pages=self.config.active_kb_max_pages)
                )
            except Exception as exc:
                logger.warning("Active KB fetch failed: %s", exc)
        if not additions:
            return verify_results
        seen = {p.lower() for p in self._seed_passages}
        for p in additions:
            k = p.strip().lower()
            if k and k not in seen:
                seen.add(k)
                self._seed_passages.append(p.strip())
        self._rebuild_kb_from_seed()
        return self.verifier.verify_batch(
            claims, context=context, use_selfcheck=use_selfcheck,
        )

    # ── Cache key ──────────────────────────────────────────────────────────────

    def _cache_key(self, llm_response: str, question: str, use_selfcheck: bool | None) -> str:
        sc = self.use_selfcheck if use_selfcheck is None else use_selfcheck
        payload = json.dumps(
            {
                "q": question.strip(),
                "r": llm_response.strip(),
                "sc": sc,
                "fp": list(self.config.cache_fingerprint()),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ── Main run (sync) ────────────────────────────────────────────────────────

    def run(
        self,
        llm_response: str,
        question: str = "",
        use_selfcheck: bool | None = None,
    ) -> dict:
        """
        Run the full hallucination detection and correction pipeline.

        Args:
            llm_response : Raw LLM output to verify
            question     : Original question that produced the response (optional)
            use_selfcheck: Override instance-level use_selfcheck setting

        Returns:
            Full pipeline result as a dict
        """
        if not self.kb_loaded:
            raise RuntimeError("KB not loaded. Call load_kb() or load_kb_from_disk() first.")
        if not llm_response or not llm_response.strip():
            return self._empty_result(llm_response)

        overall_start = time.time()
        timings: dict = {}

        logger.info("Pipeline START | input_chars=%d | %.80s", len(llm_response), llm_response)

        # Stage 1: Decompose
        logger.info("[1/4] Decomposing claims")
        t0 = time.time()
        try:
            decomposition = decompose_claims(llm_response)
        except ClaimDecompositionError as exc:
            timings["decompose_seconds"] = round(time.time() - t0, 2)
            logger.error("Decomposition failed: %s", exc)
            return self._empty_result(llm_response, reason="Decomposition failed") | {
                "status": PIPELINE_STATUS_DECOMPOSITION_FAILED,
                "error": str(exc),
            }
        timings["decompose_seconds"] = round(time.time() - t0, 2)
        claims = decomposition.claims
        logger.info("[1/4] %d claims in %.2fs", len(claims), timings["decompose_seconds"])

        if not claims:
            logger.warning("No claims extracted — returning unverified response")
            return self._empty_result(llm_response, reason="No claims extracted") | {
                "status": PIPELINE_STATUS_UNVERIFIED,
            }

        if self.verifier is None:
            raise RuntimeError("Verifier not initialized after KB load.")

        cache_key = self._cache_key(llm_response, question, use_selfcheck)
        cached = self._response_cache.get(cache_key)
        if cached is not None:
            logger.info("Pipeline cache hit")
            return cached

        self._augment_kb_with_wikipedia(question, claims)

        # Stage 2: Verify
        logger.info("[2/4] Verifying %d claims", len(claims))
        t0 = time.time()
        verify_results = self.verifier.verify_batch(
            claims, context=llm_response, use_selfcheck=use_selfcheck,
        )
        verify_results = self._maybe_active_kb_expand(
            claims, verify_results, llm_response, use_selfcheck,
        )
        timings["verify_seconds"] = round(time.time() - t0, 2)
        logger.info("[2/4] Verification done in %.2fs", timings["verify_seconds"])

        # Stage 3: Correct
        logger.info("[3/4] Correcting hallucinations")
        t0 = time.time()
        corrected_results = self.regenerator.correct_batch(verify_results)
        timings["correction_seconds"] = round(time.time() - t0, 2)
        logger.info("[3/4] Correction done in %.2fs", timings["correction_seconds"])

        # Stage 4: Finalize
        logger.info("[4/4] Building final response")
        timings["total_seconds"] = round(time.time() - overall_start, 2)
        out = self._finalize_payload(
            llm_response, question, corrected_results, timings, PIPELINE_STATUS_OK,
        )
        out["timing"]["total_seconds"] = round(time.time() - overall_start, 2)
        self._response_cache.set(cache_key, out)
        self._log_pipeline_summary(out)
        return out

    # ── Main run (async) ───────────────────────────────────────────────────────

    async def run_async(
        self,
        llm_response: str,
        question: str = "",
        use_selfcheck: bool | None = None,
    ) -> dict:
        """Async version of run() for FastAPI and event-loop contexts."""
        if not self.kb_loaded:
            raise RuntimeError("KB not loaded. Call load_kb() or load_kb_from_disk() first.")
        if not llm_response or not llm_response.strip():
            return self._empty_result(llm_response)

        overall_start = time.time()
        timings: dict = {}

        logger.info("Async Pipeline START | input_chars=%d | %.80s", len(llm_response), llm_response)

        # Stage 1: Decompose
        logger.info("[1/4] Decomposing claims (async)")
        t0 = time.time()
        try:
            decomposition = await decompose_claims_async(llm_response)
        except ClaimDecompositionError as exc:
            timings["decompose_seconds"] = round(time.time() - t0, 2)
            logger.error("Async decomposition failed: %s", exc)
            return self._empty_result(llm_response, reason="Decomposition failed") | {
                "status": PIPELINE_STATUS_DECOMPOSITION_FAILED,
                "error": str(exc),
            }
        timings["decompose_seconds"] = round(time.time() - t0, 2)
        claims = decomposition.claims
        logger.info("[1/4] %d claims in %.2fs", len(claims), timings["decompose_seconds"])

        if not claims:
            logger.warning("No claims extracted — returning unverified response (async)")
            return self._empty_result(llm_response, reason="No claims extracted") | {
                "status": PIPELINE_STATUS_UNVERIFIED,
            }

        if self.verifier is None:
            raise RuntimeError("Verifier not initialized after KB load.")

        cache_key = self._cache_key(llm_response, question, use_selfcheck)
        cached = self._response_cache.get(cache_key)
        if cached is not None:
            logger.info("Async pipeline cache hit")
            return cached

        await asyncio.to_thread(self._augment_kb_with_wikipedia, question, claims)

        # Stage 2: Verify
        logger.info("[2/4] Verifying %d claims (async)", len(claims))
        t0 = time.time()
        verify_results = await self.verifier.verify_batch_async(
            claims, context=llm_response, use_selfcheck=use_selfcheck,
        )
        verify_results = self._maybe_active_kb_expand(
            claims, verify_results, llm_response, use_selfcheck,
        )
        timings["verify_seconds"] = round(time.time() - t0, 2)
        logger.info("[2/4] Verification done in %.2fs (async)", timings["verify_seconds"])

        # Stage 3: Correct
        logger.info("[3/4] Correcting hallucinations (async)")
        t0 = time.time()
        corrected_results = await asyncio.to_thread(
            self.regenerator.correct_batch, verify_results
        )
        timings["correction_seconds"] = round(time.time() - t0, 2)
        logger.info("[3/4] Correction done in %.2fs (async)", timings["correction_seconds"])

        # Stage 4: Finalize
        logger.info("[4/4] Building final response (async)")
        timings["total_seconds"] = round(time.time() - overall_start, 2)
        out = self._finalize_payload(
            llm_response, question, corrected_results, timings, PIPELINE_STATUS_OK,
        )
        out["timing"]["total_seconds"] = round(time.time() - overall_start, 2)
        self._response_cache.set(cache_key, out)
        self._log_pipeline_summary(out)
        return out

    # ── Answer mode ────────────────────────────────────────────────────────────

    def ask(self, question: str, use_selfcheck: bool | None = None) -> dict:
        """
        Generate an LLM response to *question* then run it through the pipeline.
        Returns verified + corrected result.
        """
        logger.info("ask | generating response for: %.80s", question)
        llm_response = call_llm(_ASK_PROMPT.format(question=question), temperature=0.3)
        if not llm_response:
            return self._empty_result("", reason="LLM failed to generate response")
        logger.info("ask | generated %d chars", len(llm_response))
        return self.run(llm_response, question=question, use_selfcheck=use_selfcheck)

    async def ask_async(self, question: str, use_selfcheck: bool | None = None) -> dict:
        """Async version of ask()."""
        logger.info("ask_async | generating response for: %.80s", question)
        llm_response = await call_llm_async(_ASK_PROMPT.format(question=question), temperature=0.3)
        if not llm_response:
            return self._empty_result("", reason="LLM failed to generate response")
        logger.info("ask_async | generated %d chars", len(llm_response))
        return await self.run_async(llm_response, question=question, use_selfcheck=use_selfcheck)

    # ── Finalization ───────────────────────────────────────────────────────────

    def _finalize_payload(
        self,
        llm_response: str,
        question: str,
        corrected_results: list[dict],
        timings: dict,
        status: str,
        error: str | None = None,
    ) -> dict:
        t0 = time.time()
        summary = self._summarize(corrected_results)
        final_response = self._build_final_response(corrected_results)
        annotated, ann_html = self._annotate_sentences(llm_response, corrected_results)
        timings["rebuild_seconds"] = round(time.time() - t0, 2)

        payload = {
            "original_response": llm_response,
            "question": question,
            "claims": corrected_results,
            "summary": summary,
            "timing": timings,
            "final_response": final_response,
            "annotated_sentences": annotated,
            "final_response_html": ann_html,
            "status": status,
        }
        if error is not None:
            payload["error"] = error

        # Structured run logging — failures here must not crash the pipeline
        log_entry = {
            "question": question,
            "status": status,
            "summary": summary,
            "timing": timings,
            "claims": [
                {
                    "claim": r.get("claim"),
                    "final_label": r.get("final_label"),
                    "fused_score": r.get("fused_score"),
                    "hallucination_type": r.get("hallucination_type"),
                    "retrieval_score": r.get("retrieval_score"),
                    "nli_evidence_meta": r.get("nli_evidence_meta", [])[:5],
                }
                for r in corrected_results
            ],
        }
        try:
            append_pipeline_run(self.config.structured_log_dir, log_entry)
        except Exception as exc:
            logger.warning("append_pipeline_run failed (non-fatal): %s", exc)

        tracking_entry = {
            "dataset_tag": "pipeline_run",
            "status": status,
            "hallucination_rate": summary.get("hallucination_rate"),
            "total_claims": summary.get("total_claims"),
            "supported": summary.get("supported"),
            "hallucinated": summary.get("hallucinated"),
            "unverifiable": summary.get("unverifiable"),
            "total_seconds": timings.get("total_seconds"),
        }
        try:
            append_tracking_row(self.config.tracking_log_path, tracking_entry)
        except Exception as exc:
            logger.warning("append_tracking_row failed (non-fatal): %s", exc)

        return payload

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _label_rank(label: str) -> int:
        return {"HALLUCINATED": 3, "UNVERIFIABLE": 2, "SUPPORTED": 1}.get(label or "", 0)

    def _annotate_sentences(
        self, original: str, rows: list[dict]
    ) -> tuple[list[dict], str]:
        """
        Map each sentence in *original* to the worst-case verdict from overlapping claims.

        Heuristic: a claim is considered to overlap a sentence if the claim string
        is a substring of the sentence (case-insensitive). For short claims this can
        produce false matches — this is existing behaviour and is intentionally preserved.
        """
        sentences = split_into_sentences(original)
        annotated: list[dict] = []

        for s in sentences:
            s_low = s.lower()
            # Primary: exact substring match
            matched = [
                r for r in rows
                if r.get("claim") and str(r["claim"]).strip().lower() in s_low
            ]
            if not matched:
                # Fallback: token overlap (first 8 meaningful tokens)
                toks = [t for t in s_low.split() if len(t) > 4]
                matched = [
                    r for r in rows
                    if any(t in str(r.get("claim", "")).lower() for t in toks[:8])
                ]

            label = "UNVERIFIABLE"
            conf = 0.5
            if matched:
                worst = max(matched, key=lambda r: self._label_rank(str(r.get("final_label", ""))))
                label = str(worst.get("final_label", "UNVERIFIABLE"))
                conf = float(worst.get("claim_confidence", worst.get("fused_score", 0.5)))

            css = label.lower().replace(" ", "_")
            annotated.append({
                "sentence": s,
                "final_label": label,
                "claim_confidence": conf,
                "css_class": css,
            })

        colors = {
            "supported": "#0a6d3e",
            "hallucinated": "#b00020",
            "unverifiable": "#8a6d00",
        }
        parts = [
            '<p class="hf-{css}" style="color:{color}">{text}</p>'.format(
                css=a["css_class"],
                color=colors.get(a["css_class"], "#333333"),
                text=html.escape(a["sentence"]),
            )
            for a in annotated
        ]
        return annotated, "\n".join(parts)

    def _build_final_response(self, corrected_results: list[dict]) -> str:
        """Reconstruct a clean response using corrected claims where available."""
        sentences = []
        for r in corrected_results:
            status = r.get("correction_status", "")
            corrected = r.get("corrected_claim", "")
            original = r.get("claim", "")
            if status == "CORRECTED" and corrected:
                sentences.append(corrected)
            elif status == "UNCHANGED":
                sentences.append(original)
            elif status == "UNVERIFIABLE":
                sentences.append(corrected if corrected and corrected != original
                                  else "[Unverified] %s" % original)
            else:
                sentences.append(original)
        return " ".join(sentences)

    def _summarize(self, corrected_results: list[dict]) -> dict:
        total = len(corrected_results)
        supported = sum(1 for r in corrected_results if r.get("final_label") == "SUPPORTED")
        hallucinated = sum(1 for r in corrected_results if r.get("final_label") == "HALLUCINATED")
        unverifiable = sum(1 for r in corrected_results if r.get("final_label") == "UNVERIFIABLE")
        corrected = sum(1 for r in corrected_results if r.get("correction_status") == "CORRECTED")
        failed = sum(1 for r in corrected_results if r.get("correction_status") == "FAILED")
        return {
            "total_claims": total,
            "supported": supported,
            "hallucinated": hallucinated,
            "unverifiable": unverifiable,
            "corrected": corrected,
            "correction_failed": failed,
            "hallucination_rate": round(hallucinated / total, 3) if total > 0 else 0.0,
            "correction_rate": round(corrected / hallucinated, 3) if hallucinated > 0 else 1.0,
        }

    def _log_pipeline_summary(self, out: dict) -> None:
        s = out["summary"]
        t = out["timing"]
        logger.info(
            "Pipeline DONE | %.2fs | claims=%d supported=%d hallucinated=%d "
            "corrected=%d rate=%.0f%%",
            t.get("total_seconds", 0),
            s["total_claims"], s["supported"], s["hallucinated"],
            s["corrected"], s["hallucination_rate"] * 100,
        )

    def _empty_result(self, response: str, reason: str = "Empty input") -> dict:
        return {
            "original_response": response,
            "question": "",
            "claims": [],
            "summary": {
                "total_claims": 0,
                "supported": 0,
                "hallucinated": 0,
                "unverifiable": 0,
                "corrected": 0,
                "correction_failed": 0,
                "hallucination_rate": 0.0,
                "correction_rate": 1.0,
            },
            "timing": {"total_seconds": 0},
            "final_response": response,
            "annotated_sentences": [],
            "final_response_html": "",
            "status": PIPELINE_STATUS_INVALID_INPUT,
            "error": reason,
        }

    # ── Demo / CLI helper ──────────────────────────────────────────────────────

    def print_result(self, result: dict) -> None:
        # Demo / CLI helper — uses print() intentionally for human-readable output.
        print("\n" + "=" * 60)
        print("HALLUCINATION FIREWALL — FULL REPORT")
        print("=" * 60)
        if result.get("question"):
            print("Question : %s" % result["question"])
        print("\nOriginal Response:")
        print("  %s" % result["original_response"])
        summary = result["summary"]
        timing = result["timing"]
        print("\nSummary:")
        print("  Total claims      : %d" % summary["total_claims"])
        print("  Supported         : %d  ✅" % summary["supported"])
        print("  Hallucinated      : %d  ❌" % summary["hallucinated"])
        print("  Unverifiable      : %d  ⚠️" % summary["unverifiable"])
        print("  Corrected         : %d  🔧" % summary["corrected"])
        print("  Hallucination rate: %.0f%%" % (summary["hallucination_rate"] * 100))
        print("  Total time        : %.2fs" % timing.get("total_seconds", 0))
        print("\nClaim-by-claim breakdown:")
        icons = {"SUPPORTED": "✅", "HALLUCINATED": "❌", "UNVERIFIABLE": "⚠️ "}
        fix_icons = {"CORRECTED": "🔧", "UNCHANGED": "  ", "UNVERIFIABLE": "⚠️ ", "FAILED": "❌"}
        for i, r in enumerate(result["claims"], 1):
            label = r.get("final_label", "")
            status = r.get("correction_status", "")
            print("\n  [%d] %s %s  %s %s" % (
                i, icons.get(label, "❓"), label, fix_icons.get(status, ""), status,
            ))
            print("       Original  : %s" % r.get("claim", ""))
            corrected = r.get("corrected_claim", "")
            if corrected and corrected != r.get("claim", ""):
                print("       Corrected : %s" % corrected)
        print("\nFinal Response:")
        print("  %s" % result["final_response"])
        print("\n" + "=" * 60)


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import logging as _logging
    _logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not is_ollama_running():
        print("❌ Ollama not running. Run: ollama serve")
        sys.exit(1)
    print("✅ Ollama is running\n")

    passages = [
        "Albert Einstein was born on March 14, 1879, in Ulm, Germany.",
        "Einstein received the Nobel Prize in Physics in 1921 for the photoelectric effect.",
        "Einstein emigrated to the United States in December 1932.",
        "Einstein died on April 18, 1955, at Princeton Hospital in New Jersey.",
        "Python was created by Guido van Rossum and first released in 1991.",
        "The Eiffel Tower was completed in 1889 and stands 330 metres tall in Paris, France.",
    ]

    pipeline = HallucinationPipeline(use_selfcheck=False)
    pipeline.load_kb(passages)

    hallucinated_response = (
        "Einstein was born in France in 1885 and won the Nobel Prize in Chemistry in 1930. "
        "He emigrated to Canada in 1945 and died in Toronto in 1960."
    )
    result = pipeline.run(hallucinated_response, question="Tell me about Einstein's life")
    pipeline.print_result(result)
    sys.exit(0)
