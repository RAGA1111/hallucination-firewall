"""
Central configuration for thresholds and pipeline behaviour.

Override via environment variables (see PipelineConfig.from_env).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class PipelineConfig:
    min_retrieval_score: float = 0.30
    top_k_retrieval: int = 5
    nli_entailment_threshold: float = 0.70
    nli_contradiction_threshold: float = 0.70
    fuse_high: float = 0.65
    fuse_low: float = 0.35
    selfcheck_consistency_threshold: float = 0.5
    wiki_min_similarity: float = 0.72
    kb_retrieve_min_score: float = 0.30
    use_sentence_chunks: bool = True
    dynamic_wikipedia: bool = True
    wiki_max_pages_per_run: int = 5
    wiki_max_queries: int = 8
    selfcheck_concurrency: int = 4
    claim_cache_max_entries: int = 512
    claim_cache_ttl_seconds: float = 0.0
    pipeline_response_cache_max: int = 64
    structured_log_dir: str = "logs/runs"
    tracking_log_path: str = "eval/tracking.jsonl"
    active_kb_expansion: bool = False
    active_kb_max_pages: int = 2
    active_kb_min_retrieval: float = 0.15
    selfcheck_samples: int = 3
    verify_claim_timeout_seconds: float = 120.0

    def cache_fingerprint(self) -> tuple:
        """Stable tuple for cache keys when config affects verification."""
        return (
            round(self.min_retrieval_score, 4),
            self.top_k_retrieval,
            round(self.nli_entailment_threshold, 4),
            round(self.nli_contradiction_threshold, 4),
            round(self.fuse_high, 4),
            round(self.fuse_low, 4),
            round(self.selfcheck_consistency_threshold, 4),
            round(self.kb_retrieve_min_score, 4),
        )

    @classmethod
    def from_env(cls, base: PipelineConfig | None = None) -> PipelineConfig:
        b = base or cls()
        return cls(
            min_retrieval_score=_env_float("HF_MIN_RETRIEVAL_SCORE", b.min_retrieval_score),
            top_k_retrieval=_env_int("HF_TOP_K_RETRIEVAL", b.top_k_retrieval),
            nli_entailment_threshold=_env_float("HF_NLI_ENTAIL_THRESHOLD", b.nli_entailment_threshold),
            nli_contradiction_threshold=_env_float("HF_NLI_CONTRADICT_THRESHOLD", b.nli_contradiction_threshold),
            fuse_high=_env_float("HF_FUSE_HIGH", b.fuse_high),
            fuse_low=_env_float("HF_FUSE_LOW", b.fuse_low),
            selfcheck_consistency_threshold=_env_float("HF_SELFCHECK_CONSISTENCY", b.selfcheck_consistency_threshold),
            wiki_min_similarity=_env_float("HF_WIKI_MIN_SIMILARITY", b.wiki_min_similarity),
            kb_retrieve_min_score=_env_float("HF_KB_RETRIEVE_MIN_SCORE", b.kb_retrieve_min_score),
            use_sentence_chunks=_env_bool("HF_USE_SENTENCE_CHUNKS", b.use_sentence_chunks),
            dynamic_wikipedia=_env_bool("HF_DYNAMIC_WIKIPEDIA", b.dynamic_wikipedia),
            wiki_max_pages_per_run=_env_int("HF_WIKI_MAX_PAGES", b.wiki_max_pages_per_run),
            wiki_max_queries=_env_int("HF_WIKI_MAX_QUERIES", b.wiki_max_queries),
            selfcheck_concurrency=_env_int("HF_SELFCHECK_CONCURRENCY", b.selfcheck_concurrency),
            claim_cache_max_entries=_env_int("HF_CLAIM_CACHE_MAX", b.claim_cache_max_entries),
            claim_cache_ttl_seconds=_env_float("HF_CLAIM_CACHE_TTL", b.claim_cache_ttl_seconds),
            pipeline_response_cache_max=_env_int("HF_PIPELINE_CACHE_MAX", b.pipeline_response_cache_max),
            structured_log_dir=os.environ.get("HF_STRUCTURED_LOG_DIR", b.structured_log_dir),
            tracking_log_path=os.environ.get("HF_TRACKING_LOG_PATH", b.tracking_log_path),
            active_kb_expansion=_env_bool("HF_ACTIVE_KB_EXPANSION", b.active_kb_expansion),
            active_kb_max_pages=_env_int("HF_ACTIVE_KB_MAX_PAGES", b.active_kb_max_pages),
            active_kb_min_retrieval=_env_float("HF_ACTIVE_KB_MIN_RETRIEVAL", b.active_kb_min_retrieval),
            selfcheck_samples=_env_int("HF_SELFCHECK_SAMPLES", b.selfcheck_samples),
            verify_claim_timeout_seconds=_env_float("HF_VERIFY_CLAIM_TIMEOUT", b.verify_claim_timeout_seconds),
        )


DEFAULT_CONFIG = PipelineConfig()
