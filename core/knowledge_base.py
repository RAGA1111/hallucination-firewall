"""
Vector knowledge base built on FAISS + sentence-transformers.

Passages are embedded once, stored in a flat cosine-similarity index, and
retrieved by semantic similarity at query time. When the best internal score
falls below min_score, a Wikipedia fallback fetch is attempted.

Public API:
    KnowledgeBase          — build / retrieve / save / load
    load_passages_from_file — load passages from a plain-text file

Note: INDEX_PATH and PASSAGES_PATH are relative to the process working
directory. When running via the API or pipeline, CWD must be the project root.
"""

from __future__ import annotations

import json
import logging
import os

import faiss
import numpy as np
import wikipedia
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

__all__ = ["KnowledgeBase", "load_passages_from_file"]

# ── Constants ──────────────────────────────────────────────────────────────────

EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # 90 MB, CPU-friendly

# Paths relative to project root (CWD must be project root at runtime)
INDEX_PATH: str = "data/kb.index"
PASSAGES_PATH: str = "data/kb_passages.json"


# ── Knowledge base ─────────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    FAISS-backed vector store for evidence retrieval.

    Usage:
        kb = KnowledgeBase()
        kb.build(passages)           # build from a list of strings
        results = kb.retrieve(query) # returns list[dict] sorted by score
        kb.save()                    # persist index + passages to disk
        kb.load()                    # reload from disk
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self.embedder = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.passages: list[str] = []
        self.dimension: int | None = None
        logger.debug("Embedding model loaded")

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, passages: list[str], show_progress_bar: bool = False) -> None:
        """
        Build the FAISS index from *passages*.

        Args:
            passages         : list of text strings to index
            show_progress_bar: show tqdm bar during encoding (off by default
                               to avoid noise in production / API context)
        """
        if not passages:
            raise ValueError("Cannot build knowledge base from an empty passage list.")

        logger.info("Building knowledge base: %d passages", len(passages))

        embeddings = self.embedder.encode(
            passages,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        ).astype("float32")

        self.dimension = embeddings.shape[1]
        self.passages = list(passages)

        # Normalise for cosine similarity (inner product after L2 normalisation)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        logger.info("Knowledge base built: %d vectors | dim=%d", self.index.ntotal, self.dimension)

    # ── Retrieve ───────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.30,
        wiki_min_similarity: float = 0.72,
    ) -> list[dict]:
        """
        Return the top-k passages most semantically similar to *query*.

        If the best internal score is below *min_score*, a Wikipedia search
        is attempted. If the Wikipedia passage scores at or above
        *wiki_min_similarity* it replaces the internal results entirely
        (existing behaviour — internal results are not merged).

        Each result dict has keys: passage, score, source.
        """
        if self.index is None:
            raise RuntimeError("Knowledge base not built. Call build() or load() first.")

        q_vec = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_vec)

        k = min(top_k, len(self.passages))
        scores, indices = self.index.search(q_vec, k)

        results: list[dict] = []
        best_score: float = 0.0

        # idx == -1 means FAISS found fewer results than k (e.g. very small index)
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            s = round(float(score), 4)
            if s > best_score:
                best_score = s
            results.append({"passage": self.passages[idx], "score": s, "source": "Internal KB"})

        # Wikipedia fallback when internal KB has no strong match
        if best_score < min_score:
            wiki_results = self._wikipedia_fallback(query, q_vec, wiki_min_similarity)
            if wiki_results:
                # Wikipedia replaces internal results when it scores above the
                # similarity threshold — internal low-confidence results are not merged.
                results = wiki_results

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def _wikipedia_fallback(
        self,
        query: str,
        q_vec: np.ndarray,
        wiki_min_similarity: float,
    ) -> list[dict] | None:
        """
        Fetch a Wikipedia summary for *query* and score it against *q_vec*.

        Returns a single-item list if similarity >= wiki_min_similarity,
        otherwise returns None.
        """
        from core.wiki_ingest import _fetch_rest_summary

        logger.info("KB | internal score too low — trying Wikipedia for: %.60s", query)
        wiki_title = ""
        wiki_summary = ""

        try:
            hits = wikipedia.search(query, results=1)
            if hits:
                page = wikipedia.page(hits[0], auto_suggest=False)
                wiki_summary = page.summary[:800]
                wiki_title = page.title
        except Exception as exc:
            logger.debug("KB | wikipedia library failed (%s) — trying REST API", exc)
            rest = _fetch_rest_summary(query)
            if rest:
                wiki_title, wiki_summary = rest[0], rest[1][:800]

        if not wiki_summary:
            return None

        try:
            wiki_vec = self.embedder.encode(
                [wiki_summary], convert_to_numpy=True
            ).astype("float32")
            faiss.normalize_L2(wiki_vec)
            wiki_score = float(np.dot(q_vec[0], wiki_vec[0]))

            if wiki_score >= wiki_min_similarity:
                logger.info("KB | Wikipedia accepted: %r score=%.4f", wiki_title, wiki_score)
                return [
                    {
                        "passage": wiki_summary,
                        "score": round(wiki_score, 4),
                        "source": "Wikipedia (%s)" % wiki_title,
                    }
                ]
            logger.warning(
                "KB | Wikipedia rejected: similarity %.4f < threshold %.4f",
                wiki_score, wiki_min_similarity,
            )
        except Exception as exc:
            logger.warning("KB | Wikipedia scoring failed: %s", exc)

        return None

    def retrieve_best(self, query: str, min_score: float = 0.30) -> str:
        """
        Return the single highest-scoring passage string, or empty string
        if no passage meets *min_score*.
        """
        results = self.retrieve(query, top_k=1, min_score=min_score)
        if results and results[0]["score"] >= min_score:
            return results[0]["passage"]
        return ""

    # ── Persist ────────────────────────────────────────────────────────────────

    def save(self, index_path: str = INDEX_PATH, passages_path: str = PASSAGES_PATH) -> None:
        """Save the FAISS index and passages list to disk."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(passages_path, "w", encoding="utf-8") as f:
            json.dump(self.passages, f, ensure_ascii=False, indent=2)
        logger.info("Knowledge base saved: %s, %s", index_path, passages_path)

    def load(self, index_path: str = INDEX_PATH, passages_path: str = PASSAGES_PATH) -> bool:
        """
        Load a previously saved knowledge base from disk.

        Returns True on success, False if the files are not found.
        """
        if not os.path.exists(index_path) or not os.path.exists(passages_path):
            logger.warning("KB | no saved index found at %s", index_path)
            return False
        self.index = faiss.read_index(index_path)
        with open(passages_path, "r", encoding="utf-8") as f:
            self.passages = json.load(f)
        logger.info(
            "Knowledge base loaded: %d vectors | %d passages",
            self.index.ntotal, len(self.passages),
        )
        return True

    # ── Info ───────────────────────────────────────────────────────────────────

    def info(self) -> dict:
        """Return basic stats about the current knowledge base."""
        return {
            "total_passages": len(self.passages),
            "total_vectors": self.index.ntotal if self.index else 0,
            "embedding_model": EMBEDDING_MODEL,
            "dimension": self.dimension,
        }


# ── File helper ────────────────────────────────────────────────────────────────

def load_passages_from_file(filepath: str, delimiter: str = "\n\n") -> list[str]:
    """
    Load passages from a plain-text file split on *delimiter* (default: blank line).

    Args:
        filepath  : path to the text file
        delimiter : string used to split passages (default: double newline)

    Returns:
        List of non-empty passage strings.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError("Passage file not found: %s" % filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    passages = [p.strip() for p in content.split(delimiter) if p.strip()]
    logger.info("Loaded %d passages from %s", len(passages), filepath)
    return passages


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import logging as _logging
    _logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    sample_passages = [
        "Albert Einstein was born on March 14, 1879, in Ulm, Germany.",
        "Einstein received the Nobel Prize in Physics in 1921 for the photoelectric effect.",
        "Einstein emigrated to the United States in December 1932.",
        "Einstein died on April 18, 1955, at Princeton Hospital in New Jersey.",
        "Python programming language was created by Guido van Rossum and first released in 1991.",
        "The Eiffel Tower is located in Paris, France, and was completed in 1889.",
    ]

    print("\n=== Test 1: Build ===")
    kb = KnowledgeBase()
    kb.build(sample_passages)
    print("Info:", kb.info())

    print("\n=== Test 2: Retrieve ===")
    for q in ["When did Einstein win the Nobel Prize?", "How tall is the Eiffel Tower?"]:
        results = kb.retrieve(q, top_k=2)
        print(f"\nQuery: {q}")
        for r in results:
            print(f"  [{r['score']:.4f}] {r['passage'][:80]}")

    print("\n=== Test 3: retrieve_best ===")
    best = kb.retrieve_best("Einstein Nobel Prize")
    print("Best:", best[:80])

    print("\n=== Test 4: Save / Load ===")
    kb.save()
    kb2 = KnowledgeBase()
    assert kb2.load(), "load() should return True"
    result = kb2.retrieve_best("Einstein Nobel Prize")
    assert result, "loaded KB should return a result"
    print("Loaded KB works:", result[:80])

    print("\n=== All knowledge base tests complete ✅ ===")
    sys.exit(0)
