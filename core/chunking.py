"""
Sentence-level chunking for retrieval quality.
"""

from __future__ import annotations

import re

# Sentence boundaries: period/exclamation/question followed by space or end
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_into_sentences(text: str) -> list[str]:
    """Split plain text into sentences (heuristic, no external NLP deps)."""
    if not text or not text.strip():
        return []
    parts = _SENTENCE_SPLIT.split(text.strip())
    out: list[str] = []
    for p in parts:
        s = p.strip()
        if len(s) < 2:
            continue
        out.append(s)
    return out if out else [text.strip()]


def chunk_passages_to_sentences(passages: list[str]) -> list[str]:
    """
    Expand each passage into individual sentence chunks, deduplicated in order.
    """
    seen: set[str] = set()
    chunks: list[str] = []
    for passage in passages:
        if not passage or not str(passage).strip():
            continue
        for sent in split_into_sentences(str(passage)):
            key = sent.lower()
            if key in seen:
                continue
            seen.add(key)
            chunks.append(sent)
    return chunks
