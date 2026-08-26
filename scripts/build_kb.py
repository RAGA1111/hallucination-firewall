"""
Build or extend the FAISS KB from Wikipedia titles/URLs, plain text, or PDF.

Usage:
  python scripts/build_kb.py --wiki-title "Albert Einstein" --out data/kb_passages.json
  python scripts/build_kb.py --text-file myfacts.txt --chunk-sentences
  python scripts/build_kb.py --pdf paper.pdf --out data/custom_passages.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.chunking import chunk_passages_to_sentences  # noqa: E402
from core.knowledge_base import INDEX_PATH, PASSAGES_PATH, KnowledgeBase  # noqa: E402
from core.knowledge_base import load_passages_from_file  # noqa: E402
from core.wiki_ingest import fetch_wikipedia_passages  # noqa: E402


def _read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("pypdf is required for --pdf (pip install pypdf)") from e
    reader = PdfReader(path)
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts)


def _wiki_title_from_url(url: str) -> str | None:
    m = re.search(r"wikipedia\.org/wiki/([^?#]+)", url)
    if not m:
        return None
    from urllib.parse import unquote

    return unquote(m.group(1)).replace("_", " ")


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest passages into KB JSON + FAISS index")
    p.add_argument("--out", default=os.path.join("data", "kb_passages.json"), help="Output passages JSON")
    p.add_argument("--index", default=INDEX_PATH, help="FAISS index output path")
    p.add_argument("--wiki-title", default=None, help="Wikipedia article title")
    p.add_argument("--wiki-url", default=None, help="Wikipedia article URL")
    p.add_argument("--text-file", default=None, help="Plain text file (paragraphs = blank lines)")
    p.add_argument("--pdf", default=None, help="Path to PDF file")
    p.add_argument("--chunk-sentences", action="store_true", help="Split into sentence-level chunks before indexing")
    p.add_argument("--merge-existing", action="store_true", help="Merge with existing --out JSON if present")
    args = p.parse_args()

    passages: list[str] = []

    if args.merge_existing and os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            passages.extend(json.load(f))

    if args.text_file:
        passages.extend(load_passages_from_file(args.text_file))

    if args.pdf:
        passages.append(_read_pdf(args.pdf))

    wiki_query = args.wiki_title
    if args.wiki_url:
        wiki_query = _wiki_title_from_url(args.wiki_url) or wiki_query

    if wiki_query:
        passages.extend(
            fetch_wikipedia_passages(
                [wiki_query],
                max_pages=3,
                max_chars_per_page=12000,
            )
        )

    passages = [str(x).strip() for x in passages if str(x).strip()]
    if not passages:
        print("No passages collected. Provide --wiki-title, --text-file, and/or --pdf.")
        sys.exit(1)

    if args.chunk_sentences:
        to_index = chunk_passages_to_sentences(passages)
    else:
        to_index = passages

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kb = KnowledgeBase()
    kb.build(to_index)
    kb.save(index_path=args.index, passages_path=args.out)
    print(f"Wrote {len(to_index)} indexed passages → {args.out}")
    print(f"FAISS index → {args.index}")


if __name__ == "__main__":
    main()
