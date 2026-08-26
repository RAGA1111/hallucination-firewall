"""
Fetch Wikipedia text for dynamic knowledge-base expansion.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

import requests
import wikipedia

from core.chunking import split_into_sentences

logger = logging.getLogger(__name__)

# Set custom User-Agent to avoid HTTP 403 / JSONDecodeError from Wikipedia API
USER_AGENT = "HallucinationFirewallBot/1.0 (https://github.com/hallucination-firewall; dev@example.com)"
try:
    wikipedia.set_user_agent(USER_AGENT)
except Exception:
    pass


def _fetch_rest_summary(title_or_query: str) -> tuple[str, str] | None:
    """Fallback fetch using Wikipedia REST API directly when wikipedia package fails."""
    headers = {"User-Agent": USER_AGENT}
    try:
        # Search API
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": title_or_query,
            "format": "json",
            "utf8": 1,
        }
        res = requests.get(search_url, params=params, headers=headers, timeout=5)
        if res.status_code != 200:
            return None
        data = res.json()
        search_results = data.get("query", {}).get("search", [])
        if not search_results:
            return None
        
        top_title = search_results[0]["title"]

        # Fetch extract
        extract_params = {
            "action": "query",
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "titles": top_title,
            "format": "json",
            "utf8": 1,
        }
        res2 = requests.get(search_url, params=extract_params, headers=headers, timeout=5)
        if res2.status_code != 200:
            return None
        pages = res2.json().get("query", {}).get("pages", {})
        for _, pdata in pages.items():
            extract = pdata.get("extract", "").strip()
            if extract:
                return top_title, extract
    except Exception as e:
        logger.debug(f"Direct REST API fallback failed for '{title_or_query}': {e}")
    return None



def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        s = (x or "").strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def queries_from_question_and_claims(
    question: str,
    claims: list[str],
    max_queries: int = 8,
) -> list[str]:
    """Build short search strings from the user question and decomposed claims."""
    qs: list[str] = []
    if question and question.strip():
        q = re.sub(r"\s+", " ", question.strip())
        qs.append(q[:200])
    for c in claims[:12]:
        if not c or not str(c).strip():
            continue
        qs.append(re.sub(r"\s+", " ", str(c).strip())[:200])
    return _dedupe_preserve_order(qs)[:max_queries]


def fetch_wikipedia_passages(
    queries: list[str],
    max_pages: int = 5,
    max_chars_per_page: int = 8000,
    sentences_per_chunk_cap: int = 400,
) -> list[str]:
    """
    Run Wikipedia searches and return sentence-level chunks from page summaries + truncated content.
    """
    titles_seen: set[str] = set()
    raw_blocks: list[str] = []

    for q in queries:
        if len(titles_seen) >= max_pages:
            break
        hits = None
        try:
            hits = wikipedia.search(q, results=3)
        except Exception as e:
            logger.info(f"Standard wikipedia library failed for query={q!r}: {e}. Trying REST API...")
            rest_res = _fetch_rest_summary(q)
            if rest_res:
                rtitle, rsummary = rest_res
                if rtitle not in titles_seen:
                    titles_seen.add(rtitle)
                    raw_blocks.append(rsummary)
            continue

        if not hits:
            rest_res = _fetch_rest_summary(q)
            if rest_res:
                rtitle, rsummary = rest_res
                if rtitle not in titles_seen:
                    titles_seen.add(rtitle)
                    raw_blocks.append(rsummary)
            continue

        for title in hits or []:
            if len(titles_seen) >= max_pages:
                break
            if title in titles_seen:
                continue
            try:
                page = wikipedia.page(title, auto_suggest=False)
            except wikipedia.exceptions.DisambiguationError as e:
                try:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                except Exception:
                    continue
            except Exception:
                rest_res = _fetch_rest_summary(title)
                if rest_res:
                    rtitle, rsummary = rest_res
                    if rtitle not in titles_seen:
                        titles_seen.add(rtitle)
                        raw_blocks.append(rsummary)
                continue

            titles_seen.add(page.title)

            blob = (page.summary or "").strip()
            content = getattr(page, "content", "") or ""
            if content:
                blob = blob + "\n\n" + content[:max_chars_per_page]
            raw_blocks.append(blob)

    passages: list[str] = []
    for block in raw_blocks:
        for sent in split_into_sentences(block):
            passages.append(sent)
            if len(passages) >= sentences_per_chunk_cap:
                return _dedupe_preserve_order(passages)

    return _dedupe_preserve_order(passages)


def fetch_passages_for_claim(claim: str, max_pages: int = 1) -> list[str]:
    """Fetch a small set of sentence chunks for a single claim (active KB expansion)."""
    if not claim or not claim.strip():
        return []
    return fetch_wikipedia_passages(
        queries=[claim.strip()[:200]],
        max_pages=max_pages,
        max_chars_per_page=6000,
        sentences_per_chunk_cap=80,
    )
