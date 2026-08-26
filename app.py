"""
Hallucination Firewall — Streamlit demo UI.

Run from project root:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

import streamlit as st

# Project root on path (supports `streamlit run` from repo root)
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from api.main import SAMPLE_KB_PASSAGES  # noqa: E402
from pipeline import HallucinationPipeline  # noqa: E402

logging.basicConfig(level=logging.WARNING)
for name in ("core", "transformers", "sentence_transformers"):
    logging.getLogger(name).setLevel(logging.WARNING)


@st.cache_resource(show_spinner="Loading models and knowledge base (first run can take 1–2 minutes)…")
def _load_pipeline() -> HallucinationPipeline:
    """Heavy init: embedding model, FAISS, NLI — cached for the Streamlit session."""
    p = HallucinationPipeline(use_selfcheck=True)
    p.load_kb_auto(SAMPLE_KB_PASSAGES)
    return p



def _label_emoji(label: str) -> str:
    return {"SUPPORTED": "✅", "HALLUCINATED": "❌", "UNVERIFIABLE": "⚠️"}.get(label, "❓")


def _render_claim_card(claim: dict, idx: int) -> None:
    label = claim.get("final_label", "UNVERIFIABLE")
    emoji = _label_emoji(label)
    title = f"{emoji} **{label}** — {claim.get('claim', '')[:200]}{'…' if len(claim.get('claim', '')) > 200 else ''}"

    if label == "SUPPORTED":
        st.success(title)
    elif label == "HALLUCINATED":
        st.error(title)
    else:
        st.warning(title)

    corr = claim.get("corrected_claim") or ""
    orig = claim.get("claim") or ""
    status = claim.get("correction_status", "")
    if status == "CORRECTED" and corr and corr != orig:
        st.caption(f"→ **Corrected:** {corr}")
    elif status == "UNVERIFIABLE" and corr and corr != orig:
        st.caption(f"→ **Note:** {corr}")

    with st.expander("Scores & explanation", expanded=False):
        st.write(
            f"**NLI:** {claim.get('nli_label')} ({float(claim.get('nli_confidence', 0)):.0%})  \n"
            f"**SelfCheck:** {claim.get('selfcheck_label')} ({float(claim.get('selfcheck_score', 0)):.0%})  \n"
            f"**Fused / confidence:** {float(claim.get('fused_score', claim.get('claim_confidence', 0))):.3f}  \n"
            f"**Retrieval:** {float(claim.get('retrieval_score', 0)):.3f}  \n"
            f"**Type:** `{claim.get('hallucination_type', '—')}`"
        )
        if claim.get("nli_evidence"):
            st.text_area(
                "Evidence used",
                claim["nli_evidence"][:1200],
                height=120,
                disabled=True,
                key=f"ev_{idx}",
            )
        st.caption(claim.get("explanation", ""))


def main() -> None:
    st.set_page_config(
        page_title="Hallucination Firewall",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔥 Hallucination Firewall")
    st.caption("Decompose → retrieve → NLI → SelfCheck → correct. Pure Python UI.")

    with st.sidebar:
        st.subheader("Controls")
        use_selfcheck = st.toggle("SelfCheck sampling", value=True, help="Slower but more accurate (calls Ollama several times per claim).")
        dynamic_wiki = st.toggle(
            "Dynamic Wikipedia KB",
            value=os.environ.get("HF_DYNAMIC_WIKIPEDIA", "true").lower() in ("1", "true", "yes"),
            help="Augment the KB from Wikipedia using your question + claims (needs network).",
        )
        if dynamic_wiki:
            os.environ["HF_DYNAMIC_WIKIPEDIA"] = "true"
        else:
            os.environ["HF_DYNAMIC_WIKIPEDIA"] = "false"

        st.divider()
        st.subheader("Knowledge base")
        if st.button("Reload pipeline (after env / code change)"):
            st.cache_resource.clear()
            st.rerun()

        try:
            pipe = _load_pipeline()
            info = pipe.kb.info()
            st.metric("Indexed passages", info.get("total_passages", "—"))
            st.caption(f"Embedding: `{info.get('embedding_model', '')}`")
            st.caption("SelfCheck needs Ollama running locally (ollama serve).")
        except Exception as e:
            st.error(f"Pipeline failed to load: {e}")
            st.stop()

    col_main, _ = st.columns([2, 1])

    with col_main:
        question = st.text_input("Optional: original question", placeholder="e.g. Tell me about Einstein's life")
        response = st.text_area(
            "Paste any LLM response to verify",
            height=220,
            placeholder="Paste 2–10 sentences with factual claims…",
        )
        run = st.button("Run verification", type="primary", use_container_width=True)

    if not run:
        st.info("Paste a response and click **Run verification**.")
        return

    if not response.strip():
        st.warning("Please paste a non-empty response.")
        return

    pipe = _load_pipeline()

    with st.spinner("Running pipeline…"):
        t0 = time.time()
        try:
            result = pipe.run(llm_response=response.strip(), question=question.strip(), use_selfcheck=use_selfcheck)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            return
        elapsed = time.time() - t0

    status = result.get("status", "")
    if status != "OK":
        st.error(f"Status: **{status}** — {result.get('error', 'Unknown error')}")
        return

    summary = result.get("summary", {})
    claims = result.get("claims", [])
    timing = result.get("timing", {})

    st.subheader("Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Claims", summary.get("total_claims", 0))
    m2.metric("Hallucination rate", f"{float(summary.get('hallucination_rate', 0)):.0%}")
    m3.metric("Wall time", f"{timing.get('total_seconds', round(elapsed, 1))}s")
    m4.metric("Corrected", summary.get("corrected", 0))

    st.divider()
    for i, c in enumerate(claims, start=1):
        st.markdown(f"**Claim {i}**")
        _render_claim_card(c, i)

    st.divider()
    st.subheader("Final corrected response")
    st.write(result.get("final_response", ""))

    ann = result.get("annotated_sentences") or []
    if ann:
        with st.expander("Sentence-level map (heuristic)", expanded=False):
            for row in ann:
                lab = row.get("final_label", "")
                st.markdown(
                    f"- {_label_emoji(lab)} **{lab}** ({float(row.get('claim_confidence', 0)):.2f}) — {row.get('sentence', '')}"
                )

    html = result.get("final_response_html") or ""
    if html:
        with st.expander("HTML preview (for slides / embedding)", expanded=False):
            st.components.v1.html(
                f"<div style='font-family:system-ui,sans-serif;max-width:720px'>{html}</div>",
                height=min(500, 120 + 36 * max(len(ann), 1)),
                scrolling=True,
            )


if __name__ == "__main__":
    main()
