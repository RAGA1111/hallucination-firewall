"""
Fast multi-domain test runner for Hallucination Firewall.
Uses Python sentence decomposer + DeBERTa cross-encoder NLI + FAISS disk KB.
Executes in under 5 seconds!
"""

import os
import sys
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Environment overrides for fast test run
os.environ["HF_DYNAMIC_WIKIPEDIA"] = "false"

from pipeline import HallucinationPipeline
from core.decomposer import decompose_without_llm

logging.basicConfig(level=logging.WARNING)

def run_fast_tests():
    print("=" * 65, flush=True)
    print("FAST MULTI-DOMAIN HALLUCINATION FIREWALL TEST SUITE", flush=True)
    print("=" * 65, flush=True)

    pipe = HallucinationPipeline(use_selfcheck=False)
    pipe.load_kb_auto()

    info = pipe.kb.info()
    print(f"Knowledge Base: {info['total_passages']} passages indexed in FAISS vector store.\n", flush=True)

    test_cases = [
        {
            "domain": "Computer Science",
            "prompt": "Python was created by Guido van Rossum and first released in 1991. Python 3.0 was released on December 3, 2008.",
            "expect_hallucinated": False,
        },
        {
            "domain": "Space Exploration",
            "prompt": "The Apollo 11 mission landed the first humans on the Moon on July 20, 1969, with Neil Armstrong becoming the first person to walk on the Moon.",
            "expect_hallucinated": False,
        },
        {
            "domain": "Geography & World Landmarks (Correct)",
            "prompt": "The Eiffel Tower is located in Paris, France, on the Champ de Mars, and was completed in 1889.",
            "expect_hallucinated": False,
        },
        {
            "domain": "Geography & World Landmarks (Hallucinated)",
            "prompt": "The Eiffel Tower is located in Tokyo, Japan, and was completed in 1995.",
            "expect_hallucinated": True,
        },
        {
            "domain": "Biology & Medicine",
            "prompt": "Penicillin was discovered in 1928 by Alexander Fleming as the first effective antibiotic.",
            "expect_hallucinated": False,
        },
        {
            "domain": "Physics",
            "prompt": "The speed of light in a vacuum is equal to 299,792,458 metres per second.",
            "expect_hallucinated": False,
        },
        {
            "domain": "World History (Hallucinated)",
            "prompt": "World War II began in 1990 and ended in 2005.",
            "expect_hallucinated": True,
        },
    ]

    passed = 0
    total = len(test_cases)

    for idx, tc in enumerate(test_cases, 1):
        print(f"[{idx}/{total}] Domain: {tc['domain']}", flush=True)
        print(f"    Text: \"{tc['prompt']}\"", flush=True)

        # Decompose using Python fallback decomposer to avoid slow LLM calls
        decomp = decompose_without_llm(tc['prompt'])
        claims = decomp.claims

        # Verify batch with DeBERTa NLI cross-encoder against 1,130 passage FAISS index
        verify_results = pipe.verifier.verify_batch(claims, use_selfcheck=False)
        corrected_results = pipe.regenerator.correct_batch(verify_results)
        summary = pipe._summarize(corrected_results)
        final_resp = pipe._build_final_response(corrected_results)

        is_hal = summary.get('hallucinated', 0) > 0

        print(f"    Claims Found : {summary.get('total_claims')} | Supported: {summary.get('supported')} | Hallucinated: {summary.get('hallucinated')}", flush=True)
        print(f"    Final Output : \"{final_resp}\"", flush=True)

        for c in corrected_results:
            label = c.get("final_label")
            score = c.get("fused_score", 0.0)
            print(f"      -> Claim: '{c.get('claim')}' => [{label}] (Score: {score:.3f}, NLI: {c.get('nli_label')})", flush=True)

        if is_hal == tc['expect_hallucinated']:
            print(f"    Verdict      : [OK] PASSED\n", flush=True)
            passed += 1
        else:
            print(f"    Verdict      : [WARNING] MISMATCH (Expected hallucination={tc['expect_hallucinated']}, got={is_hal})\n", flush=True)

    print("=" * 65, flush=True)
    print(f"TOTAL RESULT: {passed}/{total} domains verified ({passed/total:.0%})", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    run_fast_tests()

