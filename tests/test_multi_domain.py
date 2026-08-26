"""
Multi-domain test suite for Hallucination Firewall.

Tests verification and self-correction across non-Einstein topics:
1. Computer Science & AI
2. Geography & Famous Landmarks
3. Space Exploration & Astronomy
4. Biology & Medicine
5. World History & Civilizations
6. Intentionally Hallucinated Non-Einstein Claims (e.g. Eiffel Tower in Tokyo built in 1995)
"""

import sys
import os
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline import HallucinationPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def test_multi_domain_prompts():
    print("=" * 60)
    print("TESTING HALLUCINATION FIREWALL ACROSS MULTIPLE DOMAINS")
    print("=" * 60)

    # Initialize pipeline with disk KB
    pipe = HallucinationPipeline(use_selfcheck=False)
    pipe.load_kb_auto()

    info = pipe.kb.info()
    print(f"Loaded Knowledge Base: {info['total_passages']} passages indexed [OK]\n")

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
            "domain": "Geography & Landmarks (Correct)",
            "prompt": "The Eiffel Tower is located in Paris, France, on the Champ de Mars, and was completed in 1889.",
            "expect_hallucinated": False,
        },
        {
            "domain": "Geography & Landmarks (Hallucinated)",
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
            "domain": "History (Hallucinated)",
            "prompt": "World War II began in 1990 and ended in 2005.",
            "expect_hallucinated": True,
        },
    ]

    passed = 0
    total = len(test_cases)

    for idx, tc in enumerate(test_cases, 1):
        print(f"\n[{idx}/{total}] Domain: {tc['domain']}")
        print(f"    Input Response: \"{tc['prompt']}\"")
        
        result = pipe.run(llm_response=tc['prompt'])
        summary = result.get('summary', {})
        final_resp = result.get('final_response', '')

        print(f"    Total Claims      : {summary.get('total_claims')}")
        print(f"    Supported Claims  : {summary.get('supported')}")
        print(f"    Hallucinated      : {summary.get('hallucinated')}")
        print(f"    Corrected Claims  : {summary.get('corrected')}")
        print(f"    Final Output      : \"{final_resp}\"")

        is_hal = summary.get('hallucinated', 0) > 0
        if is_hal == tc['expect_hallucinated']:
            print(f"    Verdict: [OK] PASSED (Matching expected classification)")
            passed += 1
        else:
            print(f"    Verdict: [WARNING] MISMATCH (Expected hallucination={tc['expect_hallucinated']}, got={is_hal})")


    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{total} test cases passed ({passed/total:.0%})")
    print("=" * 60)


if __name__ == "__main__":
    test_multi_domain_prompts()
