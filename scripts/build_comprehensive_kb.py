"""
Build a multi-domain Knowledge Base for Hallucination Firewall.

Ingests articles across diverse topics:
- Physics & Relativity
- Astronomy & Space Exploration
- Computer Science, Artificial Intelligence, & Programming
- World History & Civilizations
- Geography & Global Landmarks
- Biology, Genetics, & Medicine
- Chemistry & Chemical Elements
- Environmental Science & Earth Sciences
- Literature, Philosophy, & Art
- Economics & Global Systems
- Famous Historical Figures & Biographies

Usage:
  python scripts/build_comprehensive_kb.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.chunking import chunk_passages_to_sentences  # noqa: E402
from core.knowledge_base import INDEX_PATH, PASSAGES_PATH, KnowledgeBase  # noqa: E402
from core.wiki_ingest import fetch_wikipedia_passages  # noqa: E402

# Curated set of high-yield Wikipedia topics covering major domain knowledge
COMPREHENSIVE_TOPICS = [
    # Physics & Space
    "Albert Einstein",
    "Isaac Newton",
    "General relativity",
    "Quantum mechanics",
    "Speed of light",
    "Big Bang",
    "Solar System",
    "Apollo 11",
    "James Webb Space Telescope",
    "International Space Station",
    "Mars rover",
    "Black hole",

    # Computer Science & Technology
    "Python (programming language)",
    "C (programming language)",
    "JavaScript",
    "Artificial intelligence",
    "Machine learning",
    "Linux",
    "Internet",
    "Computer hardware",
    "Alan Turing",
    "Ada Lovelace",
    "ChatGPT",
    "Quantum computing",

    # Geography & Landmarks
    "Eiffel Tower",
    "Taj Mahal",
    "Great Wall of China",
    "Statue of Liberty",
    "Mount Everest",
    "Amazon rainforest",
    "Great Barrier Reef",
    "Grand Canyon",
    "Pyramids of Giza",
    "Colosseum",

    # History & World Civilizations
    "World War II",
    "World War I",
    "Ancient Rome",
    "Ancient Egypt",
    "French Revolution",
    "Industrial Revolution",
    "United Nations",
    "Indian Independence Movement",
    "American Revolutionary War",
    "Silk Road",

    # Biology, Genetics & Medicine
    "DNA",
    "Photosynthesis",
    "Human body",
    "Penicillin",
    "Vaccine",
    "Evolution",
    "Cell (biology)",
    "Immune system",
    "Brain",
    "Genomics",

    # Chemistry & Earth Science
    "Periodic table",
    "Water",
    "Climate change",
    "Plate tectonics",
    "Atmosphere of Earth",
    "Renewable energy",

    # Literature & Philosophy
    "William Shakespeare",
    "Socrates",
    "Renaissance",
    "Nobel Prize",
    "Aristotle",
]


def build_kb(
    topics: list[str] = COMPREHENSIVE_TOPICS,
    out_passages: str = PASSAGES_PATH,
    out_index: str = INDEX_PATH,
    max_pages_per_topic: int = 1,
) -> None:
    print(f"Fetching passages for {len(topics)} topics across major domains...")
    
    # 1. Start with high quality seed facts to guarantee exact coverage of key facts
    seed_passages = [
        "Albert Einstein was born on March 14, 1879, in Ulm, in the Kingdom of Württemberg in the German Empire.",
        "Einstein received the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect.",
        "Einstein developed the theory of special relativity in 1905, published in his paper 'On the Electrodynamics of Moving Bodies'.",
        "Einstein emigrated to the United States in December 1932 and joined the Institute for Advanced Study in Princeton, New Jersey.",
        "Einstein died on April 18, 1955, at the age of 76, at Princeton Hospital in New Jersey.",
        "Isaac Newton published Philosophiæ Naturalis Principia Mathematica in 1687, formulating the laws of motion and universal gravitation.",
        "Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity.",
        "Marie Curie was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.",
        "Python programming language was created by Guido van Rossum and first released in 1991.",
        "Python 3.0 was released on December 3, 2008 and was not backward compatible with Python 2.",
        "The C programming language was developed by Dennis Ritchie at Bell Labs between 1972 and 1973.",
        "JavaScript was created by Brendan Eich in 1995 while working at Netscape Communications.",
        "The first iPhone was released by Apple Inc. on June 29, 2007.",
        "ChatGPT was launched by OpenAI in November 2022 as a prototype artificial intelligence chatbot.",
        "Linux operating system kernel was created by Linus Torvalds and released on September 17, 1991.",
        "The Eiffel Tower is located in Paris, France, on the Champ de Mars, and was completed in 1889.",
        "The Eiffel Tower stands 330 metres tall including its broadcast antenna.",
        "The Taj Mahal is an ivory-white marble mausoleum located on the right bank of the river Yamuna in Agra, India, completed around 1653.",
        "The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states.",
        "The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor in New York City, dedicated on October 28, 1886.",
        "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas, standing at 8,848.86 metres.",
        "The Indian Space Research Organisation (ISRO) was founded on August 15, 1969, headquartered in Bengaluru, India.",
        "India gained independence from British rule on August 15, 1947.",
        "The Apollo 11 mission landed the first humans on the Moon on July 20, 1969, with Neil Armstrong becoming the first person to walk on the Moon.",
        "World War II lasted from 1939 to 1945 and involved the vast majority of the world's countries.",
        "The United Nations (UN) was established on October 24, 1945, after World War II.",
        "DNA (deoxyribonucleic acid) is a molecule composed of two polynucleotide chains that coil around each other to form a double helix carrying genetic instructions.",
        "Photosynthesis is a biological process used by plants and other organisms to convert light energy into chemical energy.",
        "Penicillin was discovered in 1928 by Alexander Fleming as the first effective antibiotic.",
        "The speed of light in vacuum, commonly denoted c, is a universal physical constant equal to 299,792,458 metres per second.",
        "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles.",
        "The French Revolution was a period of radical political and societal change in France that began with the Estates-General of 1789 and ended in November 1799.",
        "The Colosseum is an oval amphitheatre in the centre of the city of Rome, Italy, built under the Flavian dynasty between AD 72 and AD 80.",
        "Alan Turing was an English mathematician, computer scientist, logician, cryptanalyst, philosopher, and theoretical biologist.",
        "William Shakespeare was an English playwright, poet and actor, widely regarded as the greatest writer in the English language.",
    ]

    # 2. Fetch additional Wikipedia passages in parallel
    fetched_passages: list[str] = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_topic(topic: str) -> list[str]:
        try:
            return fetch_wikipedia_passages([topic], max_pages=max_pages_per_topic, max_chars_per_page=3000)
        except Exception as e:
            print(f"  Warning: failed to fetch {topic}: {e}")
            return []

    print(f"  Fetching {len(topics)} topics concurrently with 10 workers...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_fetch_topic, topic): topic for topic in topics}
        for future in as_completed(futures):
            res = future.result()
            if res:
                fetched_passages.extend(res)

    all_raw = seed_passages + fetched_passages


    # Clean and deduplicate
    seen = set()
    cleaned = []
    for s in all_raw:
        st = str(s).strip()
        if len(st) >= 20 and st.lower() not in seen:
            seen.add(st.lower())
            cleaned.append(st)

    print(f"Collected {len(cleaned)} unique passages. Chunking into sentence-level units...")
    to_index = chunk_passages_to_sentences(cleaned)
    print(f"Total sentence chunks to index: {len(to_index)}")

    os.makedirs(os.path.dirname(out_passages) or ".", exist_ok=True)

    kb = KnowledgeBase()
    kb.build(to_index)
    kb.save(index_path=out_index, passages_path=out_passages)

    print("\n" + "="*50)
    print(f"[OK] Knowledge Base build complete!")
    print(f"   Passages file: {out_passages} ({len(to_index)} passages)")
    print(f"   FAISS index  : {out_index}")
    print("="*50)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build comprehensive multi-domain Knowledge Base")
    parser.add_argument("--out-passages", default=PASSAGES_PATH, help="Output passages JSON path")
    parser.add_argument("--out-index", default=INDEX_PATH, help="Output FAISS index path")
    args = parser.parse_args()

    build_kb(out_passages=args.out_passages, out_index=args.out_index)
