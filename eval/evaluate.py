"""
eval/evaluate.py
────────────────
Benchmarks the hallucination detection pipeline against HaluEval.

HaluEval provides labelled (question, answer, hallucinated) triplets.
We run our verifier on each answer and compare against ground truth labels.

Metrics computed:
    Precision  — of claims flagged as hallucinated, how many actually were?
    Recall     — of actual hallucinations, how many did we catch?
    F1 Score   — harmonic mean of precision and recall
    Accuracy   — overall correct classifications

Usage:
    python eval/evaluate.py                  # full eval, fast mode
    python eval/evaluate.py --samples 50     # limit to 50 samples
    python eval/evaluate.py --selfcheck      # include SelfCheck (slow)
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from pipeline import HallucinationPipeline
from core.tracking import append_tracking_row

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Suppress noisy logs during evaluation
logging.getLogger("core.call_llm").setLevel(logging.ERROR)
logging.getLogger("core.verifier").setLevel(logging.ERROR)
logging.getLogger("core.selfcheck").setLevel(logging.ERROR)
logging.getLogger("core.nli_scorer").setLevel(logging.ERROR)
logging.getLogger("core.decomposer").setLevel(logging.ERROR)

RESULTS_PATH = "eval/results.json"


# ── Load HaluEval ──────────────────────────────────────────────────────────────

def load_halueval(split: str = "data", max_samples: int = 100) -> list[dict]:
    """
    Load HaluEval QA samples from HuggingFace.

    Each sample has:
        question        : str
        right_answer    : str — factually correct answer
        hallucinated_answer : str — answer with hallucinations injected

    Returns list of dicts with keys:
        question, answer, label (0=correct, 1=hallucinated)
    """
    print("Downloading HaluEval dataset from HuggingFace...")

    try:
        dataset = load_dataset("pminervini/HaluEval", "qa_samples")

        # Use the available split
        available = list(dataset.keys())
        split_name = available[0]
        data = dataset[split_name]

        print(f"Loaded {len(data)} samples from HaluEval ({split_name} split)")

        # Print actual column names so we can verify
        print(f"Columns: {data.column_names}")

    except Exception as e:
        print(f"HaluEval download failed: {e}")
        print("Using synthetic fallback dataset instead...")
        return _synthetic_fallback(max_samples)

    # Detect column names dynamically — dataset schema changed over time
    col_names = data.column_names
    print(f"Detected columns: {col_names}")

    # Map known variations of column names
    def find_col(candidates):
        for c in candidates:
            if c in col_names:
                return c
        return None

    question_col   = find_col(["question", "query", "input"])
    correct_col    = find_col(["right_answer", "correct_answer", "answer", "response"])
    hallucin_col   = find_col(["hallucinated_answer", "hallucinated_response", "wrong_answer"])

    print(f"Using columns → question='{question_col}' correct='{correct_col}' hallucinated='{hallucin_col}'")

    if not question_col or not correct_col or not hallucin_col:
        print("Could not find required columns. Falling back to synthetic dataset.")
        print(f"Available columns were: {col_names}")
        return _synthetic_fallback(max_samples)

    samples = []

    for i, row in enumerate(data):
        if len(samples) >= max_samples * 2:
            break

        question          = str(row.get(question_col, "")).strip()
        right_answer      = str(row.get(correct_col, "")).strip()
        hallucinated_answer = str(row.get(hallucin_col, "")).strip()

        if not question or not right_answer or not hallucinated_answer:
            continue

        # Add correct answer (label=0)
        samples.append({
            "question": question,
            "answer": right_answer,
            "label": 0,
            "source": "halueval_correct"
        })

        # Add hallucinated answer (label=1)
        samples.append({
            "question": question,
            "answer": hallucinated_answer,
            "label": 1,
            "source": "halueval_hallucinated"
        })

    # Balance and limit
    correct = [s for s in samples if s["label"] == 0][:max_samples // 2]
    hallucinated = [s for s in samples if s["label"] == 1][:max_samples // 2]
    balanced = correct + hallucinated

    print(f"Using {len(balanced)} samples ({len(correct)} correct, {len(hallucinated)} hallucinated)")
    return balanced


def _synthetic_fallback(max_samples: int) -> list[dict]:
    """
    Synthetic fallback if HaluEval download fails.
    Uses hand-crafted examples covering common hallucination patterns.
    """
    print("Building synthetic evaluation set...")

    correct_samples = [
        {"question": "When was Einstein born?",
         "answer": "Albert Einstein was born on March 14, 1879 in Ulm, Germany.",
         "label": 0, "source": "synthetic"},
        {"question": "Who created Python?",
         "answer": "Python was created by Guido van Rossum and first released in 1991.",
         "label": 0, "source": "synthetic"},
        {"question": "Where is the Eiffel Tower?",
         "answer": "The Eiffel Tower is located in Paris, France and was completed in 1889.",
         "label": 0, "source": "synthetic"},
        {"question": "When did Einstein win the Nobel Prize?",
         "answer": "Einstein received the Nobel Prize in Physics in 1921.",
         "label": 0, "source": "synthetic"},
        {"question": "When was Python 3 released?",
         "answer": "Python 3.0 was released on December 3, 2008.",
         "label": 0, "source": "synthetic"},
    ]

    hallucinated_samples = [
        {"question": "When was Einstein born?",
         "answer": "Albert Einstein was born in 1895 in Paris, France.",
         "label": 1, "source": "synthetic"},
        {"question": "Who created Python?",
         "answer": "Python was invented by James Gosling at Sun Microsystems in 1995.",
         "label": 1, "source": "synthetic"},
        {"question": "Where is the Eiffel Tower?",
         "answer": "The Eiffel Tower is located in London, England and was built in 1920.",
         "label": 1, "source": "synthetic"},
        {"question": "When did Einstein win the Nobel Prize?",
         "answer": "Einstein received the Nobel Prize in Chemistry in 1930.",
         "label": 1, "source": "synthetic"},
        {"question": "When was Python 3 released?",
         "answer": "Python 3.0 was released in 2015 and was fully backward compatible.",
         "label": 1, "source": "synthetic"},
    ]

    samples = (correct_samples + hallucinated_samples)[:max_samples]
    print(f"Synthetic set: {len(samples)} samples")
    return samples


# ── Build evaluation KB ────────────────────────────────────────────────────────

def load_truthfulqa_generation(max_samples: int = 100) -> list[dict]:
    """
    TruthfulQA generation split: pair best_answer (correct) vs one incorrect answer.
    label 0 = likely non-hallucinated answer, 1 = known incorrect reference answer.
    """
    print("Loading TruthfulQA (generation) from HuggingFace...")
    try:
        dataset = load_dataset("truthful_qa", "generation")
    except Exception as e:
        print(f"TruthfulQA load failed: {e}")
        return []

    split = "validation" if "validation" in dataset else list(dataset.keys())[0]
    data = dataset[split]
    samples: list[dict] = []

    for row in data:
        if len(samples) >= max_samples:
            break
        q = str(row.get("question", "")).strip()
        best = str(row.get("best_answer", "")).strip()
        wrong_list = row.get("incorrect_answers") or []
        wrong = str(wrong_list[0]).strip() if wrong_list else ""

        if not q or not best:
            continue

        samples.append({"question": q, "answer": best, "label": 0, "source": "truthfulqa_best"})
        if wrong and wrong.lower() != best.lower():
            samples.append({"question": q, "answer": wrong, "label": 1, "source": "truthfulqa_incorrect"})

    print(f"TruthfulQA samples prepared: {len(samples)}")
    return samples


def run_truthfulqa_evaluation(max_samples: int = 100, use_selfcheck: bool = False) -> dict:
    print("\n" + "="*60)
    print("HALLUCINATION FIREWALL — TruthfulQA EVALUATION")
    print("="*60)

    samples = load_truthfulqa_generation(max_samples=max_samples)
    if not samples:
        return {"error": "no_samples"}

    pipeline = HallucinationPipeline(use_selfcheck=use_selfcheck)
    pipeline.load_kb(build_eval_kb())

    true_labels: list[int] = []
    predicted_labels: list[int] = []
    detailed_results: list[dict] = []
    errors = 0
    eval_start = time.time()

    for i, sample in enumerate(samples):
        pred = predict_hallucination(pipeline, sample)
        if pred["error"]:
            errors += 1
            continue
        true_labels.append(sample["label"])
        predicted_labels.append(pred["predicted_label"])
        detailed_results.append(
            {
                "index": i,
                "question": sample["question"],
                "answer": sample["answer"][:120],
                "true_label": sample["label"],
                "predicted_label": pred["predicted_label"],
                "hallucination_rate": pred["hallucination_rate"],
                "correct": pred["predicted_label"] == sample["label"],
                "source": sample.get("source", "truthfulqa"),
            }
        )

    metrics = compute_metrics(true_labels, predicted_labels)
    total_time = round(time.time() - eval_start, 1)
    report = {
        "metadata": {
            "benchmark": "truthfulqa_generation",
            "timestamp": datetime.now().isoformat(),
            "evaluated_samples": len(true_labels),
            "errors": errors,
            "use_selfcheck": use_selfcheck,
            "total_time_seconds": total_time,
        },
        "metrics": metrics,
        "detailed_results": detailed_results,
    }
    append_tracking_row(
        "eval/tracking.jsonl",
        {
            "dataset_tag": "truthfulqa_eval",
            "f1_score": metrics.get("f1_score"),
            "accuracy": metrics.get("accuracy"),
            "samples": len(true_labels),
        },
    )
    out_path = "eval/results_truthfulqa.json"
    os.makedirs("eval", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nTruthfulQA report saved → {out_path}")
    return report


def build_eval_kb() -> list[str]:
    """
    Knowledge base passages for evaluation.
    In a real project, use Wikipedia snippets relevant to your domain.
    """
    return [
        "Albert Einstein was born on March 14, 1879, in Ulm, Germany.",
        "Einstein received the Nobel Prize in Physics in 1921 for the photoelectric effect.",
        "Einstein developed the theory of special relativity published in 1905.",
        "Einstein emigrated to the United States in December 1932.",
        "Einstein died on April 18, 1955, at Princeton Hospital in New Jersey.",
        "Python was created by Guido van Rossum and first released in 1991.",
        "Python 3.0 was released on December 3, 2008 and was not backward compatible with Python 2.",
        "The Python programming language is named after Monty Python.",
        "The Eiffel Tower was completed in 1889 and stands 330 metres tall in Paris, France.",
        "The Taj Mahal is located in Agra, India and was completed around 1653.",
        "The Great Wall of China was built over centuries starting from the 7th century BC.",
        "India gained independence from British rule on August 15, 1947.",
        "The first iPhone was released by Apple on June 29, 2007.",
        "ChatGPT was launched by OpenAI in November 2022.",
        "The speed of light in a vacuum is approximately 299,792 kilometres per second.",
        "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "The human body has 206 bones in adulthood.",
        "DNA stands for deoxyribonucleic acid.",
        "The French Revolution began in 1789.",
        "World War II ended in 1945 with the surrender of Germany and Japan.",
    ]


# ── Core Evaluation ────────────────────────────────────────────────────────────

def predict_hallucination(pipeline: HallucinationPipeline, sample: dict) -> dict:
    """
    Run pipeline on one sample and return prediction.

    Returns dict with:
        predicted_label : 0 (correct) or 1 (hallucinated)
        hallucination_rate : float — fraction of claims flagged
        total_claims : int
        hallucinated_claims : int
        processing_time : float
    """
    answer = sample["answer"]
    question = sample.get("question", "")

    t0 = time.time()

    try:
        result = pipeline.run(llm_response=answer, question=question)
        summary = result.get("summary", {})

        total = summary.get("total_claims", 0)
        hallucinated = summary.get("hallucinated", 0)
        rate = summary.get("hallucination_rate", 0.0)

        # Classify as hallucinated if ANY claim is flagged
        # You can tune this threshold — e.g. rate > 0.3 for stricter detection
        predicted = 1 if hallucinated > 0 else 0

        return {
            "predicted_label": predicted,
            "hallucination_rate": rate,
            "total_claims": total,
            "hallucinated_claims": hallucinated,
            "processing_time": round(time.time() - t0, 2),
            "error": None
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {
            "predicted_label": 0,
            "hallucination_rate": 0.0,
            "total_claims": 0,
            "hallucinated_claims": 0,
            "processing_time": round(time.time() - t0, 2),
            "error": str(e)
        }


def compute_metrics(
    true_labels: list[int],
    predicted_labels: list[int]
) -> dict:
    """
    Compute precision, recall, F1, and accuracy.

    Convention:
        Positive class = 1 (hallucinated)
        Negative class = 0 (correct)
    """
    tp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(true_labels) if true_labels else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "total_samples": len(true_labels)
    }


# ── Main Evaluation Loop ───────────────────────────────────────────────────────

def run_evaluation(
    max_samples: int = 100,
    use_selfcheck: bool = False
) -> dict:
    """
    Full evaluation run.

    Args:
        max_samples  : How many samples to evaluate
        use_selfcheck: Include SelfCheck (much slower — use False for quick eval)

    Returns:
        Full evaluation report dict
    """

    print("\n" + "="*60)
    print("HALLUCINATION FIREWALL — EVALUATION")
    print("="*60)
    print(f"Samples     : {max_samples}")
    print(f"SelfCheck   : {'ON' if use_selfcheck else 'OFF (fast mode)'}")
    print("="*60)

    # ── Load dataset ───────────────────────────────────────────────────────────
    samples = load_halueval(max_samples=max_samples)

    # ── Init pipeline ──────────────────────────────────────────────────────────
    print("\nInitializing pipeline...")
    pipeline = HallucinationPipeline(use_selfcheck=use_selfcheck)
    pipeline.load_kb(build_eval_kb())
    print("Pipeline ready ✅\n")

    # ── Run predictions ────────────────────────────────────────────────────────
    true_labels = []
    predicted_labels = []
    detailed_results = []
    errors = 0

    total = len(samples)
    eval_start = time.time()

    for i, sample in enumerate(samples):
        label_str = "HALLUCINATED" if sample["label"] == 1 else "CORRECT"

        print(
            f"[{i+1:3d}/{total}] {label_str:<12} | "
            f"Q: {sample['question'][:45]:<45}",
            end=" | "
        )

        pred = predict_hallucination(pipeline, sample)

        if pred["error"]:
            errors += 1
            print(f"ERROR: {pred['error'][:40]}")
            continue

        true_labels.append(sample["label"])
        predicted_labels.append(pred["predicted_label"])

        # Running accuracy
        correct_so_far = sum(
            1 for t, p in zip(true_labels, predicted_labels) if t == p
        )
        running_acc = correct_so_far / len(true_labels)

        pred_str = "HAL" if pred["predicted_label"] == 1 else "OK "
        true_str = "HAL" if sample["label"] == 1 else "OK "
        match = "✅" if pred["predicted_label"] == sample["label"] else "❌"

        print(
            f"True={true_str} Pred={pred_str} {match} | "
            f"rate={pred['hallucination_rate']:.0%} | "
            f"acc={running_acc:.0%} | "
            f"{pred['processing_time']}s"
        )

        detailed_results.append({
            "index": i,
            "question": sample["question"],
            "answer": sample["answer"][:100],
            "true_label": sample["label"],
            "predicted_label": pred["predicted_label"],
            "hallucination_rate": pred["hallucination_rate"],
            "total_claims": pred["total_claims"],
            "hallucinated_claims": pred["hallucinated_claims"],
            "correct": pred["predicted_label"] == sample["label"],
            "processing_time": pred["processing_time"],
            "source": sample.get("source", "unknown")
        })

    total_time = round(time.time() - eval_start, 1)

    # ── Compute metrics ────────────────────────────────────────────────────────
    metrics = compute_metrics(true_labels, predicted_labels)

    # ── Build report ───────────────────────────────────────────────────────────
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_samples": total,
            "evaluated_samples": len(true_labels),
            "errors": errors,
            "use_selfcheck": use_selfcheck,
            "total_time_seconds": total_time,
            "avg_time_per_sample": round(total_time / max(len(true_labels), 1), 2)
        },
        "metrics": metrics,
        "detailed_results": detailed_results
    }

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Samples evaluated : {len(true_labels)}/{total}")
    print(f"Errors            : {errors}")
    print(f"Total time        : {total_time}s")
    print(f"Avg per sample    : {report['metadata']['avg_time_per_sample']}s")
    print()
    print(f"Precision  : {metrics['precision']:.4f}  ({metrics['precision']:.1%})")
    print(f"Recall     : {metrics['recall']:.4f}  ({metrics['recall']:.1%})")
    print(f"F1 Score   : {metrics['f1_score']:.4f}  ({metrics['f1_score']:.1%})")
    print(f"Accuracy   : {metrics['accuracy']:.4f}  ({metrics['accuracy']:.1%})")
    print()
    print(f"True Positives  : {metrics['true_positives']}")
    print(f"False Positives : {metrics['false_positives']}")
    print(f"True Negatives  : {metrics['true_negatives']}")
    print(f"False Negatives : {metrics['false_negatives']}")
    print("="*60)

    # ── Save report ────────────────────────────────────────────────────────────
    os.makedirs("eval", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved → {RESULTS_PATH}")

    append_tracking_row(
        "eval/tracking.jsonl",
        {
            "dataset_tag": "halueval_eval",
            "f1_score": metrics.get("f1_score"),
            "accuracy": metrics.get("accuracy"),
            "samples": len(true_labels),
        },
    )

    return report


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hallucination detection pipeline")
    parser.add_argument(
        "--samples", type=int, default=20,
        help="Number of samples to evaluate (default: 20)"
    )
    parser.add_argument(
        "--selfcheck", action="store_true",
        help="Enable SelfCheck (slower but more accurate)"
    )
    parser.add_argument(
        "--benchmark",
        choices=("halueval", "truthfulqa"),
        default="halueval",
        help="Which benchmark to run",
    )
    args = parser.parse_args()

    if args.benchmark == "truthfulqa":
        run_truthfulqa_evaluation(max_samples=args.samples, use_selfcheck=args.selfcheck)
    else:
        run_evaluation(
            max_samples=args.samples,
            use_selfcheck=args.selfcheck
        )