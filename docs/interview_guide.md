# 🎙️ Hallucination Firewall — Interview Preparation Guide

This guide compiles high-probability technical questions and structured answers for interviews related to the **Hallucination Firewall** system. Use this to prepare for System Design, Machine Learning, and Engineering interviews.

---

## 🗺️ Part 1: High-Level System Architecture & Flow

### Q1: Can you describe the core architecture of your Hallucination Firewall?
**Answer**:
The Hallucination Firewall is an inference-time guardrail that verifies and corrects LLM outputs. It operates in four sequential stages:
1. **Claim Decomposition**: Splitting raw LLM text into atomic, testable claims using a fast instruction-tuned local LLM (`llama3.2:1b`).
2. **Retrieval**: Embedding each claim (`all-MiniLM-L6-v2`) and searching an internal knowledge base (`FAISS`). If the similarity score is low, it dynamically queries the Wikipedia API.
3. **Multi-Signal Verification**: Checking the claims against retrieved evidence using three parallel pipelines:
   * **NLI** (DeBERTa-v3 cross-encoder) to check if the evidence logically entails, contradicts, or is neutral to the claim.
   * **SelfCheck** (Consistency sampling) to check if the generator LLM consistently answers "yes" to the claim across multiple temperature-sampled runs.
   * **Symbolic Logic Evaluator** (Python AST parsing) to verify mathematical, numerical, and chronological constraints safely.
4. **Correction & Re-assembly**: Rewriting hallucinated claims based strictly on the evidence, gating them with NLI verification, and re-assembling the final text.

### Q2: Why is "Claim Decomposition" necessary? Why not verify the entire paragraph at once?
**Answer**:
Verifying a full paragraph at once suffers from **attention dilution** and **label washing**. 
If a paragraph contains 5 sentences, where 4 are true and 1 is a false hallucination, embedding the entire paragraph will return vectors heavily dominated by the 4 true sentences. The NLI model will see that the retrieved evidence supports the majority of the content and label the entire block as `SUPPORTED`. The single critical hallucination is washed out. 
Decomposing the response into atomic claims guarantees that every assertion is validated individually, ensuring that isolated lies are caught.

### Q3: How do you define an "atomic claim"?
**Answer**:
An atomic claim is a single sentence that contains a single, independently verifiable factual assertion (e.g., names, dates, numbers, locations, or scientific facts) without opinions, sentiment, or subjective qualifiers. For example:
* *Non-atomic*: "Einstein was born in Germany in 1879, where he studied physics before moving to the US."
* *Atomic*:
  1. "Albert Einstein was born in Germany."
  2. "Albert Einstein was born in 1879."
  3. "Albert Einstein emigrated to the United States."

---

## 🧠 Part 2: Vector Retrieval (RAG) & Grounding

### Q4: Why did you choose a local FAISS index instead of a hosted vector database like Pinecone or Milvus?
**Answer**:
For our target scale (localized knowledge bases under $100,000$ documents), a local flat inner product index (`IndexFlatIP` in FAISS) is optimal:
1. **Minimal Latency**: In-memory FAISS searches take $<1\text{ms}$, avoiding the network latency (typically $20\text{-}100\text{ms}$) of querying a cloud database.
2. **Operational Simplicity**: No external database connections, hosting costs, or API keys are required.
3. **Developer Cost**: It runs entirely on the host CPU for free, which is perfect for microservices and self-contained pipelines.
If the corpus scales to millions of documents, we would migrate to an HNSW (Hierarchical Navigable Small World) index in FAISS or a managed solution.

### Q5: Explain your "Wikipedia Fallback" logic. How does it prevent "silent failure"?
**Answer**:
If the internal knowledge base lacks information, a standard RAG system returns the closest vector, which is actually irrelevant, leading to a low similarity score.
To prevent silent verification failures:
1. We set a **minimum retrieval score** threshold (`0.30`).
2. If the internal search score is below `0.30`, we identify the claim as outside our KB domain.
3. We generate a search query, fetch the top Wikipedia article page summary, and embed it.
4. We calculate the semantic similarity of the Wikipedia summary to the claim. If it is $\ge 0.72$, we accept it as ground-truth evidence. If it is below, we treat the claim as `UNVERIFIABLE` and label it with a warning rather than blindly supporting it.

---

## ⚖️ Part 3: Verification Logic & Models

### Q6: Why do you use a Cross-Encoder DeBERTa model for NLI rather than a Bi-Encoder (like SentenceTransformers)?
**Answer**:
* **Bi-encoders** encode the premise and hypothesis independently into vector embeddings and compute cosine similarity. This measures **semantic similarity**, not **logic**. For example:
  * Premise: "The patient was prescribed Ibuprofen."
  * Hypothesis: "The patient was prescribed Aspirin."
  * A bi-encoder yields a high similarity score ($>0.85$) because both sentences share highly similar clinical contexts. However, they are logically contradictory.
* **Cross-encoders** feed both sentences together into the transformer, allowing full self-attention across all tokens. The model can identify negation words, subject-verb swaps, and numerical mismatches, outputting true logical entailment probability.

### Q7: What is "SelfCheckGPT" and how does consistency checking complement NLI?
**Answer**:
NLI relies entirely on the presence of external evidence. If a claim cannot be found in the KB or Wikipedia, NLI output is `NEUTRAL` (uncertain).
**SelfCheckGPT** leverages the consistency of the LLM's own internal weights:
1. We sample the generator LLM multiple times (e.g., 3x) at temperature `0.7` with a yes/no question about the claim.
2. If the LLM consistently answers "yes", its internal weights have high confidence in the fact.
3. If the LLM's answers are split (e.g., 1 yes, 2 nos), it means the LLM is **probabilistically guessing** (hallucinating).
This allows us to identify hallucinations even when external search indices fail to find evidence.

### Q8: How does the Symbolic Logic Checker prevent Python `eval()` code injection?
**Answer**:
Passing raw LLM strings directly to `eval()` is a massive security vulnerability (Remote Code Execution / RCE). 
To make it completely secure:
1. We ask the LLM to output math/date bounds as an assert statement (e.g., `assert 2026 - 1996 == 30`).
2. We compile the string into an Abstract Syntax Tree (AST) using Python's native `ast.parse(code, mode="eval")`.
3. We walk the AST and validate each node against a strict whitelist (e.g., `ast.BinOp`, `ast.Compare`, `ast.Constant`).
4. If the AST contains unauthorized nodes like `ast.Call` (function calls), `ast.Name` (variables), or `ast.Attribute`, the evaluator throws an error and rejects execution.
5. The validated AST is evaluated recursively using a safe, custom math parser.

---

## ⚡ Part 4: Production, Performance, and Scaling

### Q9: The pipeline uses multiple model calls. How do you keep latency low?
**Answer**:
To achieve sub-second latencies:
1. **Parallelism**: SelfCheck Ollama calls and NLI scoring runs are fired concurrently using python's `asyncio.gather`.
2. **Multi-tier Caching**:
   * **Response Cache**: Hashes the prompt and response; if identical, returns instantly ($<1\text{ms}$).
   * **Claim Cache**: Cache individual verified claim strings in an LRU cache.
3. **Tiny Models**: We use tiny, CPU-optimized models: `all-MiniLM-L6-v2` (90MB) and `nli-deberta-v3-small` (170MB). These run locally in memory, keeping execution times under $15\text{ms}$.

### Q10: How do you tune the trade-off between false positives and false negatives?
**Answer**:
This is controlled by adjusting NLI thresholds:
* **Entailment Threshold (default 0.70)**: Increasing this means we require stronger evidence to label a claim as `SUPPORTED`. This reduces false negatives (missed hallucinations) but increases false positives (marking true statements as unverified).
* **Contradiction Threshold (default 0.70)**: Lowering this makes the system highly sensitive to contradictory evidence, flagging more claims as `HALLUCINATED`.
* **F1 Optimization**: During benchmarking on the HaluEval dataset, we sweep these thresholds to find the peak F1 score (balancing precision and recall).

### Q11: How would you deploy this to handle enterprise traffic (e.g., 10,000+ RPS)?
**Answer**:
1. **Model Distillation / Server Hosting**: Host the NLI cross-encoder on a specialized inference framework like **Triton Inference Server** or **vLLM** with dynamic batching.
2. **Distributed Queue**: Decouple the FastAPI application from the verification workers using a message broker like Redis/RabbitMQ and Celery.
3. **Scalable Vector Database**: Replace the local FAISS index with a distributed cluster like Qdrant or Milvus.
4. **Shared Cache**: Move the in-memory LRU claim cache to a shared Redis cluster.
5. **Streaming verification**: Stream the generator LLM response, queue sentences as they finish generating, verify them asynchronously, and release them to the client UI progressively.
