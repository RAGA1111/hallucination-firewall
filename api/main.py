import os
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pipeline import HallucinationPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Global pipeline instance ───────────────────────────────────────────────────
# Initialized once at startup — not per request
# This is critical: loading NLI + KB per request would be 20+ seconds each time

pipeline: HallucinationPipeline | None = None
startup_time: float | None = None


# ── Sample KB — replace with your domain passages ─────────────────────────────

SAMPLE_KB_PASSAGES = [
    # Physics & Science
    "Albert Einstein was born on March 14, 1879, in Ulm, in the Kingdom of Württemberg in the German Empire.",
    "Einstein received the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect.",
    "Einstein developed the theory of special relativity in 1905, published in his paper 'On the Electrodynamics of Moving Bodies'.",
    "Einstein emigrated to the United States in December 1932 and joined the Institute for Advanced Study in Princeton, New Jersey.",
    "Einstein died on April 18, 1955, at the age of 76, at Princeton Hospital in New Jersey.",
    "Isaac Newton published Philosophiæ Naturalis Principia Mathematica in 1687, formulating the laws of motion and universal gravitation.",
    "Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity.",
    "Marie Curie was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.",
    
    # Technology & Programming
    "Python programming language was created by Guido van Rossum and first released in 1991.",
    "Python 3.0 was released on December 3, 2008 and was not backward compatible with Python 2.",
    "The C programming language was developed by Dennis Ritchie at Bell Labs between 1972 and 1973.",
    "JavaScript was created by Brendan Eich in 1995 while working at Netscape Communications.",
    "The first iPhone was released by Apple Inc. on June 29, 2007.",
    "ChatGPT was launched by OpenAI in November 2022 as a prototype artificial intelligence chatbot.",
    "Linux operating system kernel was created by Linus Torvalds and released on September 17, 1991.",
    
    # Geography & Landmarks
    "The Eiffel Tower is located in Paris, France, on the Champ de Mars, and was completed in 1889.",
    "The Eiffel Tower stands 330 metres tall including its broadcast antenna.",
    "The Taj Mahal is an ivory-white marble mausoleum located on the right bank of the river Yamuna in Agra, India, completed around 1653.",
    "The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states.",
    "The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor in New York City, dedicated on October 28, 1886.",
    "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas, standing at 8,848.86 metres.",

    # History & Space
    "The Indian Space Research Organisation (ISRO) was founded on August 15, 1969, headquartered in Bengaluru, India.",
    "India gained independence from British rule on August 15, 1947.",
    "The Apollo 11 mission landed the first humans on the Moon on July 20, 1969, with Neil Armstrong becoming the first person to walk on the Moon.",
    "World War II lasted from 1939 to 1945 and involved the vast majority of the world's countries.",
    "The United Nations (UN) was established on October 24, 1945, after World War II.",
]


# ── Lifespan handler — startup + shutdown ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline at startup, clean up at shutdown."""
    global pipeline, startup_time

    logger.info("="*50)
    logger.info("Starting Hallucination Firewall API...")
    logger.info("="*50)

    t0 = time.time()

    try:
        pipeline = HallucinationPipeline(use_selfcheck=True)
        pipeline.load_kb_auto(SAMPLE_KB_PASSAGES)
        startup_time = round(time.time() - t0, 2)
        logger.info(f"Pipeline ready in {startup_time}s ✅")

    except Exception as e:
        logger.error(f"Pipeline startup failed: {e}")
        raise

    yield  # API runs here

    logger.info("Shutting down API...")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hallucination Firewall API",
    description=(
        "Inference-time hallucination detection and self-correction for LLM outputs. "
        "Decomposes responses into atomic claims, verifies each via RAG + NLI + SelfCheck, "
        "and rewrites hallucinated claims using retrieved evidence."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Allow all origins for local dev — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ──────────────────────────────────────────────────

class VerifyRequest(BaseModel):
    response: str = Field(
        description="The raw LLM response to verify"
    )
    question: str = Field(
        default="",
        description="Original question that produced the response (optional)"
    )
    use_selfcheck: bool = Field(
        default=True,
        description="Whether to run SelfCheck consistency sampling (slower but more accurate)"
    )


class AskRequest(BaseModel):
    question: str = Field(
        description="Question to ask the LLM — response will be auto-verified"
    )


class AddPassagesRequest(BaseModel):
    passages: list[str] = Field(
        description="List of text passages to add to the knowledge base"
    )


class ClaimSummary(BaseModel):
    claim: str
    final_label: str
    fused_score: float
    claim_confidence: float | None = None
    retrieval_score: float | None = None
    nli_label: str
    nli_confidence: float
    nli_all_scores: dict | None = None
    hallucination_type: str | None = None
    selfcheck_label: str
    selfcheck_score: float
    corrected_claim: str
    correction_status: str
    explanation: str


class VerifyResponse(BaseModel):
    original_response: str
    question: str
    final_response: str
    summary: dict
    claims: list
    timing: dict
    status: str
    annotated_sentences: list | None = None
    final_response_html: str | None = None
    error: str | None = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """API root — basic info."""
    return {
        "name": "Hallucination Firewall API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /verify": "Verify an LLM response for hallucinations",
            "POST /ask": "Generate + verify an answer to a question",
            "POST /kb/add": "Add passages to the knowledge base",
            "GET /kb/info": "Get knowledge base statistics",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
def health():
    """Health check for production deployment."""
    provider_configured = bool(os.getenv("GROQ_API_KEY"))
    pipeline_ready = pipeline is not None and pipeline.kb_loaded

    status = "healthy" if pipeline_ready else "initializing"

    return {
        "status": status,
        "version": os.getenv("BUILD_VERSION", "dev"),
        "provider": "groq" if provider_configured else "not configured",
        "pipeline": "ready" if pipeline_ready else "not initialized",
        "kb_passages": (
            len(pipeline.kb.passages)
            if pipeline_ready and pipeline is not None
            else 0
        ),
        "startup_time_seconds": startup_time,
    }


@app.post("/verify", response_model=VerifyResponse)
async def verify(request: VerifyRequest):
    """
    Main endpoint — verify an LLM response for hallucinations.

    Takes a raw LLM response, runs the full pipeline:
    - Decomposes into atomic claims
    - Verifies each claim via RAG + NLI + SelfCheck
    - Corrects hallucinated claims using retrieved evidence
    - Returns structured result with per-claim breakdown
    """
    if not pipeline or not pipeline.kb_loaded:
        raise HTTPException(status_code=503, detail="Pipeline not ready. Try again shortly.")

    if not request.response.strip():
        raise HTTPException(status_code=400, detail="Response field cannot be empty.")

    if len(request.response) > 5000:
        raise HTTPException(status_code=400, detail="Response too long. Max 5000 characters.")

    try:
        logger.info(f"POST /verify — {len(request.response)} chars")

        result = await pipeline.run_async(
            llm_response=request.response,
            question=request.question,
            use_selfcheck=request.use_selfcheck
        )

        return result

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.post("/ask", response_model=VerifyResponse)
async def ask(request: AskRequest):
    """
    Ask mode — generate an LLM response to a question, then verify it.

    Useful for testing: just ask a question and see if the LLM hallucinates.
    """
    if not pipeline or not pipeline.kb_loaded:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        logger.info(f"POST /ask — '{request.question[:60]}'")
        result = await pipeline.ask_async(question=request.question)
        return result

    except Exception as e:
        logger.error(f"Ask error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.post("/kb/add")
def add_passages(request: AddPassagesRequest):
    """
    Add new passages to the knowledge base dynamically.
    Rebuilds the index — takes a few seconds.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")

    if not request.passages:
        raise HTTPException(status_code=400, detail="Passages list cannot be empty.")

    try:
        existing = pipeline.kb.passages.copy()
        new_passages = [p.strip() for p in request.passages if p.strip()]
        all_passages = existing + new_passages

        pipeline.load_kb(all_passages)
        logger.info(f"KB updated: +{len(new_passages)} passages, total={len(all_passages)}")

        return {
            "status": "success",
            "added": len(new_passages),
            "total_passages": len(all_passages)
        }

    except Exception as e:
        logger.error(f"KB update error: {e}")
        raise HTTPException(status_code=500, detail=f"KB update error: {str(e)}")


@app.get("/kb/info")
def kb_info():
    """Get knowledge base statistics."""
    if not pipeline or not pipeline.kb_loaded:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")

    info = pipeline.kb.info()
    info["sample_passages"] = pipeline.kb.passages[:3]
    return info


# ── Run directly ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # Don't reload — pipeline init is expensive
        log_level="info"
    )