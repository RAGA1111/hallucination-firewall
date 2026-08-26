"""
LLM client for local Ollama inference.

Public API:
    call_llm(prompt, ...)       -> str
    call_llm_async(prompt, ...) -> str   (coroutine)
    is_ollama_running()         -> bool
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx
import requests

logger = logging.getLogger(__name__)

__all__ = ["call_llm", "call_llm_async", "is_ollama_running"]

# ── Constants ──────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_URL: str = f"{OLLAMA_BASE_URL}/api/generate"
DEFAULT_MODEL: str = "llama3.2:1b"

# Seconds to wait between retries
_SLEEP_EMPTY: float = 2.0    # empty response from model
_SLEEP_TIMEOUT: float = 3.0  # request timed out


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_payload(prompt: str, model: str, temperature: float) -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }


# ── Sync client ────────────────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_retries: int = 3,
    timeout: float = 120.0,
) -> str:
    """
    Send a prompt to a local Ollama model and return the response.

    Returns an empty string on failure — never raises.
    """
    payload = _build_payload(prompt, model, temperature)

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug("call_llm | model=%s attempt=%d temp=%.1f", model, attempt, temperature)

            response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            response.raise_for_status()

            text = response.json().get("response", "").strip()

            if not text:
                logger.warning("call_llm | empty response (attempt %d/%d)", attempt, max_retries)
                time.sleep(_SLEEP_EMPTY)
                continue

            logger.debug("call_llm | received %d chars", len(text))
            return text

        except requests.exceptions.ConnectionError:
            logger.error("call_llm | cannot connect to Ollama at %s", OLLAMA_BASE_URL)
            break  # No point retrying when Ollama is not running

        except requests.exceptions.Timeout:
            logger.warning("call_llm | timeout (attempt %d/%d)", attempt, max_retries)
            time.sleep(_SLEEP_TIMEOUT)

        except requests.exceptions.HTTPError as exc:
            logger.error("call_llm | HTTP error: %s", exc)
            break

        except Exception as exc:
            logger.error("call_llm | unexpected error: %s", exc)
            break

    logger.error("call_llm | all %d attempts failed", max_retries)
    return ""


# ── Async client ───────────────────────────────────────────────────────────────

async def call_llm_async(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_retries: int = 3,
    timeout: float = 120.0,
) -> str:
    """
    Async version of call_llm using httpx.

    Returns an empty string on failure — never raises.
    """
    payload = _build_payload(prompt, model, temperature)

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug("call_llm_async | model=%s attempt=%d temp=%.1f", model, attempt, temperature)

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(OLLAMA_URL, json=payload)
                response.raise_for_status()

                text = response.json().get("response", "").strip()

                if not text:
                    logger.warning(
                        "call_llm_async | empty response (attempt %d/%d)", attempt, max_retries
                    )
                    await asyncio.sleep(_SLEEP_EMPTY)
                    continue

                logger.debug("call_llm_async | received %d chars", len(text))
                return text

        except httpx.ConnectError:
            logger.error("call_llm_async | cannot connect to Ollama at %s", OLLAMA_BASE_URL)
            break

        except httpx.ReadTimeout:
            logger.warning("call_llm_async | timeout (attempt %d/%d)", attempt, max_retries)
            await asyncio.sleep(_SLEEP_TIMEOUT)

        except httpx.HTTPStatusError as exc:
            logger.error("call_llm_async | HTTP error: %s", exc)
            break

        except Exception as exc:
            logger.error("call_llm_async | unexpected error: %s", exc)
            break

    logger.error("call_llm_async | all %d attempts failed", max_retries)
    return ""


# ── Health check ───────────────────────────────────────────────────────────────

def is_ollama_running() -> bool:
    """Return True if the local Ollama server is reachable."""
    try:
        response = requests.get(OLLAMA_BASE_URL, timeout=5.0)
        return response.status_code == 200
    except (requests.exceptions.RequestException, OSError):
        return False


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    print("\n=== Checking Ollama status ===")
    if not is_ollama_running():
        print("❌ Ollama is not running. Run: ollama serve")
        sys.exit(1)
    print("✅ Ollama is running\n")

    print("=== Test 1: Basic factual question ===")
    reply = call_llm("What is the capital of France? Reply in one sentence.")
    print(f"Response: {reply}\n")

    print("=== Test 2: Temperature 0.0 (deterministic) ===")
    reply1 = call_llm("Give me a number between 1 and 10.", temperature=0.0)
    reply2 = call_llm("Give me a number between 1 and 10.", temperature=0.0)
    print(f"Run 1: {reply1}")
    print(f"Run 2: {reply2}")
    print(f"Same response (expected at temp=0.0): {reply1 == reply2}\n")

    print("=== Test 3: Temperature 0.7 (varied) ===")
    reply3 = call_llm("Give me a number between 1 and 10.", temperature=0.7)
    reply4 = call_llm("Give me a number between 1 and 10.", temperature=0.7)
    print(f"Run 1: {reply3}")
    print(f"Run 2: {reply4}")
    print(f"Different responses possible at temp=0.7: {reply3 != reply4}\n")

    print("=== All tests passed ✅ ===")
