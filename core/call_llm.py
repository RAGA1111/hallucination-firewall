"""
LLM client for Groq cloud inference.

Public API:
    call_llm(prompt, ...)       -> str
    call_llm_async(prompt, ...) -> str (coroutine)
    is_ollama_running()         -> bool

The is_ollama_running name is intentionally preserved because existing
callers use it as a provider health gate.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from dotenv import load_dotenv
from groq import Groq, AsyncGroq

load_dotenv()

logger = logging.getLogger(__name__)

__all__ = ["call_llm", "call_llm_async", "is_ollama_running"]

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

_SLEEP_BASE = 1.0
_DEFAULT_TIMEOUT = 30.0


def _get_api_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()


def _get_model(model: str | None) -> str:
    return model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)


def call_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_retries: int = 2,
    timeout: float = _DEFAULT_TIMEOUT,
) -> str:
    """
    Send a prompt to Groq and return the response.

    Returns an empty string on failure rather than raising.
    """

    api_key = _get_api_key()

    if not api_key:
        logger.error("call_llm | GROQ_API_KEY is not configured")
        return ""

    selected_model = _get_model(model)

    client = Groq(
        api_key=api_key,
        timeout=timeout,
    )

    for attempt in range(max_retries + 1):
        try:
            logger.debug(
                "call_llm | provider=groq model=%s attempt=%d temp=%.1f",
                selected_model,
                attempt + 1,
                temperature,
            )

            response = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=temperature,
            )

            text = ""

            if response.choices:
                text = (response.choices[0].message.content or "").strip()

            if text:
                logger.debug("call_llm | received %d chars", len(text))
                return text

            logger.warning(
                "call_llm | empty response (attempt %d/%d)",
                attempt + 1,
                max_retries + 1,
            )

        except Exception as exc:
            logger.warning(
                "call_llm | attempt %d/%d failed: %s",
                attempt + 1,
                max_retries + 1,
                exc,
            )

        if attempt < max_retries:
            time.sleep(_SLEEP_BASE * (2**attempt))

    logger.error("call_llm | all attempts failed")
    return ""


async def call_llm_async(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_retries: int = 2,
    timeout: float = _DEFAULT_TIMEOUT,
) -> str:
    """
    Async version of call_llm using AsyncGroq.

    Returns an empty string on failure rather than raising.
    """

    api_key = _get_api_key()

    if not api_key:
        logger.error("call_llm_async | GROQ_API_KEY is not configured")
        return ""

    selected_model = _get_model(model)

    client = AsyncGroq(
        api_key=api_key,
        timeout=timeout,
    )

    for attempt in range(max_retries + 1):
        try:
            logger.debug(
                "call_llm_async | provider=groq model=%s attempt=%d temp=%.1f",
                selected_model,
                attempt + 1,
                temperature,
            )

            response = await client.chat.completions.create(
                model=selected_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=temperature,
            )

            text = ""

            if response.choices:
                text = (response.choices[0].message.content or "").strip()

            if text:
                logger.debug(
                    "call_llm_async | received %d chars",
                    len(text),
                )
                return text

            logger.warning(
                "call_llm_async | empty response (attempt %d/%d)",
                attempt + 1,
                max_retries + 1,
            )

        except Exception as exc:
            logger.warning(
                "call_llm_async | attempt %d/%d failed: %s",
                attempt + 1,
                max_retries + 1,
                exc,
            )

        if attempt < max_retries:
            await asyncio.sleep(_SLEEP_BASE * (2**attempt))

    logger.error("call_llm_async | all attempts failed")
    return ""


def is_ollama_running() -> bool:
    """
    Backwards-compatible provider health check.

    The function name is intentionally unchanged because existing
    callers use it. It now checks whether the Groq provider is configured.
    """

    return bool(_get_api_key())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print("=== Checking Groq configuration ===")

    if not is_ollama_running():
        print("GROQ_API_KEY is not configured.")
    else:
        print("Groq API key is configured.")
        result = call_llm(
            "Reply with exactly: GROQ_OK",
            temperature=0.0,
        )
        print(result)