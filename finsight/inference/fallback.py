"""Groq API fallback when local Ollama is slow or unavailable."""

import re
import time

import httpx

from finsight.config.logging import get_logger
from finsight.config.settings import settings
from finsight.inference.prompt_templates import SYSTEM_PROMPT

logger = get_logger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-70b-versatile"
OLLAMA_TIMEOUT = 60


def _clean_answer(text: str) -> str:
    """Strip training-artefact prefixes like '=== ANSWER ===' from model output."""
    text = re.sub(r"^=+\s*ANSWER\s*=+\s*\n?", "", text.strip())
    text = re.sub(r"^Based on the news and market data above, here is my detailed analysis:\s*\n?", "", text)
    return text.strip()


def query_with_fallback(user_prompt: str) -> dict:
    """Try Ollama first; fall back to Groq if Ollama is too slow or fails."""
    import ollama as ollama_client

    start = time.time()
    try:
        response = ollama_client.chat(
            model=settings.ollama_llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.4, "num_ctx": 8192, "num_predict": 1024},
        )
        latency = time.time() - start
        logger.info("ollama_query_success", latency=round(latency, 1))
        raw_answer = response["message"]["content"]
        answer = _clean_answer(raw_answer)
        return {
            "answer": answer,
            "provider": "ollama",
            "latency": round(latency, 1),
        }
    except Exception as e:
        logger.warning("ollama_failed_trying_groq", error=str(e))
        return _query_groq(user_prompt)


def _query_groq(user_prompt: str) -> dict:
    """Query Groq API as a fallback."""
    if not settings.groq_api_key:
        return {
            "answer": "Both local model and Groq fallback are unavailable. Please check Ollama status.",
            "provider": "none",
            "latency": 0,
        }

    start = time.time()
    try:
        resp = httpx.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 4096,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        latency = time.time() - start

        logger.info("groq_fallback_success", latency=round(latency, 1))
        return {
            "answer": answer,
            "provider": "groq",
            "latency": round(latency, 1),
        }
    except Exception as e:
        logger.error("groq_fallback_failed", error=str(e))
        return {
            "answer": f"All inference backends failed. Groq error: {str(e)}",
            "provider": "none",
            "latency": 0,
        }
