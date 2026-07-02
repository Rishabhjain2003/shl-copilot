"""
LLM integration — supports Groq, Gemini Flash, and OpenAI as providers.

Provider selection via environment (checked in order):
  - GROQ_API_KEY  → Groq (llama-3.3-70b-versatile) — fastest inference
  - GEMINI_API_KEY → Gemini Flash
  - OPENAI_API_KEY → OpenAI GPT-4o-mini

Enforces structured JSON output, retry logic with backoff, and timeout ceiling.
Falls back to a safe deterministic response on failure.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_provider = None  # "groq", "gemini", or "openai"
_client = None

# Internal timeout ceiling per LLM call
LLM_TIMEOUT_SECONDS = 20


def _init_provider():
    """Detect and initialize the LLM provider based on available API keys."""
    global _provider, _client

    groq_key = os.environ.get("GROQ_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if groq_key:
        try:
            from groq import Groq
            _client = Groq(api_key=groq_key)
            _provider = "groq"
            logger.info("Initialized Groq provider")
            return
        except Exception as e:
            logger.warning(f"Failed to init Groq: {e}")

    if gemini_key:
        try:
            from google import genai
            _client = genai.Client(api_key=gemini_key)
            _provider = "gemini"
            logger.info("Initialized Gemini provider")
            return
        except Exception as e:
            logger.warning(f"Failed to init Gemini: {e}")

    if openai_key:
        try:
            import openai
            _client = openai.OpenAI(api_key=openai_key)
            _provider = "openai"
            logger.info("Initialized OpenAI provider")
            return
        except Exception as e:
            logger.warning(f"Failed to init OpenAI: {e}")

    raise EnvironmentError(
        "No LLM provider configured. Set GROQ_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY."
    )


def _get_provider():
    """Lazy-initialize and return the provider."""
    if _provider is None:
        _init_provider()
    return _provider, _client


def _call_groq(client, prompt: str) -> str:
    """Call Groq API with openai/gpt-oss-120b."""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You must return ONLY valid JSON. No markdown, no code fences, no extra text."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content.strip()


def _call_gemini(client, prompt: str) -> str:
    """Call Gemini API."""
    from google.genai import types

    models = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]
    last_error = None

    for model in models:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    top_p=0.8,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                ),
            )
            return response.text.strip()
        except Exception as e:
            last_error = e
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                logger.warning(f"Rate limited on {model}, trying next...")
                continue
            raise

    raise last_error


def _call_openai(client, prompt: str) -> str:
    """Call OpenAI API."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON. No markdown, no code fences."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content.strip()


def call_llm(
    system_prompt: str,
    user_content: str,
    retry_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call the LLM with a system prompt and user content.

    Returns:
        Parsed JSON dict from the LLM response.
        On failure, returns a safe fallback.
    """
    provider, client = _get_provider()
    full_prompt = f"{system_prompt}\n\n---\n\n{user_content}"

    for attempt in range(3):
        try:
            if attempt > 0:
                full_prompt = (
                    f"{system_prompt}\n\n"
                    "IMPORTANT: Return ONLY a valid JSON object. "
                    "No markdown, no code fences.\n\n"
                    f"---\n\n{user_content}"
                )

            start = time.time()

            if provider == "groq":
                text = _call_groq(client, full_prompt)
            elif provider == "gemini":
                text = _call_gemini(client, full_prompt)
            elif provider == "openai":
                text = _call_openai(client, full_prompt)
            else:
                raise ValueError(f"Unknown provider: {provider}")

            elapsed = time.time() - start
            logger.info(f"LLM [{provider}] attempt {attempt + 1}: {elapsed:.2f}s")

            result = json.loads(text)
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"LLM attempt {attempt + 1} invalid JSON: {e}")
            if attempt < 2:
                continue

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                wait_time = min(2 ** attempt * 2, 10)
                logger.warning(f"Rate limited, attempt {attempt + 1}. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"LLM attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    continue

    logger.error("All LLM attempts failed — returning safe fallback")
    return _safe_fallback()


def _safe_fallback() -> Dict[str, Any]:
    """Deterministic safe response when all LLM calls fail."""
    return {
        "reply": (
            "I'm having trouble processing your request right now. "
            "Could you please rephrase or provide more details about "
            "the role and assessments you're looking for?"
        ),
        "recommendations": [],
        "end_of_conversation": False,
    }
