"""
LLM integration — supports Groq, Gemini Flash, and OpenAI as providers.

Uses dynamic failover at query time:
  1. Tries Groq (llama-3.3-70b-versatile) — fastest
  2. Falls back to Gemini (gemini-2.0-flash)
  3. Falls back to OpenAI (gpt-4o-mini)

Enforces structured JSON output, fast retries on rate limits (0.5s), and timeout ceiling.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Configured clients
_groq_client = None
_gemini_client = None
_openai_client = None

# Internal timeout ceiling per LLM call
LLM_TIMEOUT_SECONDS = 12


def _init_clients():
    """Initialize all available LLM clients."""
    global _groq_client, _gemini_client, _openai_client

    groq_key = os.environ.get("GROQ_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if groq_key:
        try:
            from groq import Groq
            _groq_client = Groq(api_key=groq_key)
            logger.info("Initialized Groq client")
        except Exception as e:
            logger.warning(f"Failed to init Groq client: {e}")

    if gemini_key:
        try:
            from google import genai
            _gemini_client = genai.Client(api_key=gemini_key)
            logger.info("Initialized Gemini client")
        except Exception as e:
            logger.warning(f"Failed to init Gemini client: {e}")

    if openai_key:
        try:
            import openai
            _openai_client = openai.OpenAI(api_key=openai_key)
            logger.info("Initialized OpenAI client")
        except Exception as e:
            logger.warning(f"Failed to init OpenAI client: {e}")


def _call_groq(prompt: str) -> str:
    """Call Groq API with llama-3.3-70b-versatile (openai/gpt-oss-120b fallback)."""
    # Use openai/gpt-oss-120b as specified by user, fallback to llama-3.3-70b-versatile
    models = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"]
    last_err = None

    for model in models:
        try:
            response = _groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You must return ONLY valid JSON. No markdown, no code fences."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=4096,
                response_format={"type": "json_object"},
                timeout=LLM_TIMEOUT_SECONDS,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            error_str = str(e).lower()
            if (
                "404" in error_str
                or "not found" in error_str
                or "429" in error_str
                or "rate" in error_str
                or "quota" in error_str
            ):
                logger.warning(f"Groq model {model} failed (rate limit or not found), trying fallback...")
                continue
            raise

    raise last_err


def _call_gemini(prompt: str) -> str:
    """Call Gemini API."""
    from google.genai import types

    models = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]
    last_err = None

    for model in models:
        try:
            response = _gemini_client.models.generate_content(
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
            last_err = e
            error_str = str(e).lower()
            if "429" in error_str or "resource_exhausted" in error_str:
                logger.warning(f"Gemini model {model} rate limited, trying fallback...")
                continue
            raise

    raise last_err


def _call_openai(prompt: str) -> str:
    """Call OpenAI API."""
    response = _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON. No markdown, no code fences."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=4096,
        response_format={"type": "json_object"},
        timeout=LLM_TIMEOUT_SECONDS,
    )
    return response.choices[0].message.content.strip()


def call_llm(
    system_prompt: str,
    user_content: str,
    retry_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call the LLM using dynamic failover across available providers.
    """
    # Lazy-init clients once
    if _groq_client is None and _gemini_client is None and _openai_client is None:
        _init_clients()

    # Define the execution chain based on configured keys
    chain = []
    if _groq_client:
        chain.append(("groq", _call_groq))
    if _gemini_client:
        chain.append(("gemini", _call_gemini))
    if _openai_client:
        chain.append(("openai", _call_openai))

    if not chain:
        logger.error("No LLM clients configured!")
        return _safe_fallback()

    full_prompt_base = f"{system_prompt}\n\n---\n\n{user_content}"

    for provider, call_fn in chain:
        for attempt in range(2):  # Max 2 attempts per provider
            try:
                prompt = full_prompt_base
                if attempt > 0:
                    prompt = (
                        system_prompt
                        + "\n\nIMPORTANT: Return ONLY a valid JSON object. No markdown, no code fences."
                        + f"\n\n---\n\n{user_content}"
                    )

                start = time.time()
                text = call_fn(prompt)
                elapsed = time.time() - start
                logger.info(f"LLM [{provider}] attempt {attempt + 1} succeeded in {elapsed:.2f}s")

                result = json.loads(text)
                return result

            except json.JSONDecodeError as e:
                logger.warning(f"LLM [{provider}] attempt {attempt + 1} returned invalid JSON: {e}")
                if attempt < 1:
                    continue

            except Exception as e:
                error_str = str(e).lower()
                logger.warning(f"LLM [{provider}] attempt {attempt + 1} failed: {e}")

                # If rate limited, sleep briefly and try once more before failing over
                if "429" in error_str or "rate" in error_str or "resource_exhausted" in error_str:
                    if attempt < 1:
                        sleep_time = 0.5
                        logger.warning(f"Rate limit hit on {provider}. Sleeping {sleep_time}s before retry...")
                        time.sleep(sleep_time)
                        continue

                # For other errors, or if second attempt failed, break loop to try next provider in chain
                break

        logger.warning(f"Provider {provider} failed, falling back to next provider in chain...")

    logger.error("All LLM providers in execution chain failed — returning safe fallback")
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
