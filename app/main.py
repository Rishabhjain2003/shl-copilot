"""
FastAPI application — /health and /chat endpoints.

Stateless server: every call reconstructs state from messages[] payload.
No session store, no persistence layer.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import catalog
from app.models import ChatRequest, ChatResponse
from app.state import process_chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Internal timeout budget for the /chat handler
HANDLER_TIMEOUT_SECONDS = 25


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    # Load catalog data at startup — fail loudly if missing
    try:
        cat = catalog.get_catalog()
        logger.info(f"Startup: catalog loaded with {len(cat)} items")
    except FileNotFoundError as e:
        logger.error(f"STARTUP FAILURE: {e}")
        raise
    yield


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL Individual Test Solution recommendations",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for evaluation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """
    Health check — returns 200 with no expensive dependencies.
    Does not hit the LLM or re-load embeddings.
    """
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a conversation and return assessment recommendations.

    Stateless: reconstructs all state from the messages[] payload on every call.
    """
    start_time = time.time()

    try:
        # Run the blocking LLM/retrieval pipeline in a thread pool
        # to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(None, process_chat, request.messages),
            timeout=HANDLER_TIMEOUT_SECONDS,
        )

        elapsed = time.time() - start_time
        logger.info(
            f"/chat completed in {elapsed:.2f}s — "
            f"recs={len(response.recommendations)}, "
            f"eoc={response.end_of_conversation}"
        )
        return response

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.error(f"/chat timed out after {elapsed:.2f}s")
        # Deterministic fallback — never return a raw 500
        return ChatResponse(
            reply=(
                "I'm taking longer than expected to process your request. "
                "Could you try rephrasing with more specific details about "
                "the role and skills you need to assess?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"/chat error after {elapsed:.2f}s: {e}", exc_info=True)
        # Never let an unhandled error return a raw 500 with no body
        return ChatResponse(
            reply=(
                "I encountered an issue processing your request. "
                "Please try again or provide more details about what you're looking for."
            ),
            recommendations=[],
            end_of_conversation=False,
        )


@app.exception_handler(422)
async def validation_error_handler(request: Request, exc):
    """Custom handler for validation errors — return a useful message."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request format. Expected: {\"messages\": [{\"role\": \"user\", \"content\": \"...\"}]}"
        },
    )
