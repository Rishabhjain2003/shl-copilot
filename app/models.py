"""Pydantic request/response models for the /chat endpoint."""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in the conversation."""
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Request body for POST /chat."""
    messages: List[Message] = Field(..., min_length=1)


class Recommendation(BaseModel):
    """A single assessment recommendation."""
    name: str
    url: str
    test_type: str  # e.g. "K", "P,C", "A,S"


class ChatResponse(BaseModel):
    """Response body for POST /chat."""
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False
