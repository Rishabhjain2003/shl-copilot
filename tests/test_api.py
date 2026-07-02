"""
API-level tests: health endpoint, schema validation, error handling.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    """GET /health returns 200 with {status: ok}."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_chat_requires_messages():
    """POST /chat with empty body returns 422."""
    resp = client.post("/chat", json={})
    assert resp.status_code == 422


def test_chat_rejects_empty_messages():
    """POST /chat with empty messages list returns 422."""
    resp = client.post("/chat", json={"messages": []})
    assert resp.status_code == 422


def test_chat_rejects_invalid_role():
    """POST /chat with invalid role returns 422."""
    resp = client.post("/chat", json={
        "messages": [{"role": "system", "content": "hello"}]
    })
    assert resp.status_code == 422


def test_chat_returns_valid_schema():
    """POST /chat with valid input returns schema-compliant response."""
    resp = client.post("/chat", json={
        "messages": [{"role": "user", "content": "I need an assessment for Java developers"}]
    })
    assert resp.status_code == 200
    data = resp.json()

    # Schema compliance
    assert "reply" in data
    assert isinstance(data["reply"], str)
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
    assert "end_of_conversation" in data
    assert isinstance(data["end_of_conversation"], bool)

    # Each recommendation must have name, url, test_type
    for rec in data["recommendations"]:
        assert "name" in rec
        assert "url" in rec
        assert "test_type" in rec


def test_chat_no_raw_500():
    """Malformed but parseable request should not return a raw 500."""
    resp = client.post("/chat", json={
        "messages": [{"role": "user", "content": ""}]
    })
    # Should return 200 with a graceful response, not 500
    assert resp.status_code == 200
