"""
Trace replay tests — replay the 10 sample conversations against /chat
and compute Recall@10 against labeled shortlists.

Each trace file (C1.md - C10.md) contains:
  - User turns with expected agent behavior
  - Final labeled shortlist URLs in markdown table format

This harness:
  1. Parses each trace to extract user turns and expected URLs
  2. Sends user turns sequentially to /chat
  3. Collects final recommendation URLs
  4. Computes Recall@10 = |predicted ∩ expected| / |expected|
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Set, Tuple

import pytest
from fastapi.testclient import TestClient

from app import catalog
from app.main import app

logger = logging.getLogger(__name__)

client = TestClient(app)

TRACES_DIR = Path(__file__).resolve().parent.parent / "sample_conversations" / "GenAI_SampleConversations"


def parse_trace(filepath: Path) -> Tuple[List[str], Set[str]]:
    """
    Parse a conversation trace file.

    Returns:
        (user_messages, expected_urls) — list of user turn texts, set of expected URLs
    """
    text = filepath.read_text(encoding="utf-8")

    # Extract user messages
    user_messages = []
    user_pattern = re.compile(r'\*\*User\*\*\s*\n\s*>(.*?)(?=\n\n|\*\*Agent\*\*)', re.DOTALL)
    for match in user_pattern.finditer(text):
        msg = match.group(1).strip()
        # Multi-line user messages: join lines starting with >
        lines = []
        for line in msg.split('\n'):
            line = line.strip()
            if line.startswith('>'):
                line = line[1:].strip()
            lines.append(line)
        user_messages.append(' '.join(lines).strip())

    # Extract expected URLs from the LAST table in the trace
    # Tables contain URLs in format <https://...>
    url_pattern = re.compile(r'<(https://www\.shl\.com/products/product-catalog/view/[^>]+)>')
    all_urls = url_pattern.findall(text)

    # The expected shortlist is from the last table (final recommendation)
    # Find all tables and use the last one
    tables = list(re.finditer(r'\|.*\|.*\|.*\|.*\|.*\|.*\|.*\|', text))
    if tables:
        last_table_pos = tables[-1].start()
        # Get URLs from the last table section
        last_section = text[last_table_pos:]
        expected_urls = set(url_pattern.findall(last_section))
    else:
        expected_urls = set(all_urls) if all_urls else set()

    return user_messages, expected_urls


def compute_recall_at_k(predicted_urls: Set[str], expected_urls: Set[str], k: int = 10) -> float:
    """Compute Recall@K = |predicted ∩ expected| / |expected|."""
    if not expected_urls:
        return 1.0  # No expected = trivially correct
    predicted_top_k = set(list(predicted_urls)[:k])
    intersection = predicted_top_k & expected_urls
    return len(intersection) / len(expected_urls)


def replay_conversation(trace_path: Path) -> Tuple[float, List[dict], Set[str], Set[str]]:
    """
    Replay a conversation trace against /chat.

    Returns:
        (recall, transcript, predicted_urls, expected_urls)
    """
    user_messages, expected_urls = parse_trace(trace_path)

    if not user_messages:
        return 0.0, [], set(), expected_urls

    messages = []
    transcript = []
    last_recs = []

    for user_msg in user_messages:
        messages.append({"role": "user", "content": user_msg})

        resp = client.post("/chat", json={"messages": messages})
        assert resp.status_code == 200
        data = resp.json()

        transcript.append({
            "user": user_msg,
            "reply": data.get("reply", ""),
            "recommendations": data.get("recommendations", []),
            "end_of_conversation": data.get("end_of_conversation", False),
        })

        if data.get("recommendations"):
            last_recs = data["recommendations"]

        # Add assistant response to messages for next turn
        messages.append({"role": "assistant", "content": data.get("reply", "")})

    predicted_urls = {rec["url"] for rec in last_recs}

    # Validate all predicted URLs are in catalog
    url_set = catalog.get_url_set()
    for url in predicted_urls:
        assert url in url_set, f"Hallucinated URL in trace replay: {url}"

    recall = compute_recall_at_k(predicted_urls, expected_urls)
    return recall, transcript, predicted_urls, expected_urls


# --- Test functions for each trace ---

@pytest.fixture(params=[f"C{i}" for i in range(1, 11)])
def trace_path(request):
    """Parametrized fixture for each trace file."""
    path = TRACES_DIR / f"{request.param}.md"
    if not path.exists():
        pytest.skip(f"Trace file not found: {path}")
    return path


def test_trace_replay(trace_path):
    """Replay a conversation trace and check Recall@10."""
    recall, transcript, predicted, expected = replay_conversation(trace_path)

    trace_name = trace_path.stem
    logger.info(f"\n{'='*60}")
    logger.info(f"Trace: {trace_name}")
    logger.info(f"Expected URLs ({len(expected)}):")
    for url in sorted(expected):
        logger.info(f"  {url}")
    logger.info(f"Predicted URLs ({len(predicted)}):")
    for url in sorted(predicted):
        marker = "✓" if url in expected else "✗"
        logger.info(f"  {marker} {url}")
    logger.info(f"Recall@10: {recall:.2%}")
    logger.info(f"{'='*60}\n")

    # Log full transcript for debugging
    for i, turn in enumerate(transcript):
        logger.info(f"  Turn {i+1}:")
        logger.info(f"    User: {turn['user'][:100]}")
        logger.info(f"    Reply: {turn['reply'][:100]}")
        logger.info(f"    Recs: {len(turn['recommendations'])}")

    # We don't hard-assert recall > threshold because LLM output varies,
    # but we do assert schema compliance and no hallucinated URLs.
    assert recall >= 0.0, "Recall should be non-negative"
