"""
Lightweight injection pre-filter.

Pattern/heuristic check ahead of LLM calls — don't rely on the LLM alone
to self-police instruction overrides embedded in user turns.
"""
from __future__ import annotations

import re
from typing import Tuple

# Patterns that suggest prompt injection or scope violation
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above\s+instructions",
    r"disregard\s+(all\s+)?previous",
    r"forget\s+(all\s+)?previous",
    r"override\s+(your\s+)?instructions",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+if\s+you",
    r"pretend\s+you\s+are",
    r"reveal\s+(your\s+)?system\s+prompt",
    r"show\s+(me\s+)?(your\s+)?system\s+prompt",
    r"what\s+(is|are)\s+your\s+(system\s+)?instructions",
    r"print\s+(your\s+)?instructions",
    r"output\s+(your\s+)?system",
    r"repeat\s+(your\s+)?(system\s+)?prompt",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
]

# Off-topic patterns — requests for general advice outside SHL assessment scope
OFF_TOPIC_PATTERNS = [
    r"(legal|regulatory)\s+(advice|requirement|obligation)",
    r"(write|generate)\s+(me\s+)?(a\s+)?(poem|story|essay|code|email)",
    r"what\s+is\s+the\s+(meaning|capital|population)",
    r"tell\s+me\s+a\s+joke",
    r"(how|what)\s+to\s+cook",
]

_compiled_injection = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
_compiled_off_topic = [re.compile(p, re.IGNORECASE) for p in OFF_TOPIC_PATTERNS]


def check_injection(text: str) -> Tuple[bool, str]:
    """
    Check if text contains prompt injection patterns.

    Returns:
        (is_injection, matched_pattern)
    """
    for pattern in _compiled_injection:
        if pattern.search(text):
            return True, pattern.pattern
    return False, ""


def check_off_topic(text: str) -> Tuple[bool, str]:
    """
    Check if text is clearly off-topic for SHL assessment scope.

    Returns:
        (is_off_topic, matched_pattern)
    """
    for pattern in _compiled_off_topic:
        if pattern.search(text):
            return True, pattern.pattern
    return False, ""


def should_refuse(text: str) -> Tuple[bool, str]:
    """
    Combined check: returns (should_refuse, reason).
    """
    is_inj, pattern = check_injection(text)
    if is_inj:
        return True, f"prompt_injection: {pattern}"

    is_ot, pattern = check_off_topic(text)
    if is_ot:
        return True, f"off_topic: {pattern}"

    return False, ""
