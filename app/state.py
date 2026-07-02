"""
Conversation state machine — explicit constraint extraction and action decision.

This is the core agent-design decision: an explicit branch, not implicit
LLM judgment buried in a single mega-prompt.

Flow:
  1. Extract/update constraint state from full messages[]
  2. Decide next action (CLARIFY/RETRIEVE_AND_RECOMMEND/REFINE/COMPARE/REFUSE)
  3. Execute action via the appropriate prompt + retrieval pipeline
"""
from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from app import catalog, llm, prompts, security
from app.models import ChatResponse, Message, Recommendation
from app.retrieval import RetrievalConstraints, retrieve

logger = logging.getLogger(__name__)

# Maximum combined turns (user + assistant) before forcing recommendation
MAX_TURNS = 8
# Force recommendation if turn count reaches this threshold
FORCE_RECOMMEND_TURN = 6


class Action(str, Enum):
    CLARIFY = "CLARIFY"
    RETRIEVE_AND_RECOMMEND = "RETRIEVE_AND_RECOMMEND"
    REFINE = "REFINE"
    COMPARE = "COMPARE"
    REFUSE = "REFUSE"


def extract_constraints(messages: List[Message]) -> Dict[str, Any]:
    """
    Use the LLM to extract structured constraints from the full conversation.

    Returns a dict matching the ConversationState schema from prompts.py.
    """
    # Format conversation for the LLM
    conv_text = ""
    for msg in messages:
        role_label = "User" if msg.role == "user" else "Assistant"
        conv_text += f"{role_label}: {msg.content}\n\n"

    result = llm.call_llm(
        system_prompt=prompts.SYSTEM_CONSTRAINT_EXTRACTION,
        user_content=conv_text,
    )

    return result


def decide_action(
    constraints: Dict[str, Any],
    messages: List[Message],
    refused: bool = False,
) -> Action:
    """
    Decide the next action based on extracted constraints and conversation state.

    This is an explicit branch — not implicit LLM judgment.
    """
    turn_count = len(messages)

    # 1. REFUSE — pre-filter already caught injection or off-topic
    if refused:
        return Action.REFUSE

    # 2. COMPARE — user explicitly asked to compare named assessments
    if constraints.get("user_wants_comparison"):
        return Action.COMPARE

    # 3. REFINE — prior recommendations exist and user wants changes
    if constraints.get("prior_recommendations_exist"):
        wants_add = constraints.get("user_wants_to_add", [])
        wants_remove = constraints.get("user_wants_to_remove", [])
        if wants_add or wants_remove:
            return Action.REFINE

        # User confirmed the shortlist
        if constraints.get("user_confirmed_shortlist"):
            return Action.RETRIEVE_AND_RECOMMEND

        # User asked a follow-up question (compare, etc.)
        # Default to RETRIEVE_AND_RECOMMEND to re-present the list
        return Action.RETRIEVE_AND_RECOMMEND

    # 4. Check if we have enough signal to recommend
    has_role = bool(constraints.get("role"))
    has_skills = bool(constraints.get("skills"))
    has_test_types = bool(constraints.get("test_types_wanted"))
    has_intent = bool(constraints.get("intent_summary", "").strip())

    enough_signal = has_role or has_skills or has_test_types

    # 5. Force recommendation near turn cap
    if turn_count >= FORCE_RECOMMEND_TURN:
        logger.info(f"Turn count {turn_count} >= {FORCE_RECOMMEND_TURN}, forcing recommendation")
        return Action.RETRIEVE_AND_RECOMMEND

    # 6. CLARIFY if insufficient signal
    if not enough_signal:
        return Action.CLARIFY

    # 7. RETRIEVE_AND_RECOMMEND
    return Action.RETRIEVE_AND_RECOMMEND


def build_retrieval_constraints(constraints: Dict[str, Any]) -> RetrievalConstraints:
    """Convert extracted constraint dict into RetrievalConstraints for retrieval."""
    return RetrievalConstraints(
        test_types_wanted=constraints.get("test_types_wanted", []),
        test_types_excluded=constraints.get("test_types_excluded", []),
        job_levels=constraints.get("job_levels", []),
        languages=constraints.get("languages", []),
        max_duration_minutes=constraints.get("max_duration_minutes"),
        named_assessments=constraints.get("named_assessments", []),
    )


def format_candidates_for_llm(candidates) -> str:
    """Format the retrieval candidates as context for the LLM."""
    lines = []
    for i, cand in enumerate(candidates, 1):
        item = cand.item
        lines.append(f"Candidate {i}:")
        lines.append(f"  Name: {item['name']}")
        lines.append(f"  URL: {item['url']}")
        lines.append(f"  Test Type: {item['test_type']}")
        lines.append(f"  Keys: {', '.join(item.get('keys', []))}")
        lines.append(f"  Duration: {item.get('duration', 'Not specified')}")
        lines.append(f"  Languages: {', '.join(item.get('languages', [])[:5])}")
        if len(item.get('languages', [])) > 5:
            lines.append(f"    (+{len(item['languages']) - 5} more)")
        lines.append(f"  Job Levels: {', '.join(item.get('job_levels', []))}")
        lines.append(f"  Adaptive: {'Yes' if item.get('adaptive') else 'No'}")
        lines.append(f"  Description: {item.get('description', '')[:200]}")
        lines.append("")
    return "\n".join(lines)


def execute_action(
    action: Action,
    constraints: Dict[str, Any],
    messages: List[Message],
    refuse_reason: str = "",
) -> ChatResponse:
    """Execute the decided action and return a ChatResponse."""

    if action == Action.REFUSE:
        result = llm.call_llm(
            system_prompt=prompts.SYSTEM_REFUSE,
            user_content=f"Refused because: {refuse_reason}\n\nUser message: {messages[-1].content}",
        )
        return ChatResponse(
            reply=result.get("reply", "I can only help with SHL assessment selection. How can I assist you with that?"),
            recommendations=[],
            end_of_conversation=False,
        )

    if action == Action.CLARIFY:
        constraint_summary = json.dumps(constraints, indent=2, default=str)
        result = llm.call_llm(
            system_prompt=prompts.SYSTEM_CLARIFY,
            user_content=f"Current constraints:\n{constraint_summary}\n\nConversation so far:\n" +
                         "\n".join(f"{m.role}: {m.content}" for m in messages),
        )
        return ChatResponse(
            reply=result.get("reply", "Could you tell me more about the role you're hiring for?"),
            recommendations=[],
            end_of_conversation=False,
        )

    if action == Action.COMPARE:
        # Find named assessments to compare
        named = constraints.get("named_assessments", [])
        compare_items = []
        for name in named:
            for item in catalog.get_catalog():
                if name.lower() in item["name"].lower():
                    compare_items.append(item)
                    break

        if len(compare_items) < 2:
            # Not enough items to compare — fall back to recommend
            action = Action.RETRIEVE_AND_RECOMMEND
        else:
            items_text = ""
            for item in compare_items:
                items_text += f"\nAssessment: {item['name']}\n"
                items_text += f"  URL: {item['url']}\n"
                items_text += f"  Test Type: {item['test_type']}\n"
                items_text += f"  Keys: {', '.join(item.get('keys', []))}\n"
                items_text += f"  Duration: {item.get('duration', 'Not specified')}\n"
                items_text += f"  Languages: {', '.join(item.get('languages', []))}\n"
                items_text += f"  Job Levels: {', '.join(item.get('job_levels', []))}\n"
                items_text += f"  Adaptive: {'Yes' if item.get('adaptive') else 'No'}\n"
                items_text += f"  Description: {item.get('description', '')}\n"

            result = llm.call_llm(
                system_prompt=prompts.SYSTEM_COMPARE,
                user_content=f"User question: {messages[-1].content}\n\nAssessments to compare:\n{items_text}",
            )
            return ChatResponse(
                reply=result.get("reply", "Here's a comparison of the assessments you asked about."),
                recommendations=[],
                end_of_conversation=False,
            )

    # RETRIEVE_AND_RECOMMEND or REFINE — both run retrieval
    ret_constraints = build_retrieval_constraints(constraints)
    intent = constraints.get("intent_summary", "")

    # Also add skills and role to the query for better retrieval
    query_parts = [intent]
    if constraints.get("role"):
        query_parts.append(constraints["role"])
    if constraints.get("skills"):
        query_parts.extend(constraints["skills"])
    query = " ".join(filter(None, query_parts))

    if not query.strip():
        query = messages[-1].content  # fallback to latest user message

    # Run hybrid retrieval
    candidates = retrieve(query, ret_constraints)

    # Format candidates for LLM context
    candidates_text = format_candidates_for_llm(candidates)

    # Build the user content for the recommend prompt
    constraint_summary = json.dumps(constraints, indent=2, default=str)
    user_content = (
        f"User constraints:\n{constraint_summary}\n\n"
        f"Latest user message: {messages[-1].content}\n\n"
        f"Candidate assessments (select 1-10 from these ONLY):\n{candidates_text}"
    )

    # Handle REFINE specifically
    if action == Action.REFINE:
        wants_add = constraints.get("user_wants_to_add", [])
        wants_remove = constraints.get("user_wants_to_remove", [])
        user_content += f"\n\nUser wants to ADD: {wants_add}"
        user_content += f"\nUser wants to REMOVE: {wants_remove}"
        user_content += "\nUpdate the shortlist accordingly."

    # Check if user confirmed
    is_confirmed = constraints.get("user_confirmed_shortlist", False)

    result = llm.call_llm(
        system_prompt=prompts.SYSTEM_RECOMMEND,
        user_content=user_content,
    )

    # Build recommendations from LLM response
    raw_recs = result.get("recommendations", [])
    valid_recs = []
    url_set = catalog.get_url_set()

    for rec in raw_recs:
        url = rec.get("url", "")
        # Post-hoc hallucination filter: only keep URLs that exist in catalog
        if url in url_set:
            valid_recs.append(Recommendation(
                name=rec.get("name", ""),
                url=url,
                test_type=rec.get("test_type", ""),
            ))
        else:
            logger.warning(f"Filtered hallucinated URL: {url}")

    # Cap at 10
    valid_recs = valid_recs[:10]

    # Determine end_of_conversation
    end_of_conv = result.get("end_of_conversation", False)
    if is_confirmed and valid_recs:
        end_of_conv = True

    return ChatResponse(
        reply=result.get("reply", "Here are my assessment recommendations."),
        recommendations=valid_recs,
        end_of_conversation=end_of_conv,
    )


def process_chat(messages: List[Message]) -> ChatResponse:
    """
    Main entry point: process a chat request and return a response.

    This reconstructs state from the full message history on every call
    (stateless server-side, stateful reasoning).
    """
    if not messages:
        return ChatResponse(
            reply="Hello! I can help you find the right SHL assessments for your hiring needs. What role are you looking to fill?",
            recommendations=[],
            end_of_conversation=False,
        )

    # Step 1: Security pre-filter on the latest user message
    latest_user_msg = ""
    for msg in reversed(messages):
        if msg.role == "user":
            latest_user_msg = msg.content
            break

    refused = False
    refuse_reason = ""
    should_refuse_flag, reason = security.should_refuse(latest_user_msg)
    if should_refuse_flag:
        refused = True
        refuse_reason = reason
        logger.info(f"Security pre-filter triggered: {reason}")

    # Step 2: Extract constraints from full conversation
    constraints = extract_constraints(messages)

    # Step 3: Decide action
    action = decide_action(constraints, messages, refused=refused)
    logger.info(f"Decided action: {action.value}")

    # Step 4: Execute action
    response = execute_action(action, constraints, messages, refuse_reason)

    return response
