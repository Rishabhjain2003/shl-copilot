"""
Hybrid retrieval engine: hard filter + TF-IDF lexical + semantic embedding fusion.

Design:
  1. Hard filter: exclude items that violate explicit constraints
  2. Lexical (TF-IDF cosine): exact keyword matching
  3. Semantic (embedding cosine): conceptual matching
  4. Fuse with weighted sum: score = alpha * semantic + (1-alpha) * lexical
  5. Return top-K candidates for LLM ranking

Trade-off: TF-IDF instead of BM25. For 377 documents the quality difference
is negligible, and sklearn provides TF-IDF out of the box. With more time,
BM25 with term saturation would be a strict improvement.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from app import catalog

logger = logging.getLogger(__name__)

# Fusion weight: higher alpha = more semantic, lower = more lexical.
# 0.6 chosen because most user queries are conceptual ("backend engineer")
# rather than keyword-exact ("Core Java Advanced Level New").
ALPHA = 0.6
TOP_K = 20

# Lazy-loaded Gemini client
_genai_client = None


def _get_genai_client():
    """Lazy-initialize the Gemini genai client."""
    global _genai_client
    if _genai_client is None:
        import os
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client


@dataclass
class RetrievalConstraints:
    """Constraints extracted from conversation for hard filtering."""
    test_types_wanted: List[str] = field(default_factory=list)    # e.g. ["K", "P"]
    test_types_excluded: List[str] = field(default_factory=list)  # e.g. ["P"]
    job_levels: List[str] = field(default_factory=list)           # e.g. ["Graduate"]
    languages: List[str] = field(default_factory=list)            # e.g. ["Spanish"]
    max_duration_minutes: Optional[int] = None
    named_assessments: List[str] = field(default_factory=list)    # exact names to boost


@dataclass
class ScoredCandidate:
    """A catalog item with its retrieval score."""
    item: dict
    score: float
    lexical_score: float
    semantic_score: float


def hard_filter(
    items: List[dict],
    constraints: RetrievalConstraints,
) -> List[dict]:
    """Apply constraint-based exclusions to reduce the candidate pool."""
    filtered = []
    for item in items:
        item_codes = set(item.get("test_type_codes", []))

        # Exclude items with unwanted test types
        if constraints.test_types_excluded:
            if item_codes & set(constraints.test_types_excluded):
                continue

        # If specific test types wanted, item must have at least one match
        # (soft filter — don't exclude if no test type preference stated)
        if constraints.test_types_wanted:
            if not (item_codes & set(constraints.test_types_wanted)):
                continue

        # Job level filter (soft — only apply if user specified levels)
        if constraints.job_levels:
            item_levels = set(item.get("job_levels", []))
            # Allow items with no job levels (they're generic)
            if item_levels and not (item_levels & set(constraints.job_levels)):
                continue

        # Language filter
        if constraints.languages:
            item_langs = set(item.get("languages", []))
            # Allow items with no languages listed (they may be universal)
            if item_langs:
                # Check if any requested language is a substring match
                lang_match = False
                for req_lang in constraints.languages:
                    req_lower = req_lang.lower()
                    for item_lang in item_langs:
                        if req_lower in item_lang.lower():
                            lang_match = True
                            break
                    if lang_match:
                        break
                if not lang_match:
                    continue

        # Duration filter
        if constraints.max_duration_minutes is not None:
            item_dur = item.get("duration_minutes")
            if item_dur is not None and item_dur > constraints.max_duration_minutes:
                continue

        filtered.append(item)

    logger.info(f"Hard filter: {len(items)} -> {len(filtered)} candidates")
    return filtered


def lexical_score(query: str, filtered_indices: List[int]) -> np.ndarray:
    """Compute TF-IDF cosine similarity for filtered items."""
    vectorizer, tfidf_matrix = catalog.get_tfidf()
    query_vec = vectorizer.transform([query])

    # Score only filtered items
    if filtered_indices:
        sub_matrix = tfidf_matrix[filtered_indices]
    else:
        sub_matrix = tfidf_matrix

    scores = sklearn_cosine(query_vec, sub_matrix).flatten()
    return scores


def semantic_score(query: str, filtered_indices: List[int]) -> np.ndarray:
    """Compute embedding cosine similarity for filtered items using Gemini API."""
    client = _get_genai_client()
    embeddings = catalog.get_embeddings()

    # Call Gemini API to embed the query
    resp = client.models.embed_content(
        model="models/gemini-embedding-2",
        contents=query,
    )
    query_values = resp.embeddings[0].values
    query_emb = np.array(query_values, dtype=np.float32).reshape(1, -1)

    # Normalize
    norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
    if norm > 0:
        query_emb = query_emb / norm

    if filtered_indices:
        sub_embs = embeddings[filtered_indices]
    else:
        sub_embs = embeddings

    # Dot product = cosine similarity (both are already normalized)
    scores = (sub_embs @ query_emb.T).flatten()
    return scores


def retrieve(
    query: str,
    constraints: RetrievalConstraints,
    top_k: int = TOP_K,
) -> List[ScoredCandidate]:
    """
    Run hybrid retrieval: hard filter -> lexical + semantic -> fuse -> top-K.

    Args:
        query: accumulated conversation intent string
        constraints: hard constraints from conversation
        top_k: number of candidates to return

    Returns:
        List of ScoredCandidate, sorted by fused score descending
    """
    all_items = catalog.get_catalog()

    # Step 1: Hard filter
    filtered = hard_filter(all_items, constraints)

    if not filtered:
        logger.warning("Hard filter eliminated all items — falling back to full catalog")
        filtered = all_items

    # Map filtered items to their indices in the full catalog
    url_to_idx = {item["url"]: i for i, item in enumerate(all_items)}
    filtered_indices = [url_to_idx[item["url"]] for item in filtered]

    # Step 2: Lexical scoring
    lex_scores = lexical_score(query, filtered_indices)

    # Step 3: Semantic scoring
    sem_scores = semantic_score(query, filtered_indices)

    # Step 4: Boost named assessments
    name_boost = np.zeros(len(filtered))
    if constraints.named_assessments:
        for i, item in enumerate(filtered):
            item_name_lower = item["name"].lower()
            for named in constraints.named_assessments:
                if named.lower() in item_name_lower:
                    name_boost[i] = 0.3  # significant boost for exact name mentions
                    break

    # Step 5: Fuse scores
    fused = ALPHA * sem_scores + (1 - ALPHA) * lex_scores + name_boost

    # Step 6: Top-K
    top_indices = np.argsort(fused)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append(ScoredCandidate(
            item=filtered[idx],
            score=float(fused[idx]),
            lexical_score=float(lex_scores[idx]),
            semantic_score=float(sem_scores[idx]),
        ))

    logger.info(
        f"Retrieval: query='{query[:60]}...', "
        f"filtered={len(filtered)}, returning top-{len(results)}"
    )
    return results
