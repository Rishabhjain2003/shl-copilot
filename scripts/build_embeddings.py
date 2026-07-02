#!/usr/bin/env python3
from __future__ import annotations

"""
Build precomputed embeddings using Gemini embedding API and TF-IDF matrix for the catalog.

Produces:
  data/embeddings.npy  — (N, 3072) float32 matrix of sentence embeddings
  data/tfidf.pkl       — pickled (TfidfVectorizer, sparse matrix) tuple

Requires GEMINI_API_KEY environment variable.
"""

import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
TFIDF_PATH = DATA_DIR / "tfidf.pkl"

EMBEDDING_MODEL = "models/gemini-embedding-2"


def build_corpus(catalog: list[dict]) -> list[str]:
    """Build a text corpus from catalog items for embedding and TF-IDF.
    
    Combines name, description, keys, and job_levels into a single string
    per item to maximize retrieval signal.
    """
    corpus = []
    for item in catalog:
        parts = [
            item.get("name", ""),
            item.get("description", ""),
            " ".join(item.get("keys", [])),
            " ".join(item.get("job_levels", [])),
        ]
        corpus.append(" ".join(parts))
    return corpus


def main():
    if not CATALOG_PATH.exists():
        logger.error(f"Catalog not found at {CATALOG_PATH}. Run scripts/scrape.py first.")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY or GOOGLE_API_KEY environment variable must be set to build embeddings.")
        sys.exit(1)

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    logger.info(f"Loaded {len(catalog)} catalog items")

    corpus = build_corpus(catalog)

    # --- Semantic embeddings via Gemini API ---
    logger.info(f"Connecting to Gemini API to build embeddings using: {EMBEDDING_MODEL}")
    from google import genai
    client = genai.Client(api_key=api_key)

    embeddings_list = []
    logger.info("Calling Gemini embedding API (batch processing in groups of 50)...")
    batch_size = 50
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i : i + batch_size]
        logger.info(f"Encoding items {i} to {min(i + batch_size, len(corpus))}...")
        
        # Delay to avoid hitting rate limits
        time.sleep(1.5)

        # Retry logic on rate limits
        for attempt in range(5):
            try:
                resp = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                )
                # Extract vector values
                for emb in resp.embeddings:
                    embeddings_list.append(emb.values)
                break
            except Exception as e:
                if attempt < 4 and ("429" in str(e) or "rate" in str(e).lower()):
                    sleep_time = 16  # Sleep 16s to reset the per-minute quota
                    logger.warning(f"Rate limited. Waiting {sleep_time}s before retry...")
                    time.sleep(sleep_time)
                else:
                    raise e

    embeddings = np.array(embeddings_list, dtype=np.float32)

    # Normalize for cosine similarity (so dot product = cosine sim)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    embeddings = embeddings / norms

    np.save(EMBEDDINGS_PATH, embeddings)
    logger.info(f"Saved embeddings: {embeddings.shape} to {EMBEDDINGS_PATH}")

    # --- TF-IDF ---
    logger.info("Building TF-IDF matrix...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    with open(TFIDF_PATH, "wb") as f:
        pickle.dump((vectorizer, tfidf_matrix), f)

    logger.info(f"Saved TF-IDF: {tfidf_matrix.shape} to {TFIDF_PATH}")
    logger.info("Embedding build complete.")


if __name__ == "__main__":
    main()
