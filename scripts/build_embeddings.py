#!/usr/bin/env python3
from __future__ import annotations

"""
Build precomputed embeddings and TF-IDF matrix for the catalog.

Produces:
  data/embeddings.npy  — (N, 384) float32 matrix of sentence embeddings
  data/tfidf.pkl       — pickled (TfidfVectorizer, sparse matrix) tuple

Run after scripts/scrape.py has produced data/catalog.json.
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
TFIDF_PATH = DATA_DIR / "tfidf.pkl"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


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

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    logger.info(f"Loaded {len(catalog)} catalog items")

    corpus = build_corpus(catalog)

    # --- Semantic embeddings ---
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    logger.info("Encoding catalog items...")
    embeddings = model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # Normalize for cosine similarity (so dot product = cosine sim)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    embeddings = embeddings / norms

    np.save(EMBEDDINGS_PATH, embeddings)
    logger.info(f"Saved embeddings: {embeddings.shape} to {EMBEDDINGS_PATH}")

    # --- TF-IDF ---
    logger.info("Building TF-IDF matrix...")
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
