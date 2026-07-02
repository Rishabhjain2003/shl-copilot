"""
Catalog singleton — loads catalog.json + embeddings.npy + tfidf.pkl at startup.

Fails loudly if any data file is missing or stale.
Provides the URL validation set for post-hoc hallucination filtering.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Module-level singletons (loaded once at import / startup)
_catalog: Optional[List[dict]] = None
_embeddings: Optional[np.ndarray] = None
_tfidf_vectorizer = None
_tfidf_matrix = None
_url_set: Optional[Set[str]] = None
_url_to_item: Optional[Dict[str, dict]] = None


def _load():
    """Load all data files. Called once at startup."""
    global _catalog, _embeddings, _tfidf_vectorizer, _tfidf_matrix, _url_set, _url_to_item

    catalog_path = DATA_DIR / "catalog.json"
    embeddings_path = DATA_DIR / "embeddings.npy"
    tfidf_path = DATA_DIR / "tfidf.pkl"

    # Fail loudly if files are missing
    for path in [catalog_path, embeddings_path, tfidf_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required data file missing: {path}. "
                f"Run 'python scripts/scrape.py' then 'python scripts/build_embeddings.py'."
            )

    with open(catalog_path, "r", encoding="utf-8") as f:
        _catalog = json.load(f)

    _embeddings = np.load(embeddings_path)

    with open(tfidf_path, "rb") as f:
        _tfidf_vectorizer, _tfidf_matrix = pickle.load(f)

    _url_set = {item["url"] for item in _catalog}
    _url_to_item = {item["url"]: item for item in _catalog}

    logger.info(
        f"Catalog loaded: {len(_catalog)} items, "
        f"embeddings {_embeddings.shape}, "
        f"TF-IDF {_tfidf_matrix.shape}"
    )


def get_catalog() -> List[dict]:
    """Return the full catalog list."""
    if _catalog is None:
        _load()
    return _catalog


def get_embeddings() -> np.ndarray:
    """Return the precomputed embeddings matrix."""
    if _embeddings is None:
        _load()
    return _embeddings


def get_tfidf() -> Tuple:
    """Return (vectorizer, tfidf_matrix)."""
    if _tfidf_vectorizer is None:
        _load()
    return _tfidf_vectorizer, _tfidf_matrix


def get_url_set() -> Set[str]:
    """Return the set of all valid catalog URLs."""
    if _url_set is None:
        _load()
    return _url_set


def get_item_by_url(url: str) -> Optional[dict]:
    """Look up a catalog item by its URL."""
    if _url_to_item is None:
        _load()
    return _url_to_item.get(url)


def is_valid_url(url: str) -> bool:
    """Check if a URL is a literal member of the catalog."""
    return url in get_url_set()
