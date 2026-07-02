#!/usr/bin/env python3
from __future__ import annotations
"""
Download the SHL product catalog JSON and produce a cleaned data/catalog.json.

The original assignment assumes a scrapable HTML table at shl.com/products/product-catalog/.
That URL now redirects to a marketing page. SHL provides a pre-built JSON at the endpoint
below, which contains all 377 Individual Test Solutions with pre-extracted fields.

This script:
  1. Downloads the catalog from the SHL endpoint
  2. Validates schema completeness
  3. Derives test_type_codes from the 'keys' field
  4. Parses duration_minutes from the 'duration' string
  5. Runs data quality checks and logs a report
  6. Writes cleaned data/catalog.json
"""

import json
import logging
import re
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"

# Test type legend (from the original catalog page)
KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def download_catalog() -> list[dict]:
    """Download the raw catalog JSON from SHL's endpoint."""
    logger.info(f"Downloading catalog from {CATALOG_URL}")
    headers = {
        "User-Agent": "SHL-Assessment-Recommender/1.0 (catalog-ingestion)"
    }
    resp = requests.get(CATALOG_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    # The JSON has some invalid control characters; parse with strict=False
    data = json.loads(resp.text, strict=False)
    logger.info(f"Downloaded {len(data)} items")
    return data


def parse_duration_minutes(duration_str: str) -> int | None:
    """Extract numeric minutes from strings like '30 minutes', 'Untimed', etc."""
    if not duration_str:
        return None
    match = re.search(r"(\d+)\s*minute", duration_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def derive_test_type_codes(keys: list[str]) -> list[str]:
    """Map the 'keys' field to single-letter test type codes."""
    codes = []
    for key in keys:
        code = KEY_TO_CODE.get(key)
        if code:
            codes.append(code)
    return sorted(set(codes))


def clean_catalog(raw_data: list[dict]) -> list[dict]:
    """Clean and enrich catalog items."""
    cleaned = []
    for item in raw_data:
        cleaned_item = {
            "entity_id": item.get("entity_id"),
            "name": (item.get("name") or "").strip(),
            "url": (item.get("link") or "").strip(),
            "description": (item.get("description") or "").strip(),
            "job_levels": item.get("job_levels") or [],
            "languages": item.get("languages") or [],
            "duration": (item.get("duration") or "").strip(),
            "duration_minutes": parse_duration_minutes(item.get("duration") or ""),
            "remote": item.get("remote", "").lower() == "yes",
            "adaptive": item.get("adaptive", "").lower() == "yes",
            "keys": item.get("keys") or [],
            "test_type_codes": derive_test_type_codes(item.get("keys") or []),
            "test_type": ",".join(derive_test_type_codes(item.get("keys") or [])),
        }
        cleaned.append(cleaned_item)
    return cleaned


def run_quality_checks(data: list[dict]) -> dict:
    """Run data quality checks and return a report."""
    report = {
        "total_items": len(data),
        "missing_name": sum(1 for d in data if not d["name"]),
        "missing_description": sum(1 for d in data if not d["description"]),
        "missing_url": sum(1 for d in data if not d["url"]),
        "missing_test_type_codes": sum(1 for d in data if not d["test_type_codes"]),
        "missing_job_levels": sum(1 for d in data if not d["job_levels"]),
        "missing_languages": sum(1 for d in data if not d["languages"]),
        "missing_duration": sum(1 for d in data if not d["duration"]),
        "duplicate_urls": 0,
        "duplicate_entity_ids": 0,
    }

    # Check for duplicate URLs
    urls = [d["url"] for d in data if d["url"]]
    report["duplicate_urls"] = len(urls) - len(set(urls))

    # Check for duplicate entity_ids
    eids = [d["entity_id"] for d in data if d["entity_id"]]
    report["duplicate_entity_ids"] = len(eids) - len(set(eids))

    # Test type distribution
    from collections import Counter
    type_counter = Counter()
    for d in data:
        for code in d["test_type_codes"]:
            type_counter[code] += 1
    report["test_type_distribution"] = dict(type_counter.most_common())

    # Job level distribution
    jl_counter = Counter()
    for d in data:
        for jl in d["job_levels"]:
            jl_counter[jl] += 1
    report["job_level_distribution"] = dict(jl_counter.most_common())

    return report


def main():
    # Download
    raw_data = download_catalog()

    # Clean
    cleaned = clean_catalog(raw_data)

    # Quality checks
    report = run_quality_checks(cleaned)

    logger.info("=== DATA QUALITY REPORT ===")
    logger.info(f"Total items: {report['total_items']}")
    logger.info(f"Missing name: {report['missing_name']}")
    logger.info(f"Missing description: {report['missing_description']}")
    logger.info(f"Missing URL: {report['missing_url']}")
    logger.info(f"Missing test_type_codes: {report['missing_test_type_codes']}")
    logger.info(f"Missing job_levels: {report['missing_job_levels']}")
    logger.info(f"Missing languages: {report['missing_languages']}")
    logger.info(f"Missing duration: {report['missing_duration']}")
    logger.info(f"Duplicate URLs: {report['duplicate_urls']}")
    logger.info(f"Duplicate entity_ids: {report['duplicate_entity_ids']}")
    logger.info(f"Test type distribution: {report['test_type_distribution']}")
    logger.info(f"Job level distribution: {report['job_level_distribution']}")

    # Fail loudly on critical issues
    if report["missing_name"] > 0 or report["missing_url"] > 0:
        logger.error("CRITICAL: Items missing name or URL!")
        sys.exit(1)
    if report["duplicate_urls"] > 0:
        logger.error("CRITICAL: Duplicate URLs found!")
        sys.exit(1)

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    logger.info(f"Wrote {len(cleaned)} items to {OUTPUT_PATH}")
    logger.info("Data ingestion complete.")


if __name__ == "__main__":
    main()
