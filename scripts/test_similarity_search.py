from __future__ import annotations

import json
import sys

import requests

# Configuration
BASE_URL = "http://localhost:8000"
DOC_ID = "barcelona-en"
QUERY_TEXT = ["rfid"]  # e.g., ["rfid", "tracking tags"]
TAGS = None  # or ["tag1", "tag2"]
TOP_K = 10
MIN_IMPORTANCE = None  # e.g., 0.4


def main() -> int:
    payload = {
        "document_id": DOC_ID,
        "query_text": QUERY_TEXT,
        "tags": TAGS,
        "top_k": TOP_K,
        "min_importance": MIN_IMPORTANCE,
    }
    url = f"{BASE_URL.rstrip('/')}/similarity-search"
    response = requests.post(url, json=payload, timeout=120)
    print(f"POST {url} -> {response.status_code}")
    try:
        data = response.json()
        print(json.dumps(data, indent=2))
    except Exception:
        print(response.text)
        return 1 if not response.ok else 0
    return 0 if response.ok else 1


if __name__ == "__main__":
    sys.exit(main())
