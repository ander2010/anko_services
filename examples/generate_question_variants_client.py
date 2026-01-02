"""
Example client to request variant questions for an existing question_id.

Usage:
    export QUESTION_ID=<id>
    python examples/generate_question_variants_client.py [question_id]  # arg wins over env
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import requests


def get_config() -> dict[str, Any]:
    question_id = (sys.argv[1] if len(sys.argv) > 1 else os.getenv("QUESTION_ID", "f31af8ca-8313-4247-b285-8c57f42f7e62")).strip()
    if not question_id:
        sys.exit("QUESTION_ID is required (env or first arg).")
    return {
        "question_id": question_id,
        "base_url": os.getenv("PROCESS_REQUEST_BASE_URL", "http://localhost:8080").rstrip("/"),
        "quantity": int(os.getenv("QUESTION_VARIANTS_QTY", "10")),
        "difficulty": os.getenv("QUESTION_VARIANTS_DIFFICULTY", "medium"),
        "question_format": os.getenv("QUESTION_VARIANTS_FORMAT", "variety"),
        "job_id": os.getenv("QUESTION_VARIANTS_JOB_ID"),
    }


def main() -> None:
    cfg = get_config()
    url = f"{cfg['base_url']}/questions/{cfg['question_id']}/variants"
    payload: dict[str, Any] = {
        "question_id": cfg["question_id"],
        "quantity": cfg["quantity"],
        "difficulty": cfg["difficulty"],
        "question_format": cfg["question_format"],
    }
    if cfg["job_id"]:
        payload["job_id"] = cfg["job_id"]

    try:
        resp = requests.post(url, json=payload, timeout=30)
    except Exception as exc:  # pragma: no cover - network
        sys.exit(f"Request failed: {exc}")

    print(f"Status: {resp.status_code}")
    try:
        data = resp.json()
        print(json.dumps(data, indent=2))
        job_id = data.get("job_id")
        base_ws = cfg["base_url"].replace("http", "ws", 1)
        if job_id:
            ws_url = f"{base_ws}/ws/progress/{job_id}"
            print(f"\nProgress websocket: {ws_url}")
    except Exception:
        print(resp.text)


if __name__ == "__main__":
    main()
