from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kick off /process-request using a local PDF.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="FastAPI base URL")
    parser.add_argument("--pdf", default="Barcelona_EN.pdf", help="Path to the PDF to process")
    parser.add_argument("--doc-id", default="barcelona-en", help="Document id to use")
    parser.add_argument("--job-id", default=None, help="Optional job id seed; deterministic if omitted")
    parser.add_argument("--skip-qa", action="store_true", help="Skip QA generation to run faster")
    return parser.parse_args()


def derive_job_id(doc_id: str, process: str, seed: str | None = None) -> str:
    """Mirror server behavior: derive a stable UID from provided seed or doc_id+process."""
    base = seed or f"{doc_id}:{process}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    process = "process_pdf"
    job_id = derive_job_id(args.doc_id, process, args.job_id)
    payload = {
        "job_id": job_id,
        "doc_id": args.doc_id,
        # Use the existing file path; ensure workers have access to this path.
        "file_path": str(pdf_path),
        "process": process,
        "options": {
            "skip_qa": args.skip_qa,
        },
    }

    url = f"{args.base_url.rstrip('/')}/process-request"
    response = requests.post(url, json=payload, timeout=60)
    print(f"POST {url} -> {response.status_code}")

    try:
        data = response.json()
        print(json.dumps(data, indent=2))
        job_id = data.get("job_id", job_id)
    except Exception:
        print(response.text)
        return 1 if not response.ok else 0

    if not response.ok:
        return 1

    ws_url = f"{args.base_url.rstrip('/').replace('http', 'ws')}/ws/progress/{job_id}"
    print("Connect to the progress Websocket (e.g., in Postman):")
    print(f"{ws_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
