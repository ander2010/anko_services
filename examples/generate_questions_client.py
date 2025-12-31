from __future__ import annotations
import uuid
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import requests

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.append(str(EXAMPLES_DIR))
if str(EXAMPLES_DIR.parent) not in sys.path:
    sys.path.append(str(EXAMPLES_DIR.parent))

from process_request_client import derive_job_id  # type: ignore

try:
    from examples.util.env import load_env
    from examples.util.net import normalize_base_url, build_ws_url
except ImportError:  # pragma: no cover
    from util.env import load_env
    from util.net import normalize_base_url, build_ws_url


def get_local_args() -> Namespace:
    """Provide locally configured arguments instead of consuming CLI input."""
    load_env()
    return Namespace(
        base_url=os.getenv("PROCESS_REQUEST_BASE_URL", "http://localhost:8080"),
        doc_id="test",
        query_text=["Barca"],
        tags=None,
        quantity=3,
        difficulty="medium",
        question_format="true_false",
        top_k=None,
        min_importance=None,
        job_id=None,
    )


def trigger_generate_question(args: Namespace) -> tuple[int, dict]:
    """Send the generate_question request and capture response details locally."""
    process = "generate_question"
    job_id = derive_job_id(
        args.doc_id,
        process,
        args.job_id,
        theme=getattr(args, "theme", None),
        quantity=args.quantity,
        question_format=args.question_format,
        tags=args.tags,
        query_text=args.query_text,
    )
    job_id = str(uuid.uuid4())
    print(f"Job ID: {job_id}")
    query_text = None
    if args.query_text:
        query_text = args.query_text if len(args.query_text) > 1 else args.query_text[0]

    payload = {
        "job_id": job_id,
        "doc_id": args.doc_id,
        "process": process,
        "tags": args.tags,
        "query_text": query_text,
        "top_k": args.top_k,
        "min_importance": args.min_importance,
        "quantity_question": args.quantity,
        "difficulty": args.difficulty,
        "question_format": args.question_format,
        "options": {},
    }

    url = f"{normalize_base_url(args.base_url)}/process-request"
    response = requests.post(url, json=payload, timeout=120)

    result: dict = {
        "request_url": url,
        "status_code": response.status_code,
        "request_payload": payload,
    }

    try:
        response_json = response.json()
        result["response_json"] = response_json
    except Exception:
        response_json = None
        result["response_text"] = response.text

    if response.ok:
        job_id_from_response = response_json.get("job_id") if isinstance(response_json, dict) else None
        ws_job_id = job_id_from_response or job_id
        ws_url = build_ws_url(args.base_url, ws_job_id)
        result["ws_url"] = ws_url

    return (0 if response.ok else 1), result


def main() -> int:
    args = get_local_args()
    exit_code, result = trigger_generate_question(args)
    print(json.dumps(result, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
