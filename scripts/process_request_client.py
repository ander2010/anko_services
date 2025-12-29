import json
import os
import sys
import uuid
from pathlib import Path

import requests

# Support running as a module (`python -m scripts.process_request_client`)
# or directly (`python scripts/process_request_client.py`) by handling imports both ways.
try:
    from scripts.util.env import load_env
    from scripts.util.net import normalize_base_url
except ImportError:  # pragma: no cover - fallback for direct execution
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from util.env import load_env
    from util.net import normalize_base_url


def derive_job_id(
    doc_id: str,
    process: str,
    seed: str | None = None,
    *,
    theme: str | None = None,
    question_format: str | None = None,
    tags: list[str] | str | None = None,
    query_text: list[str] | str | None = None,
) -> str:
    """Mirror server behavior: deterministic job id based on key request params."""
    if seed:
        return seed

    if isinstance(tags, str):
        tags_list = [tags]
    else:
        try:
            tags_list = [str(tag) for tag in (tags or []) if str(tag)]
        except TypeError:
            tags_list = []

    if isinstance(query_text, str):
        query_list = [query_text]
    else:
        try:
            query_list = [str(text).strip() for text in (query_text or []) if str(text).strip()]
        except TypeError:
            query_list = []

    seed_data = {
        "process": process,
        "doc_id": doc_id,
        "theme": theme,
        "question_format": question_format,
        "tags": sorted(tags_list),
        "query_text": sorted(query_list),
    }
    payload = json.dumps(seed_data, sort_keys=True, separators=(",", ":"))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, payload))


def env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


def main() -> int:
    load_env()

    base_url = os.getenv("PROCESS_REQUEST_BASE_URL", "http://localhost:8080")
    pdf_path = os.getenv("PROCESS_REQUEST_FILE_PATH", "documents/barcelona-en.pdf")
    doc_id = os.getenv("PROCESS_REQUEST_DOC_ID", "test")
    job_seed = os.getenv("PROCESS_REQUEST_JOB_ID")
    skip_qa = env_bool("PROCESS_REQUEST_SKIP_QA", True)

    process = "process_pdf"
    job_id = derive_job_id(doc_id, process, job_seed)
    payload = {
        "job_id": job_id,
        "doc_id": doc_id,
        "file_path": pdf_path,
        "process": process,
        "options": {
            "skip_qa": skip_qa,
        },
    }

    url = f"{normalize_base_url(base_url)}/process-request"
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

    ws_url = f"{normalize_base_url(base_url).replace('http', 'ws', 1)}/ws/progress/{job_id}"
    print("Connect to the progress Websocket (e.g., in Postman):")
    print(f"{ws_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
