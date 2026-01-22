from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

# Support running as a module (`python -m examples.translate_client`)
# or directly (`python examples/translate_client.py`) by handling imports both ways.
EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.append(str(EXAMPLES_DIR))
if str(EXAMPLES_DIR.parent) not in sys.path:
    sys.path.append(str(EXAMPLES_DIR.parent))

try:
    from examples.util.env import load_env
    from examples.util.net import normalize_base_url
except ImportError:  # pragma: no cover - fallback for direct execution
    from util.env import load_env
    from util.net import normalize_base_url


def _post_translate(base_url: str, payload: dict) -> dict:
    url = f"{normalize_base_url(base_url)}/translate"
    response = requests.post(url, json=payload, timeout=60)
    try:
        body = response.json()
    except Exception:
        body = {"raw": response.text}
    return {"status_code": response.status_code, "response": body, "request": payload}


def main() -> int:
    load_env()
    base_url = os.getenv("TRANSLATE_BASE_URL", os.getenv("PROCESS_REQUEST_BASE_URL", "http://localhost:8080"))
    source_language = os.getenv("TRANSLATE_SOURCE_LANGUAGE", "english")
    target_language = os.getenv("TRANSLATE_TARGET_LANGUAGE", "spanish")

    single_string_payload = {
        "source_language": source_language,
        "target_language": target_language,
        "data": ["Hello world, this is a single string."],
    }

    list_payload = {
        "source_language": source_language,
        "target_language": target_language,
        "data": ["Good morning", 42, True, {"note": "leave me alone"}, None],
    }

    dict_payload = {
        "source_language": source_language,
        "target_language": target_language,
        "data": {
            "title": "Translate me",
            "count": 3,
            "active": False,
            "details": {"note": "nested objects are not translated"},
            "items": ["First item", "Second item", 100,{"data": ["Good morning", 42, True, {"note": "leave me alone"}, None]}],
        },
    }

    for label, payload in (
        ("single_string", single_string_payload),
        ("list_any", list_payload),
        ("dict_any", dict_payload),
    ):
        result = _post_translate(base_url, payload)
        print(f"\n=== {label} ===")
        print(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
