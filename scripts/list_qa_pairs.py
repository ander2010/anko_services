from __future__ import annotations

import sys
from pathlib import Path
from textwrap import shorten

# Ensure repository root is on sys.path for module imports when run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.storage import open_store

# Configuration
DB_URL = "postgresql://postgres:postgres@db:5432/pipeline"
DOC_ID = "demo-doc"
LIMIT = 10


def main() -> int:
    print(f"DB_URL: {DB_URL}")
    try:
        with open_store(DB_URL) as store:
            qa_pairs = store.load_qa_pairs(DOC_ID)
    except Exception as exc:
        print("Error accessing store:", exc)
        return 1

    print(f"QA pairs for {DOC_ID}: {len(qa_pairs)}")
    for idx, qa in enumerate(qa_pairs[:LIMIT], 1):
        question = shorten(qa.get("question", ""), width=80, placeholder="…")
        answer = shorten(qa.get("correct_response", ""), width=80, placeholder="…")
        print(f"{idx}. Q: {question}")
        print(f"   A: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
