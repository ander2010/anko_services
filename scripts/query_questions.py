from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path when running as a script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.knowledge_store import LocalKnowledgeStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect stored QA pairs for a document.")
    parser.add_argument("--doc-id", required=True, help="Document id to query")
    parser.add_argument(
        "--db-url",
        default=os.getenv("DB_URL", "hope/vector_store.db"),
        help="DB URL (Postgres or SQLite path); defaults to DB_URL env or hope/vector_store.db",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with LocalKnowledgeStore(args.db_url) as store:
        if not store.document_exists(args.doc_id):
            print(f"Document '{args.doc_id}' not found in store {args.db_url}")
            return 1

        embeddings, qa_pairs = store.load_document(args.doc_id)

    print(f"DB: {args.db_url}")
    print(f"Document: {args.doc_id}")
    print(f"Chunks: {len(embeddings)} | QA pairs: {len(qa_pairs)}")

    if not qa_pairs:
        print("No QA pairs stored.")
        return 0

    for idx, qa in enumerate(qa_pairs, start=1):
        question = qa.get("question", "")
        answer = qa.get("correct_response", "")
        metadata: Any = qa.get("metadata") or {}
        tags = metadata.get("tags") or []
        difficulty = metadata.get("difficulty")

        print(f"\n#{idx}")
        print(f"Q: {question}")
        print(f"A: {answer}")
        if tags:
            print(f"Tags: {tags}")
        if difficulty:
            print(f"Difficulty: {difficulty}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
