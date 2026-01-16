#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import sys

from sqlalchemy import delete, func, select, update

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.db.models import Flashcard, FlashcardReview
from pipeline.db.session import create_engine_and_session
from pipeline.workflow.utils.progress import flashcard_redis_key


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    job_id = os.getenv("FLASHCARD_JOB_ID") or "aec44240-6222-5f34-8b32-1ac59aee42c4"
    clear_redis = os.getenv("FLASHCARD_CLEAR_REDIS", "1") == "1"
    dry_run = os.getenv("FLASHCARD_DRY_RUN", "0") == "1"
    env_file = os.getenv("FLASHCARD_ENV_FILE", ".env")

    _load_env_file(Path(env_file))

    if clear_redis and not job_id:
        raise SystemExit("--clear-redis requires --job-id")

    db_url = os.getenv("DB_URL")
    if not db_url:
        raise SystemExit("Missing DB_URL; set it in the environment or .env.")

    _, SessionLocal = create_engine_and_session(db_url)
    filters = []
    if job_id:
        filters.append(Flashcard.job_id == job_id)

    now = dt.datetime.now(dt.timezone.utc)
    reset_values = {
        "kind": "new",
        "status": "learning",
        "learning_step_index": 0,
        "repetition": 0,
        "interval_days": 0,
        "ease_factor": 2.5,
        "due_at": now,
        "first_seen_at": None,
    }

    with SessionLocal() as session:
        count = session.execute(
            select(func.count()).select_from(Flashcard).where(*filters)
        ).scalar_one()
        if dry_run:
            print(f"Matched {count} flashcards.")
            return 0

        session.execute(update(Flashcard).where(*filters).values(**reset_values))
        session.commit()

    if clear_redis and job_id:
        from redis import Redis

        redis_url = os.getenv("PROGRESS_REDIS_URL", "redis://localhost:6379/2")
        client = Redis.from_url(redis_url, decode_responses=True)
        client.delete(flashcard_redis_key(job_id))

    print(f"Reset {count} flashcards.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
