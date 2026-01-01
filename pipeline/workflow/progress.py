from __future__ import annotations

import json
import os
from typing import Any, Dict

from redis import Redis

from pipeline.utils.logging_config import get_logger
from pipeline.workflow.utils.persistence import save_notification_async

logger = get_logger(__name__)

PROGRESS_REDIS_URL = os.getenv("PROGRESS_REDIS_URL", "redis://localhost:6379/2")
PROGRESS_DB_URL = os.getenv("DB_URL", "hope/vector_store.db")


def emit_progress(job_id: str | None, doc_id: str | None, status: str, current_step: str, progress: float | int = 0, step_progress: float | int = 0, extra: Dict[str, Any] | None = None, db_path: str | None = None) -> None:
    """Push a progress snapshot to Redis hash + pubsub channel and persist to notifications."""
    if not job_id:
        return

    payload: Dict[str, Any] = {
        "doc_id": doc_id,
        "progress": progress,
        "step_progress": step_progress,
        "status": status,
        "current_step": current_step,
    }

    if extra:
        payload.update(extra)

    # Persist durable snapshot keyed by job_id so reconnects can read notifications from storage.
    try:
        target_db = db_path or PROGRESS_DB_URL
        save_notification_async(
            target_db,
            job_id,
            {
                "status": status,
                "current_step": current_step,
                "progress": progress,
                "step_progress": step_progress,
                "doc_id": doc_id,
                **(extra or {}),
            },
        )
    except Exception:
        logger.warning("Failed to queue notification | job=%s", job_id, exc_info=True)

    client = Redis.from_url(PROGRESS_REDIS_URL, decode_responses=True)
    key = f"job:{job_id}"
    client.hset(key, mapping={k: str(v) for k, v in payload.items() if v is not None})
    client.publish(f"progress:{job_id}", json.dumps(payload))
