from __future__ import annotations

import json
import os
from typing import Any, Dict

from redis import Redis as SyncRedis
from redis.asyncio import Redis as AsyncRedis

from pipeline.utils.logging_config import get_logger
from pipeline.workflow.utils.persistence import save_notification_async

logger = get_logger(__name__)

PROGRESS_REDIS_URL = os.getenv("PROGRESS_REDIS_URL", "redis://localhost:6379/2")
PROGRESS_DB_URL = os.getenv("DB_URL", "hope/vector_store.db")
_progress_client: AsyncRedis | None = None


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

    client = SyncRedis.from_url(PROGRESS_REDIS_URL, decode_responses=True)
    key = f"job:{job_id}"
    client.hset(key, mapping={k: str(v) for k, v in payload.items() if v is not None})
    client.publish(f"progress:{job_id}", json.dumps(payload))


def flashcard_redis_key(job_id: str) -> str:
    return f"flashcards:cards:{job_id}"


async def get_progress_client() -> AsyncRedis:
    """Return a shared asyncio Redis client for progress tracking."""
    global _progress_client
    if _progress_client is None:
        _progress_client = AsyncRedis.from_url(PROGRESS_REDIS_URL, decode_responses=True)
    return _progress_client


async def read_progress(job_id: str) -> dict:
    client = await get_progress_client()
    raw = await client.hgetall(f"job:{job_id}")
    return raw or {}


async def set_progress(job_id: str, doc_id: str, *, progress: float | int = 0, step_progress: float | int = 0, status: str = "QUEUED", current_step: str = "pending", extra: dict | None = None) -> None:
    client = await get_progress_client()
    key = f"job:{job_id}"
    payload = {
        "doc_id": doc_id,
        "progress": progress,
        "step_progress": step_progress,
        "status": status,
        "current_step": current_step,
    }
    if extra:
        payload.update(extra)
    try:
        save_notification_async(
            PROGRESS_DB_URL,
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
    await client.hset(key, mapping={k: str(v) for k, v in payload.items() if v is not None})
    await client.publish(f"progress:{job_id}", json.dumps(payload))
