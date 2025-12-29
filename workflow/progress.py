from __future__ import annotations

import json
import os
from typing import Any, Dict

from redis import Redis

PROGRESS_REDIS_URL = os.getenv("PROGRESS_REDIS_URL", "redis://localhost:6379/2")


def emit_progress(job_id: str | None, doc_id: str | None, status: str, current_step: str, progress: float | int = 0, step_progress: float | int = 0, extra: Dict[str, Any] | None = None) -> None:
    """Push a progress snapshot to Redis hash + pubsub channel."""
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

    client = Redis.from_url(PROGRESS_REDIS_URL, decode_responses=True)
    key = f"job:{job_id}"
    client.hset(key, mapping={k: str(v) for k, v in payload.items() if v is not None})
    client.publish(f"progress:{job_id}", json.dumps(payload))
