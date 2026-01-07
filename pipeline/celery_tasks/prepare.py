from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from celery_app import celery_app  # type: ignore
from pipeline.utils.logging_config import get_logger
from pipeline.workflow.ingestion import PdfIngestion
from pipeline.workflow.utils.progress import emit_progress
from pipeline.workflow.utils.progress import PROGRESS_REDIS_URL
from redis import Redis

logger = get_logger(__name__)


def _compute_ranges(total_pages: int, batch_size: int) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    page = 1
    while page <= total_pages:
        end = min(total_pages, page + batch_size - 1)
        ranges.append((page, end))
        page = end + 1
    return ranges or [(1, total_pages or 1)]


@celery_app.task(name="pipeline.prepare.batches")
def prepare_batches_task(payload: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize payload, compute OCR ranges, and seed progress counters."""
    path = Path(payload.get("file_path") or payload.get("file path") or "")
    job_id = payload.get("job_id") or settings.get("job_id")
    doc_id = payload.get("doc_id") or settings.get("document_id") or path.stem

    total_pages = int(payload.get("total_pages") or 0)
    if total_pages <= 0:
        try:
            total_pages = PdfIngestion.count_pages(path)
        except Exception:
            total_pages = 1
    total_pages = max(1, total_pages)

    batch_size = max(1, int(settings.get("ocr_batch_pages", 10)))
    ranges = payload.get("ranges") or _compute_ranges(total_pages, batch_size)

    payload["doc_id"] = doc_id
    payload["job_id"] = job_id
    payload["total_pages"] = total_pages
    payload["ranges"] = ranges

    try:
        r = Redis.from_url(settings.get("progress_redis_url") or PROGRESS_REDIS_URL, decode_responses=True)
        units_key = f"job:{job_id}:units"
        r.hset(
            units_key,
            mapping={
                "total_pages": total_pages,
                "done_ocr": 0,
                "total_chunks": 0,
                "done_embed": 0,
                "done_persist": 0,
                "done_tag": 0,
            },
        )
        r.hset(f"job:{job_id}:progress", mapping={"progress": 0})
    except Exception:
        logger.debug("Failed to seed progress counters for job=%s", job_id, exc_info=True)

    emit_progress(job_id=job_id, doc_id=doc_id, progress=10, step_progress=0, status="PREPARED", current_step="prepare", extra={"batches": len(ranges), "total_pages": total_pages})

    logger.info("Prepared batches | job=%s doc=%s pages=%s batches=%s size=%s", job_id, doc_id, total_pages, len(ranges), batch_size)
    return payload
