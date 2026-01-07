from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from celery import chain, chord
from celery.result import AsyncResult

from pipeline.workflow.llm import QAFormat
from pipeline.utils.logging_config import get_logger
from pipeline.celery_tasks.embedding import embedding_task
from pipeline.celery_tasks.llm import persist_document_batch_task, finalize_batch_pipeline_task
from pipeline.celery_tasks.ocr import ocr_batch_task
from pipeline.celery_tasks.validate import validate_pdf_task
from pipeline.celery_tasks.prepare import prepare_batches_task
from pipeline.workflow.ingestion import PdfIngestion

logger = get_logger(__name__)

DEFAULT_PIPELINE_SETTINGS: Dict[str, Any] = {
    "document_id": None,
    "job_id": None,
    "dpi": 300,
    "lang": "eng",
    "min_paragraph_chars": 40,
    "min_chunk_tokens": 40,
    "max_chunk_tokens": 220,
    "chunk_overlap": 40,
    "importance_threshold": 0.4,
    "ga_answer_length": 60,
    "ga_format": QAFormat.VARIETY.value,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "ga_workers": int(os.getenv("QA_WORKERS", 4)),
    "ocr_workers": int(os.getenv("OCR_WORKERS", 2)),
    "vector_batch_size": int(os.getenv("VECTOR_BATCH_SIZE", 32)),
    "ocr_batch_pages": int(os.getenv("OCR_BATCH_PAGES", 10)),
    "ocr_queue": os.getenv("CELERY_OCR_QUEUE", "ocr"),
}


def _merged_settings(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = dict(DEFAULT_PIPELINE_SETTINGS)
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})
    for key, value in list(merged.items()):
        if isinstance(value, Path):
            merged[key] = str(value)
    return merged


def enqueue_pipeline(file_path: str | Path, settings: Optional[Dict[str, Any]] = None, persist_local: bool = False) -> AsyncResult:
    """Submit the full pipeline to Celery as a chained workflow."""
    cfg = _merged_settings(settings)
    cfg["persist_local"] = persist_local

    path = Path(file_path)
    payload: Dict[str, Any] = {
        "file_path": str(path),
        "job_id": cfg.get("job_id"),
        "doc_id": cfg.get("document_id") or path.stem,
    }

    try:
        total_pages = PdfIngestion.count_pages(path)
    except Exception:
        total_pages = 0
    if total_pages <= 0:
        total_pages = 1

    batch_size = max(1, int(cfg.get("ocr_batch_pages", 10)))
    ranges: list[tuple[int, int]] = []
    page = 1
    while page <= total_pages:
        end = min(total_pages, page + batch_size - 1)
        ranges.append((page, end))
        page = end + 1

    ocr_queue = cfg.get("ocr_queue") or os.getenv("CELERY_OCR_QUEUE", "ocr")
    header = [
        chain(
            ocr_batch_task.s(start_page=start, end_page=end, total_pages=total_pages, batch_index=idx + 1, total_batches=len(ranges), dpi=cfg["dpi"], lang=cfg["lang"]).set(queue=ocr_queue),
            embedding_task.s(settings=cfg),
            persist_document_batch_task.s(settings=cfg),
        )
        for idx, (start, end) in enumerate(ranges)
    ]

    payload["total_pages"] = total_pages
    payload["ranges"] = ranges

    ocr_step = chord(header, finalize_batch_pipeline_task.s(payload, cfg))

    workflow = chain(
        validate_pdf_task.s(),
        prepare_batches_task.s(settings=cfg),
        ocr_step,
    )

    logger.info("Enqueuing pipeline for %s", path)
    return workflow.apply_async(args=[payload], queue=cfg.get("queue"))


def revoke_job(async_result: AsyncResult, terminate: bool = False) -> None:
    """Cancel a running Celery job."""
    try:
        async_result.revoke(terminate=terminate)
    except Exception as exc:
        logger.warning("Failed to revoke task %s: %s", async_result, exc)
