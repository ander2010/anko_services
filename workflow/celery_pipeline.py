from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from celery import chain
from celery.result import AsyncResult

from celery_app import celery_app  # type: ignore
from pipeline.llm import QAFormat
from pipeline.logging_config import get_logger
from pipeline.task.embedding import embedding_task
from pipeline.task.llm import llm_task, tag_chunks_task
from pipeline.task.ocr import ocr_pdf_task
from pipeline.task.validate import validate_pdf_task

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
    "use_llm": False,
    "use_llm_qa": False,
    "ga_workers": int(os.getenv("QA_WORKERS", 4)),
    "ocr_workers": int(os.getenv("OCR_WORKERS", 2)),
    "vector_batch_size": int(os.getenv("VECTOR_BATCH_SIZE", 32)),
}


def _merged_settings(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = dict(DEFAULT_PIPELINE_SETTINGS)
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})
    # Celery JSON serializer cannot handle Path objects.
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

    workflow = chain(
        validate_pdf_task.s(),
        ocr_pdf_task.s(dpi=cfg["dpi"], lang=cfg["lang"]),
        embedding_task.s(settings=cfg),
        llm_task.s(settings=cfg),
        tag_chunks_task.s(settings=cfg),
    )

    logger.info("Enqueuing pipeline for %s", path)
    return workflow.apply_async(args=[payload], queue=cfg.get("queue"))


def revoke_job(async_result: AsyncResult, terminate: bool = False) -> None:
    """Cancel a running Celery job."""
    try:
        async_result.revoke(terminate=terminate)
    except Exception as exc:
        logger.warning("Failed to revoke task %s: %s", async_result, exc)
