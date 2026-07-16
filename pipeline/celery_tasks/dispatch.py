from __future__ import annotations

from pathlib import Path
from typing import Any

from celery_app import celery_app


@celery_app.task(name="pipeline.prepare.dispatch_document")
def dispatch_document_pipeline_task(payload: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    file_path = payload.get("file_path") or payload.get("file path")
    if not file_path:
        raise ValueError("file_path is required for process_pdf dispatch")

    merged_settings = dict(settings or {})
    if payload.get("job_id") and not merged_settings.get("job_id"):
        merged_settings["job_id"] = payload.get("job_id")
    if payload.get("doc_id") and not merged_settings.get("document_id"):
        merged_settings["document_id"] = payload.get("doc_id")

    from pipeline.workflow.utils.celery_pipeline import enqueue_pipeline

    task = enqueue_pipeline(
        Path(file_path),
        settings=merged_settings,
        persist_local=bool(merged_settings.get("persist_local", True)),
    )
    return {
        "task_id": task.id,
        "job_id": payload.get("job_id"),
        "document_id": payload.get("doc_id"),
        "status": "queued",
    }
