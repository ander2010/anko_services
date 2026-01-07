from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from celery_app import celery_app  # type: ignore
from pipeline.utils.logging_config import get_logger
from pipeline.workflow.ingestion import validate_pdf
from pipeline.workflow.utils.progress import emit_progress  # type: ignore

logger = get_logger(__name__)


class ValidateTaskService:
    """Encapsulates PDF validation task logic."""

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        path = Path(payload.get("file_path") or payload.get("file path") or "")
        job_id = payload.get("job_id")
        doc_id = payload.get("doc_id") or path.stem

        logger.info("Validate start | job=%s doc=%s path=%s", job_id, doc_id, path)

        # Validate that the PDF is accessible locally (via shared volume or s3fs mount)
        path = validate_pdf(path)
        result = dict(payload)
        result.update({"file_path": str(path), "job_id": job_id, "doc_id": doc_id})

        # Emit a small progress update to mark validation as done
        emit_progress(job_id=job_id, doc_id=doc_id, progress=5, step_progress=100, status="VALIDATED", current_step="validate", extra={})

        logger.info("Validate done  | job=%s doc=%s path=%s", job_id, doc_id, path)

        return result


@celery_app.task(name="pipeline.validate.pdf")
def validate_pdf_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight Celery task to validate a PDF before OCR."""
    return ValidateTaskService().run(payload)
