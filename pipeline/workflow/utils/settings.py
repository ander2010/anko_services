from __future__ import annotations

from typing import Any, Dict


def normalize_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common alias keys and enforce defaults."""
    if settings is None:
        return {}
    normalized = dict(settings)

    # doc_id/documents
    if normalized.get("doc_id") and not normalized.get("document_id"):
        normalized["document_id"] = normalized.get("doc_id")
    # worker aliases
    if normalized.get("qa_workers") and not normalized.get("ga_workers"):
        normalized["ga_workers"] = normalized.get("qa_workers")
    # job id alias
    if normalized.get("jobid") and not normalized.get("job_id"):
        normalized["job_id"] = normalized.get("jobid")
    return normalized
