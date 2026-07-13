from __future__ import annotations

from typing import Any, Dict

from celery_app import celery_app
from pipeline.workflow.document_intelligence import DocumentIntelligenceWorkflow


@celery_app.task(name="pipeline.document_intelligence.extract")
def extract_document_intelligence_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    return DocumentIntelligenceWorkflow.extract_structured_knowledge(
        title=str(payload.get("title") or "").strip(),
        source_type=str(payload.get("source_type") or "other").strip() or "other",
        document_ids=payload.get("document_ids") or [],
        fallback_text=payload.get("fallback_text"),
    )


@celery_app.task(name="pipeline.document_intelligence.analyze_diff")
def analyze_document_diff_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    return DocumentIntelligenceWorkflow.analyze_diff(
        knowledge_source_title=str(payload.get("knowledge_source_title") or "").strip(),
        old_summary=str(payload.get("old_summary") or "").strip(),
        new_summary=str(payload.get("new_summary") or "").strip(),
    )


@celery_app.task(name="pipeline.document_intelligence.describe_sections")
def describe_document_sections_task(payload: Dict[str, Any]) -> Dict[str, str]:
    return DocumentIntelligenceWorkflow.build_section_descriptions(
        section_titles=payload.get("section_titles") or [],
        document_summary=str(payload.get("document_summary") or "").strip(),
        chunks=payload.get("chunks") or [],
        max_chars=int(payload.get("max_chars") or 150),
    )
