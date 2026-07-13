from __future__ import annotations

import os
from typing import Any, Sequence

from pipeline.utils.logging_config import get_logger
from pipeline.workflow.knowledge_store import LocalKnowledgeStore
from pipeline.workflow.llm import LLMDocumentIntelligenceClient

logger = get_logger(__name__)

DB_URL = os.getenv("DB_URL", "hope/vector_store.db")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_SOURCE_CHARS = max(2000, int(os.getenv("DOCUMENT_INTELLIGENCE_MAX_SOURCE_CHARS", "12000")))
MAX_SECTION_CONTEXT_CHARS = max(2000, int(os.getenv("SECTION_DESCRIPTION_MAX_CONTEXT_CHARS", "8000")))
SECTION_DESCRIPTION_MAX_CHARS = max(60, int(os.getenv("SECTION_DESCRIPTION_MAX_CHARS", "150")))


class DocumentIntelligenceWorkflow:
    @staticmethod
    def _fallback_extraction(error_message: str) -> dict[str, Any]:
        return {
            "summary": f"Extraction failed: {error_message}",
            "topics": [],
            "procedures": [],
            "knowledge_nodes": [],
            "knowledge_relationships": [],
        }

    @staticmethod
    def _fallback_diff(error_message: str) -> dict[str, Any]:
        return {
            "key_changes": [],
            "impact_level": "medium",
            "affected_topics": [],
            "affected_procedures": [],
            "summary": f"Automated analysis unavailable: {error_message}",
            "recommendations": ["Review document changes manually and update training."],
        }

    @staticmethod
    def _collect_source_text(*, document_ids: Sequence[str | int], fallback_text: str | None = None) -> str:
        doc_ids = [str(doc_id).strip() for doc_id in (document_ids or []) if str(doc_id).strip()]
        blocks: list[str] = []
        total_chars = 0
        if doc_ids:
            with LocalKnowledgeStore(DB_URL) as store:
                for doc_id in doc_ids:
                    embeddings, _ = store.load_document(doc_id)
                    seen: set[str] = set()
                    doc_lines: list[str] = []
                    for emb in embeddings:
                        text = " ".join(str(emb.text or "").split())
                        if not text:
                            continue
                        key = text.casefold()
                        if key in seen:
                            continue
                        seen.add(key)
                        doc_lines.append(text)
                        projected = total_chars + sum(len(line) + 1 for line in doc_lines)
                        if projected >= MAX_SOURCE_CHARS:
                            break
                    if doc_lines:
                        block = f"Document {doc_id}\n" + "\n".join(doc_lines)
                        blocks.append(block)
                        total_chars += len(block)
                        if total_chars >= MAX_SOURCE_CHARS:
                            break
        source_text = "\n\n".join(blocks).strip()
        if source_text:
            return source_text[:MAX_SOURCE_CHARS]
        return str(fallback_text or "").strip()[:MAX_SOURCE_CHARS]

    @classmethod
    def extract_structured_knowledge(
        cls,
        *,
        title: str,
        source_type: str,
        document_ids: Sequence[str | int],
        fallback_text: str | None = None,
    ) -> dict[str, Any]:
        source_text = cls._collect_source_text(document_ids=document_ids, fallback_text=fallback_text)
        if not source_text:
            raise ValueError("No processed document content available for extraction.")
        client = LLMDocumentIntelligenceClient(model=OPENAI_MODEL)
        try:
            if not client.is_active:
                raise RuntimeError("OPENAI_API_KEY is not configured.")
            return client.extract_structured_knowledge(title=title, source_type=source_type, text=source_text)
        except Exception as exc:
            logger.warning("Structured document extraction failed | title=%s error=%s", title, exc, exc_info=True)
            return cls._fallback_extraction(str(exc))

    @classmethod
    def analyze_diff(
        cls,
        *,
        knowledge_source_title: str,
        old_summary: str,
        new_summary: str,
    ) -> dict[str, Any]:
        client = LLMDocumentIntelligenceClient(model=OPENAI_MODEL)
        try:
            if not client.is_active:
                raise RuntimeError("OPENAI_API_KEY is not configured.")
            return client.analyze_diff(
                knowledge_source_title=knowledge_source_title,
                old_summary=old_summary,
                new_summary=new_summary,
            )
        except Exception as exc:
            logger.warning("Document diff analysis failed | title=%s error=%s", knowledge_source_title, exc, exc_info=True)
            return cls._fallback_diff(str(exc))

    @classmethod
    def build_section_descriptions(
        cls,
        *,
        section_titles: Sequence[str],
        document_summary: str,
        chunks: Sequence[dict[str, Any]],
        max_chars: int = SECTION_DESCRIPTION_MAX_CHARS,
    ) -> dict[str, str]:
        cleaned_titles = [str(title).strip() for title in section_titles if str(title).strip()]
        if not cleaned_titles:
            return {}
        context_parts: list[str] = []
        total_chars = 0
        for chunk in chunks or []:
            text = " ".join(str((chunk or {}).get("text") or "").split())
            if not text:
                continue
            context_parts.append(text)
            total_chars += len(text) + 1
            if total_chars >= MAX_SECTION_CONTEXT_CHARS:
                break
        context_text = "\n".join(context_parts)[:MAX_SECTION_CONTEXT_CHARS]

        client = LLMDocumentIntelligenceClient(model=OPENAI_MODEL)
        if client.is_active and (document_summary or context_text):
            try:
                generated = client.describe_sections(
                    document_summary=document_summary,
                    section_titles=cleaned_titles,
                    context_text=context_text,
                    max_chars=max_chars,
                )
            except Exception:
                logger.warning("Section description generation failed", exc_info=True)
                generated = {}
        else:
            generated = {}

        descriptions: dict[str, str] = {}
        for title in cleaned_titles:
            description = " ".join(str(generated.get(title) or "").split())
            if not description:
                description = f"Key document section about {title}."
            if len(description) > max_chars:
                description = description[:max_chars].rstrip()
            descriptions[title] = description
        return descriptions
