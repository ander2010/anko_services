from __future__ import annotations

import os
import uuid
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, Field


class ProcessOptions(BaseModel):
    ocr_language: str | None = Field(None, description="Language code for OCR (e.g., eng)")
    chunk_size: int | None = Field(None, description="Maximum chunk token budget")
    embedding_model: str | None = Field(None, description="Embedding model name")
    importance_threshold: float | None = Field(None, description="Relevance/importance floor")
    ga_format: str | None = Field(None, description="QA format")
    max_chunks: int | None = Field(None, description="Limit number of chunks retained")


class ProcessType(str, Enum):
    PROCESS_PDF = "process_pdf"
    GENERATE_QUESTION = "generate_question"


class Language(str, Enum):
    ENGLISH = "english"
    SPANISH = "spanish"


class GenerationKind(str, Enum):
    ASSESSMENT = "assessment"
    FLASHCARDS = "flashcards"


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_string_list(values: Any) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    try:
        raw_values = values or []
    except TypeError:
        raw_values = []
    for raw in raw_values:
        text = _clean_string(raw)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


class SourceBundle(BaseModel):
    collection_id: int | str | None = Field(None, description="Generic process container id")
    document_ids: list[int | str] = Field(default_factory=list, description="Documents included in the generation source")
    section_ids: list[int | str] = Field(default_factory=list, description="Optional section ids included in the generation source")
    tag_group_ids: list[int | str] = Field(default_factory=list, description="Optional tag-group ids included in the generation source")
    tags: list[str] = Field(default_factory=list, description="Semantic tags used to focus generation")
    title_hints: list[str] = Field(default_factory=list, description="Optional human-readable hints used to derive concise titles")

    def normalized_collection_id(self) -> str | None:
        return _clean_string(self.collection_id)

    def normalized_document_ids(self) -> list[str]:
        return _clean_string_list(self.document_ids)

    def normalized_section_ids(self) -> list[str]:
        return _clean_string_list(self.section_ids)

    def normalized_tag_group_ids(self) -> list[str]:
        return _clean_string_list(self.tag_group_ids)

    def normalized_tags(self) -> list[str]:
        return _clean_string_list(self.tags)

    def normalized_title_hints(self) -> list[str]:
        return _clean_string_list(self.title_hints)

    def has_any_source(self) -> bool:
        return any(
            (
                self.normalized_collection_id(),
                self.normalized_document_ids(),
                self.normalized_section_ids(),
                self.normalized_tag_group_ids(),
                self.normalized_tags(),
                self.normalized_title_hints(),
            )
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "collection_id": self.normalized_collection_id(),
            "document_ids": self.normalized_document_ids(),
            "section_ids": self.normalized_section_ids(),
            "tag_group_ids": self.normalized_tag_group_ids(),
            "tags": self.normalized_tags(),
            "title_hints": self.normalized_title_hints(),
        }


class ProcessRequest(BaseModel):
    job_id: str | None = Field(None, description="optional job id to use for the task")
    doc_id: int = Field(..., description="External document id (integer, must already exist)")
    file_path: str | None = Field(None, description="Path to the uploaded PDF accessible to workers (required for process_pdf)")
    process: ProcessType = Field(default=ProcessType.PROCESS_PDF, description="Type of processing to run (process_pdf | generate_question)")
    options: ProcessOptions = Field(default_factory=ProcessOptions)
    metadata: dict = Field(default_factory=dict)
    theme: str | None = Field(None, description="Optional theme for question generation")
    quantity_question: int | None = Field(None, description="Number of questions to generate")
    difficulty: str | None = Field(None, description="Desired difficulty for generated questions")
    question_format: str | None = Field(None, description="Question format for generation")
    tags: list[str] | None = Field(None, description="Tags to filter chunk retrieval")
    query_text: list[str] | str | None = Field(None, description="Optional query text(s) to pick relevant chunks")
    top_k: int | None = Field(None, description="Maximum number of chunks to retrieve for similarity search")
    min_importance: float | None = Field(None, description="Minimum importance score for similarity search")


class SimilaritySearchRequest(BaseModel):
    query_text: list[str]
    document_id: str | None = None
    tags: list[str] | None = None
    min_importance: float | None = None
    top_k: int | None = None
    embedding_model: str | None = None
    db_path: str | None = None


class AskRequest(BaseModel):
    question: str = Field(..., description="User question to answer")
    context: list[int] = Field(default_factory=list, description="List of document IDs to search for context")
    top_k: int | None = Field(None, description="Max chunks to retrieve per document")
    min_importance: float | None = Field(None, description="Minimum importance threshold for retrieved chunks")
    session_id: str | None = Field(None, description="Conversation session id used for chat history lookups")
    user_id: str | None = Field(None, description="User identifier attached to chat history entries")


class TranslateRequest(BaseModel):
    source_language: Language = Field(..., description="Language to translate from")
    target_language: Language = Field(..., description="Language to translate to")
    data: list[Any] | dict[str, Any] = Field(..., description="List of any items or dict of values to translate when they are strings")


class QuestionVariantsRequest(BaseModel):
    question_id: str = Field(..., description="Existing question_id to generate variants from")
    quantity: int = Field(default=10, description="Number of variant questions to generate")
    difficulty: str = Field(default="medium", description="Difficulty hint for variants")
    question_format: str = Field(default="variety", description="Output format (e.g., variety, true_false)")
    job_id: str | None = Field(default=None, description="Optional job id; derived deterministically if omitted")


class AssessmentGenerationRequest(BaseModel):
    job_id: str | None = Field(None, description="Optional job id to use for the task")
    battery_id: int | None = Field(None, description="Optional API battery id linked to the generated output")
    title: str | None = Field(None, description="Optional explicit title; derived during processing when omitted")
    quantity_question: int | None = Field(None, description="Target number of questions to generate")
    difficulty: str | None = Field(None, description="Desired difficulty for generated questions")
    question_format: str | None = Field(None, description="Question format for generation")
    theme: str | None = Field(None, description="Optional theme for question generation")
    query_text: list[str] | str | None = Field(None, description="Optional retrieval query text(s) used to pick relevant chunks")
    top_k: int | None = Field(None, description="Maximum number of chunks to retrieve for similarity search")
    min_importance: float | None = Field(None, description="Minimum importance score for similarity search")
    source_bundle: SourceBundle = Field(default_factory=SourceBundle)
    metadata: dict[str, Any] = Field(default_factory=dict)
    prompt_version: str | None = Field(None, description="Optional prompt version override used for job derivation")

    def to_worker_payload(self, *, title: str | None = None) -> dict[str, Any]:
        source_payload = self.source_bundle.to_payload()
        return {
            "job_id": _clean_string(self.job_id),
            "battery_id": self.battery_id,
            "title": _clean_string(title) or _clean_string(self.title),
            "quantity_question": self.quantity_question,
            "difficulty": _clean_string(self.difficulty),
            "question_format": _clean_string(self.question_format),
            "theme": _clean_string(self.theme),
            "query_text": self.query_text,
            "top_k": self.top_k,
            "min_importance": self.min_importance,
            "prompt_version": _clean_string(self.prompt_version),
            "metadata": dict(self.metadata or {}),
            "source_bundle": source_payload,
            "collection_id": source_payload["collection_id"],
            "document_ids": source_payload["document_ids"],
            "section_ids": source_payload["section_ids"],
            "tag_group_ids": source_payload["tag_group_ids"],
            "tags": source_payload["tags"],
            "title_hints": source_payload["title_hints"],
        }


class FlashcardGenerationRequest(BaseModel):
    job_id: str | None = Field(None, description="Optional job id to use for the task")
    user_id: str | None = Field(None, description="End user id associated with the generated flashcards")
    deck_id: int | None = Field(None, description="Optional API deck id linked to the generated output")
    title: str | None = Field(None, description="Optional explicit title; derived during processing when omitted")
    quantity: int = Field(default=10, description="Target number of flashcards to generate")
    difficulty: str | None = Field(None, description="Desired difficulty for generated flashcards")
    notes: str | None = Field(None, description="Optional notes attached to generated cards")
    token: str | None = Field(None, description="Optional websocket/session token associated with the job")
    prompt_version: str | None = Field(None, description="Optional prompt version override used for job derivation")
    source_bundle: SourceBundle = Field(default_factory=SourceBundle)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_worker_payload(self, *, title: str | None = None) -> dict[str, Any]:
        source_payload = self.source_bundle.to_payload()
        return {
            "job_id": _clean_string(self.job_id),
            "user_id": _clean_string(self.user_id),
            "deck_id": self.deck_id,
            "title": _clean_string(title) or _clean_string(self.title),
            "quantity": max(0, int(self.quantity or 0)),
            "difficulty": _clean_string(self.difficulty),
            "notes": _clean_string(self.notes),
            "token": _clean_string(self.token),
            "prompt_version": _clean_string(self.prompt_version),
            "metadata": dict(self.metadata or {}),
            "source_bundle": source_payload,
            "collection_id": source_payload["collection_id"],
            "document_ids": source_payload["document_ids"],
            "section_ids": source_payload["section_ids"],
            "tag_group_ids": source_payload["tag_group_ids"],
            "tags": source_payload["tags"],
            "title_hints": source_payload["title_hints"],
        }


class DocumentIntelligenceExtractRequest(BaseModel):
    title: str = Field(..., description="Human-readable knowledge source title")
    source_type: str = Field(default="other", description="Knowledge source type")
    document_ids: list[int | str] = Field(default_factory=list, description="Processed documents to read from Hope storage")
    fallback_text: str | None = Field(None, description="Optional raw text fallback when chunks are unavailable")


class DocumentIntelligenceDiffRequest(BaseModel):
    knowledge_source_title: str = Field(..., description="Human-readable knowledge source title")
    old_summary: str = Field(default="", description="Previous version summary")
    new_summary: str = Field(default="", description="New version summary")


def default_settings(db_url: str, *, override: dict | None = None) -> SimpleNamespace:
    db_path = db_url if db_url.startswith(("postgres://", "postgresql://")) else Path(db_url)
    settings = SimpleNamespace(
        document_id=None,
        job_id=str(uuid.uuid4()),
        dpi=300,
        lang="eng",
        min_paragraph_chars=40,
        min_chunk_tokens=int(os.getenv("MIN_CHUNK_TOKENS", 40)),
        max_chunk_tokens=int(os.getenv("MAX_CHUNK_TOKENS", 320)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 24)),
        importance_threshold=0.4,
        qa_answer_length=60,
        qa_format="variety",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        ga_workers=int(os.getenv("QA_WORKERS", 4)),
        ocr_workers=int(os.getenv("OCR_WORKERS", 4)),
        vector_batch_size=int(os.getenv("VECTOR_BATCH_SIZE", 32)),
        max_chunks=None,
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        persist_local=True,
        db_path=db_path,
        allow_overwrite=True,
    )
    if override:
        for key, val in override.items():
            setattr(settings, key, val)
    return settings
