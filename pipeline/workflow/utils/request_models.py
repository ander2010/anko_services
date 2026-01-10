from __future__ import annotations

import os
import uuid
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

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
    context: list[str] = Field(default_factory=list, description="List of document IDs to search for context")
    top_k: int | None = Field(None, description="Max chunks to retrieve per document")
    min_importance: float | None = Field(None, description="Minimum importance threshold for retrieved chunks")
    session_id: str | None = Field(None, description="Conversation session id used for chat history lookups")
    user_id: str | None = Field(None, description="User identifier attached to chat history entries")


class QuestionVariantsRequest(BaseModel):
    question_id: str = Field(..., description="Existing question_id to generate variants from")
    quantity: int = Field(default=10, description="Number of variant questions to generate")
    difficulty: str = Field(default="medium", description="Difficulty hint for variants")
    question_format: str = Field(default="variety", description="Output format (e.g., variety, true_false)")
    job_id: str | None = Field(default=None, description="Optional job id; derived deterministically if omitted")


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
