from __future__ import annotations

import json
import os
import uuid
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from pipeline.knowledge_store import LocalKnowledgeStore
from pipeline.logging_config import get_logger
from pipeline.task.llm import generate_questions_task
from workflow.vectorizer import Chunkvectorizer
from workflow.celery_pipeline import enqueue_pipeline

logger = get_logger("pipeline.service")

PROGRESS_REDIS_URL = os.getenv("PROGRESS_REDIS_URL", "redis://localhost:6379/2")
progress_client: Redis | None = None


def default_settings() -> SimpleNamespace:
    db_url = os.getenv("DB_URL", "hope/vector_store.db")
    db_path = db_url if db_url.startswith(("postgres://", "postgresql://")) else Path(db_url)
    return SimpleNamespace(
        document_id=str(uuid.uuid4()),
        job_id=str(uuid.uuid4()),
        dpi=300,
        lang="eng",
        min_paragraph_chars=40,
        min_chunk_tokens=40,
        max_chunk_tokens=220,
        chunk_overlap=40,
        importance_threshold=0.4,
        qa_answer_length=60,
        qa_format="variety",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        ga_workers=int(os.getenv("QA_WORKERS", 4)),
        ocr_workers=int(os.getenv("OCR_WORKERS", 4)),
        vector_batch_size=int(os.getenv("VECTOR_BATCH_SIZE", 32)),
        max_chunks=None,
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model="gpt-4o-mini",
        persist_local=True,
        db_path=db_path,
        allow_overwrite=True,
    )


async def get_progress_client() -> Redis:
    global progress_client
    if progress_client is None:
        progress_client = Redis.from_url(PROGRESS_REDIS_URL, decode_responses=True)
    return progress_client


async def read_progress(job_id: str) -> dict:
    client = await get_progress_client()
    raw = await client.hgetall(f"job:{job_id}")
    return raw or {}


async def set_progress(job_id: str, doc_id: str, *, progress: float | int = 0, step_progress: float | int = 0, status: str = "QUEUED", current_step: str = "pending", extra: dict | None = None) -> None:
    client = await get_progress_client()
    key = f"job:{job_id}"
    payload = {
        "doc_id": doc_id,
        "progress": progress,
        "step_progress": step_progress,
        "status": status,
        "current_step": current_step,
    }
    if extra:
        payload.update(extra)
    await client.hset(key, mapping={k: str(v) for k, v in payload.items() if v is not None})
    await client.publish(f"progress:{job_id}", json.dumps(payload))


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
    doc_id: str = Field(..., description="External document id")
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


app = FastAPI(title="Pipeline Streaming Service")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


def derive_job_id(request: ProcessRequest) -> str:
    if request.job_id:
        return request.job_id

    if request.process == ProcessType.GENERATE_QUESTION:
        tags = request.tags or []
        if isinstance(tags, str):
            tags = [tags]

        query_texts_raw = request.query_text
        if isinstance(query_texts_raw, str):
            query_texts = [query_texts_raw]
        else:
            try:
                query_texts = [str(text).strip() for text in (query_texts_raw or []) if str(text).strip()]
            except TypeError:
                query_texts = []

        seed_data = {
            "process": request.process.value,
            "doc_id": request.doc_id,
            "theme": request.theme,
            "question_format": request.question_format,
            "tags": sorted(str(tag) for tag in tags if str(tag)),
            "query_text": sorted(query_texts),
        }
        seed = json.dumps(seed_data, sort_keys=True, separators=(",", ":"))
    else:
        seed = f"{request.doc_id}:{request.process.value}"

    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def average_embedding_vectors(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    first = vectors[0] or []
    if not first:
        return []
    length = len(first)
    sums = [0.0] * length
    count = 0
    for vec in vectors:
        if len(vec) != length:
            continue
        sums = [a + float(b) for a, b in zip(sums, vec)]
        count += 1
    if count == 0:
        return []
    return [val / count for val in sums]


@app.websocket("/ws/progress/{job_id}")
async def progress_ws(websocket: WebSocket, job_id: str):
    """Websocket endpoint that streams progress updates for a document."""
    await websocket.accept()
    client = await get_progress_client()
    channel = f"progress:{job_id}"
    key = f"job:{job_id}"
    pubsub = client.pubsub()
    await pubsub.subscribe(channel)
    try:
        snapshot = await client.hgetall(key)
        if snapshot:
            await websocket.send_json({"type": "snapshot", "job_id": job_id, **snapshot})
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=10.0)
            if message and message.get("data"):
                data = message["data"]
                try:
                    payload = json.loads(data)
                except (TypeError, json.JSONDecodeError):
                    payload = {"raw": data}
                payload.setdefault("job_id", job_id)
                payload.setdefault("type", "progress")
                snapshot = await client.hgetall(key)
                merged = {**snapshot, **payload}
                if "progress" not in merged and merged.get("progress_process") is not None:
                    merged["progress"] = merged["progress_process"]
                await websocket.send_json(merged)
                if str(merged.get("status", "")).upper() in {"COMPLETED", "FAILED", "ERROR"}:
                    break
            else:
                snapshot = await client.hgetall(key)
                heartbeat = {"type": "heartbeat", "job_id": job_id, **snapshot}
                if "progress" not in heartbeat and heartbeat.get("progress_process") is not None:
                    heartbeat["progress"] = heartbeat["progress_process"]
                await websocket.send_json(heartbeat)
                if str(heartbeat.get("status", "")).upper() in {"COMPLETED", "FAILED", "ERROR"}:
                    break
    except WebSocketDisconnect:
        logger.info("Websocket disconnected for job_id=%s", job_id)
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.close()


def apply_external_options(settings: SimpleNamespace, request: ProcessRequest) -> SimpleNamespace:
    options = request.options
    if options.ocr_language:
        settings.lang = options.ocr_language
    if options.chunk_size:
        settings.max_chunk_tokens = options.chunk_size
    if options.embedding_model:
        settings.embedding_model = options.embedding_model
    if options.importance_threshold is not None:
        settings.importance_threshold = options.importance_threshold
    if options.ga_format:
        settings.qa_format = options.ga_format
    if options.max_chunks is not None:
        settings.max_chunks = options.max_chunks
    return settings


def merge_settings(base: dict, overrides: dict | None = None) -> dict:
    merged = dict(base or {})
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                merged[key] = value
    # Ensure JSON/Celery safe values (no Path objects).
    for key, value in list(merged.items()):
        if isinstance(value, Path):
            merged[key] = str(value)
    return merged


@app.post("/process-request")
async def process_request(payload: ProcessRequest = Body(...)) -> JSONResponse:
    """Accepts a JSON request with doc_id, file_path, and options."""
    settings = default_settings()
    settings.document_id = payload.doc_id
    settings = apply_external_options(settings, payload)
    job_id = derive_job_id(payload)
    settings.job_id = job_id

    if payload.process == ProcessType.PROCESS_PDF:
        if not payload.file_path:
            return JSONResponse({"error": "file_path is required for process_pdf"}, status_code=400)
        merged_settings = merge_settings(settings.__dict__, payload.metadata or {})
        # Ensure file_path is JSON serializable before passing to Celery.
        merged_settings["file_path"] = str(payload.file_path)
        task = enqueue_pipeline(Path(payload.file_path), settings=merged_settings, persist_local=settings.persist_local)
        await set_progress(job_id=job_id, doc_id=payload.doc_id, progress=0, step_progress=0, status="QUEUED", current_step="ingestion", extra={"process": ProcessType.PROCESS_PDF.value, "task_id": task.id})
        return JSONResponse(
            {
                "task_id": task.id,
                "job_id": job_id,
                "document_id": payload.doc_id,
                "process": payload.process.value,
                "options": merged_settings,
                "metadata": payload.metadata,
                "status": "queued",
            }
        )

    if payload.process == ProcessType.GENERATE_QUESTION:
        task_payload = {
            "job_id": job_id,
            "doc_id": payload.doc_id,
            "theme": payload.theme,
            "quantity_question": payload.quantity_question,
            "difficulty": payload.difficulty,
            "question_format": payload.question_format,
            "tags": payload.tags,
            "query_text": payload.query_text,
            "top_k": payload.top_k,
            "min_importance": payload.min_importance,
            "metadata": payload.metadata,
        }
        settings_payload = merge_settings(settings.__dict__, payload.metadata or {})
        task = generate_questions_task.apply_async(args=[task_payload, settings_payload], task_id=job_id)
        await set_progress(job_id=job_id, doc_id=payload.doc_id, progress=0, step_progress=0, status="QUEUED", current_step="generate_question", extra={"process": ProcessType.GENERATE_QUESTION.value, "task_id": task.id})
        return JSONResponse(
            {
                "task_id": task.id,
                "job_id": job_id,
                "document_id": payload.doc_id,
                "process": payload.process.value,
                "options": settings_payload,
                "metadata": payload.metadata,
                "status": "queued",
            }
        )

    return JSONResponse({"error": f"Unsupported process {payload.process}"}, status_code=400)
