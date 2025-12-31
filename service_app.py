from __future__ import annotations

import json
import asyncio
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

from keybert import KeyBERT
from pipeline.knowledge_store import LocalKnowledgeStore
from pipeline.logging_config import get_logger
from pipeline.task.llm import answer_question_task, direct_answer_task, generate_questions_task
from workflow.vectorizer import Chunkvectorizer
from workflow.celery_pipeline import enqueue_pipeline
from workflow.conversation import CONVERSATION_MAX_MESSAGES, CONVERSATION_MAX_TOKENS, fetch_recent_async
from workflow.utils.persistence import save_notification_async
from workflow.safety import SafetyValidator

logger = get_logger("pipeline.service")

PROGRESS_REDIS_URL = os.getenv("PROGRESS_REDIS_URL", "redis://localhost:6379/2")
PROGRESS_DB_URL = os.getenv("DB_URL", "hope/vector_store.db")
progress_client: Redis | None = None
MAX_CONTEXT_CHUNKS = 15
CONTEXT_TOKEN_LIMIT = int(os.getenv("ASK_CONTEXT_TOKEN_LIMIT", "1800"))


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
    try:
        save_notification_async(
            PROGRESS_DB_URL,
            job_id,
            {
                "status": status,
                "current_step": current_step,
                "progress": progress,
                "step_progress": step_progress,
                "doc_id": doc_id,
                **(extra or {}),
            },
        )
    except Exception:
        logger.warning("Failed to queue notification | job=%s", job_id, exc_info=True)
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


class AskRequest(BaseModel):
    question: str = Field(..., description="User question to answer")
    context: list[str] = Field(default_factory=list, description="List of document IDs to search for context")
    top_k: int | None = Field(None, description="Max chunks to retrieve per document")
    min_importance: float | None = Field(None, description="Minimum importance threshold for retrieved chunks")
    session_id: str | None = Field(None, description="Conversation session id used for chat history lookups")
    user_id: str | None = Field(None, description="User identifier attached to chat history entries")


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
            "quantity_question": request.quantity_question,
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


def _estimate_tokens(text: str | None) -> int:
    return len((text or "").split())


def trim_chunks_to_budget(chunks: list[dict], question: str, token_budget: int = CONTEXT_TOKEN_LIMIT) -> list[dict]:
    """Sort chunks by importance/similarity and keep those that fit within the token budget (including question tokens)."""
    if not chunks:
        return []
    budget = max(0, token_budget - _estimate_tokens(question))
    scored = []
    for ch in chunks:
        meta = ch.get("metadata") or {}
        importance = meta.get("importance")
        try:
            importance = float(importance) if importance is not None else None
        except (TypeError, ValueError):
            importance = None
        similarity = ch.get("similarity")
        try:
            similarity = float(similarity) if similarity is not None else None
        except (TypeError, ValueError):
            similarity = None
        score = importance if importance is not None else (similarity if similarity is not None else 0.0)
        tokens = meta.get("tokens")
        try:
            tokens = int(tokens) if tokens is not None else None
        except (TypeError, ValueError):
            tokens = None
        if tokens is None:
            tokens = _estimate_tokens(ch.get("text"))
        scored.append((score, tokens, ch))

    scored.sort(key=lambda item: item[0], reverse=True)
    kept: list[dict] = []
    used = 0
    for _score, tokens, ch in scored:
        if tokens <= 0:
            continue
        if used + tokens > budget:
            continue
        kept.append(ch)
        used += tokens
    # If we could not keep anything, keep the top chunk.
    if not kept and scored:
        kept.append(scored[0][2])
    return kept


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


@app.websocket("/ws/chat/{session_id}")
async def chat_ws(websocket: WebSocket, session_id: str):
    """Bidirectional chat endpoint: receive questions, enqueue jobs, and stream progress on the same socket."""
    await websocket.accept()
    session_id = (session_id or "").strip() or str(uuid.uuid4())

    client = await get_progress_client()
    listeners: list[asyncio.Task] = []

    async def forward_progress(job: str) -> None:
        pubsub = client.pubsub()
        channel = f"progress:{job}"
        key = f"job:{job}"
        await pubsub.subscribe(channel)
        done_status = {"COMPLETED", "FAILED", "ERROR"}
        try:
            snapshot = await client.hgetall(key)
            if snapshot:
                status = str(snapshot.get("status", "")).upper()
                if status in done_status:
                    await websocket.send_json({"type": "final", "job_id": job, "answer": snapshot.get("answer")})
                    return
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=10.0)
                if message and message.get("data"):
                    data = message["data"]
                    try:
                        payload = json.loads(data)
                    except (TypeError, json.JSONDecodeError):
                        payload = {"raw": data}
                    snapshot = await client.hgetall(key)
                    merged = {**snapshot, **payload}
                    status = str(merged.get("status", "")).upper()
                    if status in done_status:
                        await websocket.send_json({"type": "final", "job_id": job, "answer": merged.get("answer")})
                        break
                else:
                    snapshot = await client.hgetall(key)
                    status = str(snapshot.get("status", "")).upper()
                    if status in done_status and snapshot:
                        await websocket.send_json({"type": "final", "job_id": job, "answer": snapshot.get("answer")})
                        break
        except Exception:
            logger.warning("Progress forwarding failed | job=%s", job, exc_info=True)
        finally:
            try:
                await pubsub.unsubscribe(channel)
            except Exception:
                pass
            try:
                await pubsub.close()
            except Exception:
                pass

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                await websocket.send_json({"error": "invalid JSON"})
                continue

            question = (data.get("question") or "").strip()
            if not question:
                await websocket.send_json({"error": "question is required"})
                continue

            doc_ids_raw = data.get("context") or []
            doc_ids = [str(doc).strip() for doc in doc_ids_raw if str(doc).strip()]
            top_k_raw = data.get("top_k")
            min_importance_raw = data.get("min_importance")
            user_id = (data.get("user_id") or "").strip() or None

            settings = default_settings()
            api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            validator = SafetyValidator(api_key)
            try:
                validator.validate_question(question)
                validator.validate_text_list(doc_ids, field="context document ids")
            except Exception as exc:
                await websocket.send_json({"error": str(exc)})
                continue

            min_importance = min_importance_raw if min_importance_raw is not None else settings.importance_threshold
            conversation_history: list[dict] = []
            try:
                conversation_history = await fetch_recent_async(session_id, token_budget=CONVERSATION_MAX_TOKENS, max_items=CONVERSATION_MAX_MESSAGES)
            except Exception:
                logger.warning("Failed to load conversation history | session=%s", session_id, exc_info=True)

            selected_chunks: list[dict] = []
            missing_docs: list[str] = []
            if doc_ids:
                try:
                    query_vector = embed_question(question, settings.embedding_model)
                except Exception as exc:
                    await websocket.send_json({"error": f"Failed to embed question: {exc}"})
                    continue

                top_k = MAX_CONTEXT_CHUNKS
                try:
                    if top_k_raw:
                        top_k = max(1, int(top_k_raw))
                except (TypeError, ValueError):
                    top_k = MAX_CONTEXT_CHUNKS

                retrieved_chunks: list[dict] = []
                with LocalKnowledgeStore(settings.db_path) as knowledge_store:
                    for doc_id in doc_ids:
                        if not knowledge_store.document_exists(doc_id):
                            missing_docs.append(doc_id)
                            continue
                        results = knowledge_store.query_similar_chunks(query_vector, document_ids=[doc_id], min_importance=min_importance, top_k=top_k)
                        for docid, idx, chunk, similarity in results:
                            meta = dict(chunk.metadata or {})
                            meta.setdefault("chunk_index", idx)
                            meta.setdefault("document_id", docid)
                            retrieved_chunks.append(
                                {
                                    "document_id": docid,
                                    "chunk_index": idx,
                                    "text": chunk.text,
                                    "metadata": meta,
                                    "similarity": float(similarity),
                                }
                            )
                        # Fallback: if no chunks retrieved, pull a top 10 by importance for context.
                        if not retrieved_chunks:
                            fallback = knowledge_store.query_similar_chunks([0.0], document_ids=[doc_id], min_importance=None, top_k=10)
                            for docid, idx, chunk, similarity in fallback:
                                meta = dict(chunk.metadata or {})
                                meta.setdefault("chunk_index", idx)
                                meta.setdefault("document_id", docid)
                                retrieved_chunks.append(
                                    {
                                        "document_id": docid,
                                        "chunk_index": idx,
                                        "text": chunk.text,
                                        "metadata": meta,
                                        "similarity": float(similarity),
                                    }
                                )

                retrieved_chunks.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
                max_chunks = min(len(retrieved_chunks), max(top_k * max(len(doc_ids), 1), 1), MAX_CONTEXT_CHUNKS)
                selected_chunks = trim_chunks_to_budget(retrieved_chunks[:max_chunks], question, CONTEXT_TOKEN_LIMIT)

            job_id = str(uuid.uuid4())
            settings_payload = merge_settings(settings.__dict__, {"importance_threshold": min_importance})
            task_payload = {
                "job_id": job_id,
                "question": question,
                "document_ids": doc_ids,
                "session_id": session_id,
                "user_id": user_id,
                "conversation_history": conversation_history,
            }

            if selected_chunks and not missing_docs:
                task_payload["chunks"] = selected_chunks
                task = answer_question_task.apply_async(args=[task_payload, settings_payload], task_id=job_id)
                await set_progress(job_id=job_id, doc_id=",".join(doc_ids), progress=0, step_progress=0, status="QUEUED", current_step="answer_question", extra={"chunks": len(selected_chunks)})
                mode = "contextual"
            else:
                task = direct_answer_task.apply_async(args=[task_payload, settings_payload], task_id=job_id)
                await set_progress(job_id=job_id, doc_id=",".join(doc_ids), progress=0, step_progress=0, status="QUEUED", current_step="direct_answer", extra={"missing_documents": missing_docs})
                mode = "direct"

            await websocket.send_json(
                {
                    "type": "enqueued",
                    "job_id": job_id,
                    "mode": mode,
                    "question": question,
                    "document_ids": doc_ids,
                    "chunk_count": len(selected_chunks),
                    "missing_documents": missing_docs,
                    "session_id": session_id,
                    "history_messages": len(conversation_history),
                }
            )
            listener = asyncio.create_task(forward_progress(job_id))
            listeners.append(listener)
    except WebSocketDisconnect:
        logger.info("Chat websocket disconnected | session=%s", session_id)
    finally:
        for task in listeners:
            if not task.done():
                task.cancel()


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


_KEYBERT_CACHE: dict[str, KeyBERT] = {}


def get_keyword_model(model_name: str) -> KeyBERT:
    if model_name not in _KEYBERT_CACHE:
        _KEYBERT_CACHE[model_name] = KeyBERT(model=model_name)
    return _KEYBERT_CACHE[model_name]


def embed_question(question: str, model_name: str) -> list[float]:
    vectorizer = Chunkvectorizer(model_name)
    kw_model = get_keyword_model(model_name)
    # Allow multi-language input by avoiding language-specific stopwords.
    keywords = [kw for kw, _score in kw_model.extract_keywords(question, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=8)]
    texts = [question] + keywords
    vectors = vectorizer._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    if len(vectors) > 1:
        return average_embedding_vectors([vec.tolist() for vec in vectors if hasattr(vec, "tolist")])
    return vectors[0].tolist() if len(vectors) else []


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


@app.post("/ask")
async def ask(payload: AskRequest = Body(...)) -> JSONResponse:
    """Answer a question using stored document chunks, with a direct LLM fallback."""
    settings = default_settings()
    question = (payload.question or "").strip()
    session_id = (payload.session_id or str(uuid.uuid4())).strip()
    user_id = (payload.user_id or "").strip() or None
    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    validator = SafetyValidator(api_key)
    validator.validate_question(question)
    validator.validate_text_list(payload.context, field="context document ids")
    min_importance = payload.min_importance if payload.min_importance is not None else settings.importance_threshold

    doc_ids = [str(doc).strip() for doc in payload.context if str(doc).strip()]  
    if not doc_ids:
        logger.info("No document IDs provided in context, falling back to direct LLM answer | question=%s", question)

    conversation_history: list[dict] = []
    try:
        conversation_history = await fetch_recent_async(session_id, token_budget=CONVERSATION_MAX_TOKENS, max_items=CONVERSATION_MAX_MESSAGES)
    except Exception:
        logger.warning("Failed to load conversation history | session=%s", session_id, exc_info=True)

    selected_chunks: list[dict] = []
    missing_docs: list[str] = []
    if doc_ids:
        try:
            query_vector = embed_question(question, settings.embedding_model)
        except Exception as exc:
            logger.warning("Failed to embed question", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to embed question") from exc

        top_k_raw = payload.top_k or 8
        try:
            top_k = max(1, int(top_k_raw))
        except (TypeError, ValueError):
            top_k = 8

        retrieved_chunks: list[dict] = []
        with LocalKnowledgeStore(settings.db_path) as knowledge_store:
            for doc_id in doc_ids:
                if not knowledge_store.document_exists(doc_id):
                    missing_docs.append(doc_id)
                    continue
                results = knowledge_store.query_similar_chunks(query_vector, document_ids=[doc_id], min_importance=min_importance, top_k=top_k)
                for docid, idx, chunk, similarity in results:
                    meta = dict(chunk.metadata or {})
                    meta.setdefault("chunk_index", idx)
                    meta.setdefault("document_id", docid)
                    retrieved_chunks.append(
                        {
                            "document_id": docid,
                            "chunk_index": idx,
                            "text": chunk.text,
                            "metadata": meta,
                            "similarity": float(similarity),
                        }
                    )

        retrieved_chunks.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
        max_chunks = min(len(retrieved_chunks), max(top_k * max(len(doc_ids), 1), 1), MAX_CONTEXT_CHUNKS)
        selected_chunks = trim_chunks_to_budget(retrieved_chunks[:max_chunks], question, CONTEXT_TOKEN_LIMIT)

    job_id = str(uuid.uuid4())
    settings_payload = merge_settings(settings.__dict__, {"importance_threshold": min_importance})
    task_payload = {
        "job_id": job_id,
        "question": question,
        "document_ids": doc_ids,
        "session_id": session_id,
        "user_id": user_id,
        "conversation_history": conversation_history,
    }

    if selected_chunks and not missing_docs:
        task_payload["chunks"] = selected_chunks
        task = answer_question_task.apply_async(args=[task_payload, settings_payload], task_id=job_id)
        await set_progress(job_id=job_id, doc_id=",".join(doc_ids), progress=0, step_progress=0, status="QUEUED", current_step="answer_question", extra={"chunks": len(selected_chunks)})
        mode = "contextual"
    else:
        logger.info("No relevant chunks found, falling back to direct LLM answer | question=%s", question)
        task = direct_answer_task.apply_async(args=[task_payload, settings_payload], task_id=job_id)
        await set_progress(job_id=job_id, doc_id=",".join(doc_ids), progress=0, step_progress=0, status="QUEUED", current_step="direct_answer", extra={ "missing_documents": missing_docs})
        mode = "direct"
    return JSONResponse(
        {
            "task_id": task.id,
            "job_id": job_id,
            "mode": mode,
            "question": question,
            "document_ids": doc_ids,
            "chunk_count": len(selected_chunks),
            "missing_documents": missing_docs,
            "session_id": session_id,
            "history_messages": len(conversation_history),
        }
    )
