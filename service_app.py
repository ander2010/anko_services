from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from keybert import KeyBERT
from pipeline.workflow.knowledge_store import LocalKnowledgeStore
from pipeline.utils.logging_config import get_logger
from pipeline.celery_tasks.llm import answer_question_task, direct_answer_task, generate_questions_task, generate_question_variants_task
from pipeline.celery_tasks.flashcards import generate_flashcards_task
from pipeline.workflow.vectorizer import Chunkvectorizer
from pipeline.workflow.celery_pipeline import enqueue_pipeline
from pipeline.workflow.conversation import CONVERSATION_MAX_MESSAGES, CONVERSATION_MAX_TOKENS, fetch_recent_async
from pipeline.workflow.utils.persistence import save_notification_async
from pipeline.workflow.safety import SafetyValidator
from pipeline.db.flashcard_storage import (
    init_flashcard_db,
    upsert_flashcards,
    insert_review,
    load_flashcards_for_job,
    ensure_flashcard_job,
    set_flashcard_job_status,
    get_flashcard_job,
)

logger = get_logger("pipeline.service")

PROGRESS_REDIS_URL = os.getenv("PROGRESS_REDIS_URL", "redis://localhost:6379/2")
PROGRESS_DB_URL = os.getenv("DB_URL", "hope/vector_store.db")
progress_client: Redis | None = None
MAX_CONTEXT_CHUNKS = 15
CONTEXT_TOKEN_LIMIT = int(os.getenv("ASK_CONTEXT_TOKEN_LIMIT", "1800"))
PROMPT_VERSION = "v1"  # Used for deterministic flashcard job ids.


@dataclass
class Flashcard:
    card_id: str
    user_id: str
    job_id: str
    front: str
    back: str
    source_doc_id: str | None
    tags: list[str]
    difficulty: str | None
    kind: str = "new"
    status: str = "learning"  # learning | review
    learning_step_index: int = 0
    repetition: int = 0
    interval_days: int = 0
    ease_factor: float = 2.5
    due_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    first_seen_at: dt.datetime | None = None
    created_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))


class FlashcardStartRequest(BaseModel):
    document_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    quantity: int = Field(..., gt=0, description="Number of new cards to generate")
    difficulty: str | None = Field(None, description="Difficulty hint for generation")
    user_id: str = Field(..., description="User identifier")


class RatingScale(Enum):
    HARD = 0
    GOOD = 1
    EASY = 2


LEARNING_STEPS_SECONDS = [60, 600]  # 1m, 10m
SESSION_MAX_WAIT_SECONDS = 600  # keep websocket alive for up to 10 minutes waiting for due cards


# Flashcard store keyed by job_id; each holds card_id -> Flashcard.
flashcard_store: dict[str, dict[str, Flashcard]] = {}
# Cache the latest start request per job for validation/idempotency.
flashcard_requests: dict[str, FlashcardStartRequest] = {}
# Track issued tokens per job for lightweight validation.
flashcard_tokens: dict[str, str] = {}
# Track in-flight cards awaiting feedback per job for replay.
flashcard_inflight: dict[str, dict] = {}
flashcard_lock = asyncio.Lock()


def derive_flashcard_job_id(request: FlashcardStartRequest) -> str:
    seed_data = {
        "user_id": request.user_id,
        "document_ids": sorted(request.document_ids or []),
        "tags": sorted(request.tags or []),
        "difficulty": request.difficulty or "",
        "prompt_version": PROMPT_VERSION,
    }
    seed = json.dumps(seed_data, sort_keys=True, separators=(",", ":"))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def update_card_schedule(card: Flashcard, rating: int, now: dt.datetime | None = None, time_to_answer_ms: int | None = None) -> Flashcard:
    """
    Anki-like behavior with learning steps:
    - Learning steps: 1m, 10m. Again -> step 0; Good -> next step; Easy -> graduate.
    - After graduation (status=review), apply day intervals; Again sends back to learning step 0.
    - Latency-aware: very slow responses can downgrade rating.
    """
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)

    # Adjust rating based on response time.
    if time_to_answer_ms is not None:
        try:
            t = int(time_to_answer_ms)
            if t > 10000:  # >10s: downgrade one level
                rating = max(RatingScale.HARD.value, rating - 1)
            elif rating == RatingScale.EASY.value and t > 8000:
                rating = RatingScale.GOOD.value
        except Exception:
            pass

    def schedule_learning(step_idx: int) -> None:
        card.status = "learning"
        card.learning_step_index = max(0, step_idx)
        seconds = LEARNING_STEPS_SECONDS[card.learning_step_index]
        card.due_at = now + dt.timedelta(seconds=seconds)
        card.interval_days = 0

    def graduate(interval_days: int = 1) -> None:
        card.status = "review"
        card.kind = "review"
        card.learning_step_index = 0
        card.repetition = max(1, card.repetition)
        card.interval_days = max(1, interval_days)
        card.due_at = now + dt.timedelta(days=card.interval_days)

    if card.status != "review":
        if rating == RatingScale.HARD.value:
            schedule_learning(0)
            card.repetition = 0
            card.ease_factor = max(1.3, card.ease_factor - 0.2)
        elif rating == RatingScale.GOOD.value:
            next_idx = card.learning_step_index + 1
            if next_idx < len(LEARNING_STEPS_SECONDS):
                schedule_learning(next_idx)
            else:
                graduate(1)
            card.ease_factor = max(1.3, card.ease_factor + 0.02)
        else:  # EASY
            graduate(1)
            card.ease_factor = max(1.3, card.ease_factor + 0.08)
    else:
        if rating == RatingScale.HARD.value:
            schedule_learning(0)
            card.repetition = 0
            card.ease_factor = max(1.3, card.ease_factor - 0.2)
        elif rating == RatingScale.GOOD.value:
            card.repetition = max(1, card.repetition + 1)
            if card.repetition == 1:
                card.interval_days = 1
            elif card.repetition == 2:
                card.interval_days = 6
            else:
                card.interval_days = max(1, int(round(card.interval_days * card.ease_factor)))
            card.ease_factor = max(1.3, card.ease_factor + 0.02)
            card.due_at = now + dt.timedelta(days=card.interval_days)
        else:  # EASY
            card.repetition = max(1, card.repetition + 1)
            if card.repetition <= 1:
                card.interval_days = 1
            elif card.repetition == 2:
                card.interval_days = 6
            else:
                card.interval_days = max(1, int(round(card.interval_days * (card.ease_factor + 0.05))))
            card.ease_factor = max(1.3, card.ease_factor + 0.08)
            card.due_at = now + dt.timedelta(days=card.interval_days)

    if card.first_seen_at is None:
        card.first_seen_at = now
    return card


def _select_next_due_card(cards: dict[str, Flashcard]) -> Flashcard | None:
    now = dt.datetime.now(dt.timezone.utc)
    due_cards = [
        c
        for c in cards.values()
        if c.due_at and c.due_at <= now
    ]
    if not due_cards:
        return None
    due_cards.sort(
        key=lambda c: (
            0 if c.kind == "review" else 1,
            c.due_at,
            c.card_id,
        )
    )
    return due_cards[0]


def _flashcard_stats(cards: dict[str, Flashcard]) -> tuple[int, int]:
    delivered_new = sum(1 for c in cards.values() if c.kind == "new" and c.first_seen_at)
    delivered_review = sum(1 for c in cards.values() if c.repetition > 1)
    return delivered_new, delivered_review


def _next_due_seconds(cards: dict[str, Flashcard]) -> float | None:
    now = dt.datetime.now(dt.timezone.utc)
    deltas = []
    for card in cards.values():
        if not card.due_at:
            continue
        delta = (card.due_at - now).total_seconds()
        if delta > 0:
            deltas.append(delta)
    if not deltas:
        return None
    return min(deltas)


def _generate_cards(request: FlashcardStartRequest, job_id: str, count: int) -> list[Flashcard]:
    now = dt.datetime.now(dt.timezone.utc)
    cards: list[Flashcard] = []
    for idx in range(count):
        card_id = str(uuid.uuid4())
        front = f"Q{idx + 1}: Explain concept for docs {', '.join(request.document_ids)} with tags {', '.join(request.tags)}"
        back = "Placeholder answer. Replace with LLM-generated content."
        card = Flashcard(
            card_id=card_id,
            user_id=request.user_id,
            job_id=job_id,
            front=front,
            back=back,
            source_doc_id=request.document_ids[0] if request.document_ids else None,
            tags=request.tags,
            difficulty=request.difficulty,
            kind="new",
            status="learning",
            learning_step_index=0,
            repetition=0,
            interval_days=0,
            ease_factor=2.5,
            due_at=now,
            first_seen_at=None,
            created_at=now,
        )
        cards.append(card)
    return cards


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


def flashcard_redis_key(job_id: str) -> str:
    return f"flashcards:cards:{job_id}"


async def load_flashcards(job_id: str, user_id: str) -> dict[str, Flashcard]:
    client = await get_progress_client()
    raw = await client.get(flashcard_redis_key(job_id))
    if not raw:
        # Fallback to DB if Redis is empty.
        cards_from_db = load_flashcards_for_job(PROGRESS_DB_URL, user_id, job_id)
        cards_dict: dict[str, Flashcard] = {}
        for db_card in cards_from_db:
            cards_dict[db_card.card_id] = Flashcard(
                card_id=db_card.card_id,
                user_id=db_card.user_id,
                job_id=db_card.job_id,
                front=db_card.front,
                back=db_card.back,
                source_doc_id=db_card.source_doc_id,
                tags=db_card.tags or [],
                difficulty=db_card.difficulty,
                kind=db_card.kind,
                repetition=db_card.repetition,
                interval_days=db_card.interval_days,
                ease_factor=db_card.ease_factor,
                due_at=db_card.due_at or dt.datetime.now(dt.timezone.utc),
                first_seen_at=db_card.first_seen_at,
                created_at=db_card.created_at or dt.datetime.now(dt.timezone.utc),
            )
        if cards_dict:
            await save_flashcards(job_id, cards_dict)
        return cards_dict
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    cards: dict[str, Flashcard] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                card = Flashcard(
                    card_id=item["card_id"],
                    user_id=item.get("user_id") or "",
                    job_id=item.get("job_id") or "",
                    front=item.get("front") or "",
                    back=item.get("back") or "",
                    source_doc_id=item.get("source_doc_id"),
                    tags=item.get("tags") or [],
                    difficulty=item.get("difficulty"),
                    kind=item.get("kind", "new"),
                    status=item.get("status", "learning"),
                    learning_step_index=int(item.get("learning_step_index", 0) or 0),
                    repetition=int(item.get("repetition", 0) or 0),
                    interval_days=int(item.get("interval_days", 0) or 0),
                    ease_factor=float(item.get("ease_factor", 2.5) or 2.5),
                    due_at=dt.datetime.fromisoformat(item.get("due_at")) if item.get("due_at") else dt.datetime.now(dt.timezone.utc),
                    first_seen_at=dt.datetime.fromisoformat(item["first_seen_at"]) if item.get("first_seen_at") else None,
                    created_at=dt.datetime.fromisoformat(item.get("created_at")) if item.get("created_at") else dt.datetime.now(dt.timezone.utc),
                )
            except Exception:
                continue
            cards[card.card_id] = card
    return cards


async def save_flashcards(job_id: str, cards: dict[str, Flashcard]) -> None:
    client = await get_progress_client()
    payload = []
    for card in cards.values():
        payload.append(
            {
                "card_id": card.card_id,
                "user_id": card.user_id,
                "job_id": card.job_id,
                "front": card.front,
                "back": card.back,
                "source_doc_id": card.source_doc_id,
                "tags": card.tags,
                "difficulty": card.difficulty,
                "kind": card.kind,
                "status": card.status,
                "learning_step_index": card.learning_step_index,
                "repetition": card.repetition,
                "interval_days": card.interval_days,
                "ease_factor": card.ease_factor,
                "due_at": card.due_at.isoformat(),
                "first_seen_at": card.first_seen_at.isoformat() if card.first_seen_at else None,
                "created_at": card.created_at.isoformat(),
            }
        )
    await client.set(flashcard_redis_key(job_id), json.dumps(payload, separators=(",", ":")))
    try:
        upsert_flashcards(PROGRESS_DB_URL, payload)
    except Exception:
        logger.warning("Failed to upsert flashcards to DB | job=%s", job_id, exc_info=True)


async def wait_for_cards(job_id: str, user_id: str, *, retries: int = 60, delay: float = 1.0) -> dict[str, Flashcard]:
    """Poll Redis/DB for cards after async generation."""
    for attempt in range(retries):
        cards = await load_flashcards(job_id, user_id)
        if cards:
            return cards
        await asyncio.sleep(delay)
    return {}


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


class QuestionVariantsRequest(BaseModel):
    question_id: str = Field(..., description="Existing question_id to generate variants from")
    quantity: int = Field(default=10, description="Number of variant questions to generate")
    difficulty: str = Field(default="medium", description="Difficulty hint for variants")
    question_format: str = Field(default="variety", description="Output format (e.g., variety, true_false)")
    job_id: str | None = Field(default=None, description="Optional job id; derived deterministically if omitted")


app = FastAPI(title="Pipeline Streaming Service")
init_flashcard_db(PROGRESS_DB_URL)


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


def derive_variant_job_id(question_id: str, quantity: int, difficulty: str, question_format: str) -> str:
    seed_data = {
        "question_id": question_id,
        "quantity": quantity,
        "difficulty": difficulty,
        "question_format": question_format,
    }
    seed = json.dumps(seed_data, sort_keys=True, separators=(",", ":"))
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


@app.post("/questions/{question_id}/variants")
async def generate_question_variants(question_id: str, payload: QuestionVariantsRequest = Body(...)) -> JSONResponse:
    """Generate variant questions for an existing question_id."""
    settings = default_settings()
    quantity = payload.quantity or 10
    difficulty = (payload.difficulty or "medium").strip() or "medium"
    question_format = (payload.question_format or "variety").strip() or "variety"
    job_id = payload.job_id or derive_variant_job_id(question_id, quantity, difficulty, question_format)

    task_payload = {
        "job_id": job_id,
        "question_id": question_id,
        "quantity": quantity,
        "difficulty": difficulty,
        "question_format": question_format,
    }
    settings_payload = merge_settings(settings.__dict__, {})
    task = generate_question_variants_task.apply_async(args=[task_payload, settings_payload], task_id=job_id)
    await set_progress(job_id=job_id, doc_id=None, progress=0, step_progress=0, status="QUEUED", current_step="qa_variants", extra={"parent_question_id": question_id, "quantity": quantity})
    return JSONResponse(
        {
            "task_id": task.id,
            "job_id": job_id,
            "parent_question_id": question_id,
            "quantity": quantity,
            "difficulty": difficulty,
            "question_format": question_format,
            "status": "queued",
        }
    )


@app.post("/study/start")
async def flashcards_start(payload: FlashcardStartRequest = Body(...)) -> JSONResponse:
    """Start a flashcard session; returns deterministic job_id and WS info."""
    job_id = derive_flashcard_job_id(payload)
    token = str(uuid.uuid4())
    async with flashcard_lock:
        flashcard_requests[job_id] = payload
        flashcard_tokens[job_id] = token
    return JSONResponse(
        {
            "job_id": job_id,
            "ws_path": "/ws/flashcards",
            "token": token,
        }
    )


@app.websocket("/ws/flashcards")
async def flashcards_ws(websocket: WebSocket):
    await websocket.accept()
    session = {
        "seq": 0,
        "job_id": None,
        "user_id": None,
        "in_flight_card": None,
    }

    try:
        raw = await websocket.receive_text()
        data = json.loads(raw) if raw else {}
        if data.get("message_type") != "subscribe_job":
            await websocket.send_json({"error": "first message must be subscribe_job"})
            await websocket.close()
            return

        job_id = str(data.get("job_id") or "").strip()
        user_id = str(data.get("user_id") or "").strip()
        last_seq = data.get("last_seq") or 0
        request_id = data.get("request_id") or str(uuid.uuid4())
        token = str(data.get("token") or websocket.headers.get("authorization") or "").replace("Bearer ", "").strip()
        if not job_id or not user_id:
            await websocket.send_json({"error": "job_id and user_id are required"})
            await websocket.close()
            return

        async with flashcard_lock:
            req = flashcard_requests.get(job_id)
            if req is None:
                logger.warning("WS subscribe unknown job | job_id=%s user_id=%s", job_id, user_id)
                await websocket.send_json({"error": "unknown job_id"})
                await websocket.close()
                return
            if req.user_id != user_id:
                logger.warning("WS subscribe user mismatch | job_id=%s user_id=%s expected=%s", job_id, user_id, req.user_id)
                await websocket.send_json({"error": "job_id does not belong to user"})
                await websocket.close()
                return
            expected_token = flashcard_tokens.get(job_id)
            if expected_token and token and token != expected_token:
                logger.warning("WS subscribe token invalid | job_id=%s user_id=%s", job_id, user_id)
                await websocket.send_json({"error": "invalid token"})
                await websocket.close()
                return
            ensure_flashcard_job(PROGRESS_DB_URL, job_id, user_id, req.quantity)
            store = flashcard_store.setdefault(job_id, {})
            existing_cards = await load_flashcards(job_id, user_id)
            if existing_cards:
                store.update(existing_cards)
            existing_count = len(store)
            to_generate = max(0, req.quantity - existing_count)
            if to_generate:
                logger.info("WS triggering generation | job_id=%s user_id=%s existing=%s to_generate=%s", job_id, user_id, existing_count, to_generate)
                # Offload generation to Celery; idempotent on job_id + request.
                generate_flashcards_task.apply_async(args=[job_id, req.model_dump()])
            inflight = flashcard_inflight.get(job_id)
            inflight = flashcard_inflight.get(job_id)

        session["seq"] = int(last_seq) if last_seq else 0
        session["job_id"] = job_id
        session["user_id"] = user_id
        session["in_flight_card"] = inflight["card_id"] if inflight else None

        await websocket.send_json({"message_type": "accepted", "job_id": job_id, "request_id": request_id})
        logger.info("WS accepted | job_id=%s user_id=%s seq=%s", job_id, user_id, session["seq"])

        async def send_card(card: Flashcard) -> None:
            session["seq"] += 1
            seq = session["seq"]
            session["in_flight_card"] = card.card_id
            async with flashcard_lock:
                flashcard_inflight[job_id] = {"seq": seq, "card_id": card.card_id}
                await save_flashcards(job_id, flashcard_store.get(job_id, {}))
            await websocket.send_json(
                {
                    "message_type": "card",
                    "seq": seq,
                    "job_id": job_id,
                    "kind": card.kind,
                    "card": {
                        "id": card.card_id,
                        "front": card.front,
                        "back": card.back,
                        "source_doc_id": card.source_doc_id,
                        "tags": card.tags,
                        "difficulty": card.difficulty,
                    },
                }
            )

        async def send_done() -> None:
            async with flashcard_lock:
                cards = flashcard_store.get(job_id, {})
            delivered_new, delivered_review = _flashcard_stats(cards)
            await websocket.send_json(
                {
                    "message_type": "done",
                    "job_id": job_id,
                    "delivered_new": delivered_new,
                    "delivered_review": delivered_review,
                }
            )

        # If reconnect and a card was in-flight, resend it with same seq.
        if session["in_flight_card"]:
            async with flashcard_lock:
                cards = flashcard_store.get(job_id, {})
                inflight_card = cards.get(session["in_flight_card"])
                inflight_info = flashcard_inflight.get(job_id, {})
            if inflight_card and inflight_info:
                await websocket.send_json(
                    {
                        "message_type": "card",
                        "seq": inflight_info.get("seq", session["seq"]),
                        "job_id": job_id,
                        "kind": inflight_card.kind,
                        "card": {
                            "id": inflight_card.card_id,
                            "front": inflight_card.front,
                            "back": inflight_card.back,
                            "source_doc_id": inflight_card.source_doc_id,
                            "tags": inflight_card.tags,
                            "difficulty": inflight_card.difficulty,
                        },
                    }
                )
        else:
            async with flashcard_lock:
                if to_generate:
                    # Wait briefly for background generation to finish.
                    refreshed = await wait_for_cards(job_id, user_id)
                    if refreshed:
                        flashcard_store[job_id] = refreshed
                        logger.info("WS refreshed after generation | job_id=%s cards=%s", job_id, len(refreshed))
                next_card = _select_next_due_card(flashcard_store.get(job_id, {}))
            if next_card:
                await send_card(next_card)
            else:
                await websocket.send_json({"message_type": "idle", "job_id": job_id})
                logger.info("WS idle after subscribe | job_id=%s", job_id)
                # Wait a bit more for cards to appear before completing.
                refreshed = await wait_for_cards(job_id, user_id, retries=4, delay=0.5)
                if refreshed:
                    flashcard_store[job_id] = refreshed
                    next_card = _select_next_due_card(refreshed)
                    if next_card:
                        await send_card(next_card)
                        # fall through to main loop
                    else:
                        await send_done()
                        logger.info("WS done no card after refresh | job_id=%s", job_id)
                        return
                else:
                    await send_done()
                    logger.info("WS done no cards found | job_id=%s", job_id)
                    return

        # Main loop: wait for feedback, update, then send next.
        while True:
            raw_msg = await websocket.receive_text()
            try:
                message = json.loads(raw_msg) if raw_msg else {}
            except json.JSONDecodeError:
                await websocket.send_json({"error": "invalid JSON"})
                continue

            if message.get("message_type") != "card_feedback":
                await websocket.send_json({"error": "expected card_feedback"})
                continue

            feedback_seq = message.get("seq")
            card_id = message.get("card_id")
            rating = message.get("rating")
            if session["in_flight_card"] != card_id:
                logger.warning("WS feedback card mismatch | job_id=%s expected=%s got=%s", job_id, session["in_flight_card"], card_id)
                await websocket.send_json({"error": "card mismatch"})
                continue

            try:
                rating_int = int(rating)
            except (TypeError, ValueError):
                await websocket.send_json({"error": "rating must be int 0-2"})
                continue
            if rating_int not in {RatingScale.HARD.value, RatingScale.GOOD.value, RatingScale.EASY.value}:
                await websocket.send_json({"error": "rating must be 0 (hard), 1 (good), or 2 (easy)"})
                continue

            async with flashcard_lock:
                cards = flashcard_store.get(job_id, {})
                card = cards.get(card_id)
                if not card:
                    logger.warning("WS feedback card not found | job_id=%s card_id=%s", job_id, card_id)
                    await websocket.send_json({"error": "card not found"})
                    continue
                time_to_answer_ms = message.get("time_to_answer_ms")
                update_card_schedule(card, rating_int, time_to_answer_ms=time_to_answer_ms)
                flashcard_inflight.pop(job_id, None)
                session["in_flight_card"] = None
                await save_flashcards(job_id, cards)
            logger.info(
                "WS feedback applied | job_id=%s card_id=%s rating=%s status=%s step=%s interval_days=%s due_at=%s",
                job_id,
                card_id,
                rating_int,
                card.status,
                card.learning_step_index,
                card.interval_days,
                card.due_at,
            )
            try:
                insert_review(
                    PROGRESS_DB_URL,
                    {
                        "card_id": card_id,
                        "user_id": user_id,
                        "job_id": job_id,
                        "rating": rating_int,
                        "time_to_answer_ms": message.get("time_to_answer_ms"),
                        "notes": message.get("notes"),
                        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    },
                )
            except Exception:
                logger.warning("Failed to insert review | card=%s job=%s", card_id, job_id, exc_info=True)

            await websocket.send_json(
                {
                    "message_type": "ack",
                    "card_id": card_id,
                    "seq": feedback_seq,
                    "job_id": job_id,
                    "next_due": card.due_at.isoformat(),
                    "interval_days": card.interval_days,
                    "ease_factor": round(card.ease_factor, 3),
                }
            )

            async with flashcard_lock:
                next_card = _select_next_due_card(flashcard_store.get(job_id, {}))
            if next_card:
                await send_card(next_card)
            else:
                # No card due; wait briefly in case new cards become due/generated.
                refreshed = await wait_for_cards(job_id, user_id)
                if refreshed:
                    async with flashcard_lock:
                        flashcard_store[job_id] = refreshed
                    next_card = _select_next_due_card(refreshed)
                    if next_card:
                        await send_card(next_card)
                        continue
                await websocket.send_json({"message_type": "idle", "job_id": job_id})
                job_state = get_flashcard_job(PROGRESS_DB_URL, job_id)
                if job_state and job_state.status not in {"completed", "failed"}:
                    # Keep the session alive until either a card becomes due or the job finishes.
                    next_due = _next_due_seconds(flashcard_store.get(job_id, {}))
                    if next_due is not None and next_due <= SESSION_MAX_WAIT_SECONDS:
                        await asyncio.sleep(next_due)
                        continue
                    continue
                else:
                    next_due = _next_due_seconds(flashcard_store.get(job_id, {}))
                    if next_due is not None and next_due <= SESSION_MAX_WAIT_SECONDS:
                        await asyncio.sleep(next_due)
                        continue
                await send_done()
                logger.info("WS loop done | job_id=%s delivered=%s", job_id, _flashcard_stats(flashcard_store.get(job_id, {})))
                break

    except WebSocketDisconnect:
        logger.info("Flashcard websocket disconnected | job=%s", session.get("job_id"))
    except Exception:
        logger.warning("Flashcard websocket error", exc_info=True)
        try:
            await websocket.send_json({"error": "internal error"})
        except Exception:
            pass
