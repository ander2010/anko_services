from __future__ import annotations

import datetime as dt
import json
import os
import uuid
from typing import Any, List

import redis
import numpy as np

from celery_app import celery_app
from pipeline.db.flashcard_storage import (
    init_flashcard_db,
    upsert_flashcards,
)
from pipeline.workflow.knowledge_store import LocalKnowledgeStore
from pipeline.workflow.llm import LLMFlashcardGenerator
from pipeline.utils.logging_config import get_logger
from pipeline.workflow.vectorizer import Chunkvectorizer
from pipeline.workflow.utils.progress import emit_progress


PROGRESS_REDIS_URL = os.getenv("PROGRESS_REDIS_URL", "redis://localhost:6379/2")
DB_URL = os.getenv("DB_URL", "hope/vector_store.db")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
init_flashcard_db(DB_URL)
logger = get_logger(__name__)
_EMBEDDER = None


def _get_client() -> redis.Redis:
    return redis.from_url(PROGRESS_REDIS_URL, decode_responses=True)


def _redis_key(job_id: str) -> str:
    return f"flashcards:cards:{job_id}"


def _serialize_cards(cards: list[dict[str, Any]]) -> str:
    return json.dumps(cards, separators=(",", ":"))


def _deserialize_cards(raw: str | None) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [c for c in data if isinstance(c, dict)]
    except Exception:
        return []
    return []


def _get_embedder() -> Chunkvectorizer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Chunkvectorizer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDER


def _embed_text(text: str) -> np.ndarray:
    vec = _get_embedder()._model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return vec[0] if vec is not None and len(vec) else np.zeros(1, dtype=float)


def _fetch_context_chunks(request: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    """
    Retrieve similar chunks using doc_id and optional tags for flashcard context.
    Falls back to doc-only results when tags are absent.
    """
    doc_ids = request.get("document_ids") or []
    tags = request.get("tags") or []
    query_text = " ".join(tags) if tags else " ".join(doc_ids)
    if not query_text:
        query_text = "flashcard context"
    try:
        query_vec = _embed_text(query_text)
    except Exception:
        return []

    try:
        with LocalKnowledgeStore(DB_URL) as ks:
            results = ks.query_similar_chunks(
                query_vec.tolist(),
                document_ids=doc_ids or None,
                tags=tags or None,
                min_importance=None,
                top_k=max(1, top_k),
            )
    except Exception:
        return []

    contexts: list[dict[str, Any]] = []
    for doc_id, chunk_index, chunk, score in results:
        text = getattr(chunk, "text", None) or getattr(chunk, "context", "") or ""
        contexts.append(
            {
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "text": text,
                "score": score,
                "tags": (getattr(chunk, "metadata", {}) or {}).get("tags"),
            }
        )
    return contexts


def _is_semantic_duplicate(card: dict[str, Any], existing_cards: list[dict[str, Any]], threshold: float = 0.92) -> bool:
    if not existing_cards:
        return False
    try:
        new_vec = _embed_text(card.get("front", ""))
        existing_vecs = [_embed_text(c.get("front", "")) for c in existing_cards]
        existing_matrix = np.vstack(existing_vecs)
        sims = existing_matrix.dot(new_vec)
        return float(np.max(sims)) >= threshold
    except Exception:
        return False


def _llm_prompt(request: dict[str, Any], count: int) -> List[dict[str, str]]:
    """
    Generate flashcards, supplying retrieved context chunks (doc_id/tags) to the LLM.
    """
    generator = LLMFlashcardGenerator(model=OPENAI_MODEL)
    if not generator.is_active or count <= 0:
        return []
    try:
        topics = request.get("tags") or []
        docs = request.get("document_ids") or []
        difficulty = request.get("difficulty")
        contexts = request.get("_contexts") or []
        context_lines = "\n".join(
            [
                f"- ({ctx.get('doc_id')}#{ctx.get('chunk_index')}) {ctx.get('text', '')[:400]}"
                for ctx in contexts[: max(1, count * 2)]
            ]
        )
        context_block = f"\nUse these context snippets:\n{context_lines}\n" if context_lines else ""
        cards: List[dict[str, str]] = []
        MAX_ATTEMPTS = count * 4
        attempts = 0
        while len(cards) < count and attempts < MAX_ATTEMPTS:
            attempts += 1
            batch = generator.generate(
                topics=topics,
                documents=docs,
                difficulty=difficulty,
                count=1,
                avoid_fronts=[c.get("front") for c in cards[-8:]],
                prompt_context=context_block,
            )
            if not batch:
                continue
            card = batch[0]
            if _is_semantic_duplicate(card, cards):
                continue
            cards.append(card)
        return cards
    except Exception:
        return []


@celery_app.task(name="flashcards.generate")
def generate_flashcards_task(job_id: str, request: dict[str, Any]) -> dict[str, Any]:
    """Generate placeholder flashcards for a job if missing; idempotent."""

    client = _get_client()
    key = _redis_key(job_id)
    existing_raw = client.get(key)
    existing = _deserialize_cards(existing_raw)

    quantity = max(0, int(request.get("quantity") or 0))
    existing_count = len(existing)
    to_generate = max(0, quantity - existing_count)
    contexts = _fetch_context_chunks(request, max(to_generate, quantity))
    request_with_context = dict(request)
    request_with_context["_contexts"] = contexts
    if to_generate > 0 and not contexts:
        emit_progress(
            job_id=job_id,
            doc_id=None,
            progress=100,
            step_progress=100,
            status="COMPLETED",
            current_step="flashcard_generation",
            extra={"generated": 0, "total": existing_count, "reason": "no_context_hits"},
        )
        logger.info("Flashcard generation skipped | job=%s reason=no_context_hits", job_id)
        return {"job_id": job_id, "generated": 0, "total": existing_count}
    if to_generate == 0:
        emit_progress(job_id=job_id, doc_id=None, progress=100, step_progress=100, status="COMPLETED", current_step="flashcard_generation", extra={"generated": 0, "total": existing_count})
        return {"job_id": job_id, "generated": 0, "total": existing_count}

    emit_progress(job_id=job_id, doc_id=None, progress=0, step_progress=0, status="RUNNING", current_step="flashcard_generation", extra={"to_generate": to_generate})
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    llm_cards = _llm_prompt(request_with_context, to_generate)
    logger.info(
        "Flashcard generation start | job=%s user=%s doc_ids=%s tags=%s to_generate=%s llm_cards=%s",
        job_id,
        request.get("user_id"),
        request.get("document_ids"),
        request.get("tags"),
        to_generate,
        len(llm_cards),
    )
    for idx in range(to_generate):
        card_id = str(uuid.uuid4())
        if idx < len(llm_cards):
            front = llm_cards[idx]["front"]
            back = llm_cards[idx]["back"]
        else:
            front = f"Q{existing_count + idx + 1}: Explain concept for docs {', '.join(request.get('document_ids') or [])} with tags {', '.join(request.get('tags') or [])}"
            back = "Placeholder answer. Replace with LLM-generated content."
        card = {
            "card_id": card_id,
            "user_id": request.get("user_id"),
            "job_id": job_id,
            "front": front,
            "back": back,
            "source_doc_id": (request.get("document_ids") or [None])[0],
            "tags": request.get("tags") or [],
            "difficulty": request.get("difficulty"),
            "kind": "new",
            "repetition": 0,
            "interval_days": 0,
            "ease_factor": 2.5,
            "due_at": now,
            "first_seen_at": None,
            "created_at": now,
        }
        existing.append(card)
        logger.info(
            "Flashcard generated | job=%s card_id=%s doc_id=%s tags=%s front=%s back=%s",
            job_id,
            card_id,
            card["source_doc_id"],
            card["tags"],
            front,
            back,
        )
        try:
            emit_progress(
                job_id=job_id,
                doc_id=card["source_doc_id"],
                progress=int(((existing_count + idx + 1) / max(quantity or 1, 1)) * 100),
                step_progress=int(((idx + 1) / to_generate) * 100),
                status="RUNNING",
                current_step="flashcard_generation",
                extra={"generated": idx + 1, "total": quantity, "card_id": card_id},
            )
        except Exception:
            logger.warning("Failed to emit progress | job=%s card_id=%s", job_id, card_id, exc_info=True)

    client.set(key, _serialize_cards(existing))
    try:
        upsert_flashcards(DB_URL, existing)
    except Exception:
        # Log silently; Celery logger not wired here.
        logger.warning("Flashcard upsert failed | job=%s", job_id, exc_info=True)
    emit_progress(job_id=job_id, doc_id=None, progress=100, step_progress=100, status="COMPLETED", current_step="flashcard_generation", extra={"generated": to_generate, "total": len(existing)})
    logger.info("Flashcard generation complete | job=%s total=%s generated=%s", job_id, len(existing), to_generate)
    return {"job_id": job_id, "generated": to_generate, "total": len(existing)}
