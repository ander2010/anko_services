from __future__ import annotations

import datetime as dt
import json
import os
import time
import uuid
from typing import Any, List

import redis
import numpy as np
import requests

from celery_app import celery_app
from pipeline.db.flashcard_storage import (
    init_flashcard_db,
    upsert_flashcards,
    upsert_flashcard_summaries,
)
from pipeline.workflow.knowledge_store import LocalKnowledgeStore
from pipeline.workflow.llm import LLMFlashcardGenerator, LLMOutputSummarizer
from pipeline.utils.logging_config import get_logger
from pipeline.workflow.vectorizer import Chunkvectorizer
from pipeline.workflow.utils.progress import emit_progress


PROGRESS_REDIS_URL = os.getenv("PROGRESS_REDIS_URL", "redis://localhost:6379/2")
DB_URL = os.getenv("DB_URL", "hope/vector_store.db")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
init_flashcard_db(DB_URL)
logger = get_logger(__name__)
_EMBEDDER = None
_FLASHCARD_BATCH_SIZE = max(1, int(os.getenv("FLASHCARD_LLM_BATCH_SIZE", "5")))
_FLASHCARD_MAX_ATTEMPT_MULTIPLIER = max(1, int(os.getenv("FLASHCARD_LLM_MAX_ATTEMPT_MULTIPLIER", "3")))


def _finalize_callback_attempts() -> int:
    return max(1, int(os.getenv("FLASHCARD_FINALIZE_CALLBACK_ATTEMPTS", "3")))


def _finalize_callback_delay_seconds() -> float:
    return max(0.0, float(os.getenv("FLASHCARD_FINALIZE_CALLBACK_DELAY_SECONDS", "1.0")))


def _post_flashcard_finalize_callback(request: dict[str, Any], result: dict[str, Any], *, status_value: str, error_message: str | None = None) -> None:
    metadata = request.get("metadata") or {}
    callback_url = str(metadata.get("callback_url") or "").strip()
    if not callback_url:
        return

    callback_token = str(metadata.get("callback_token") or os.getenv("INTERNAL_SERVICE_TOKEN", "andelef")).strip()
    source_bundle = request.get("source_bundle") or {}
    callback_payload = {
        "token": callback_token,
        "job_id": request.get("job_id"),
        "deck_id": request.get("deck_id"),
        "status": status_value,
        "title": result.get("title") or request.get("title"),
        "source_bundle": source_bundle,
        "generated": result.get("generated"),
        "total": result.get("total"),
        "error": error_message or result.get("error"),
    }
    attempts = _finalize_callback_attempts()
    delay_seconds = _finalize_callback_delay_seconds()
    headers = {"X-Internal-Token": callback_token} if callback_token else None

    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(
                callback_url,
                json=callback_payload,
                headers=headers,
                timeout=30,
            )
            logger.info(
                "Flashcard finalize callback | url=%s status=%s ok=%s job=%s attempt=%s/%s",
                callback_url,
                response.status_code,
                response.ok,
                request.get("job_id"),
                attempt,
                attempts,
            )
            response.raise_for_status()
            return
        except Exception:
            logger.warning(
                "Flashcard finalize callback failed | url=%s job=%s attempt=%s/%s",
                callback_url,
                request.get("job_id"),
                attempt,
                attempts,
                exc_info=True,
            )
            if attempt >= attempts:
                break
            if delay_seconds > 0:
                time.sleep(delay_seconds)


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
    vectors = _get_embedder().encode_texts([text])
    return np.array(vectors[0], dtype=float) if vectors else np.zeros(1, dtype=float)


def _summary_enabled(request: dict[str, Any]) -> bool:
    metadata = request.get("metadata") or {}
    if bool(metadata.get("skip_summary")):
        return False
    return True


def _normalize_request_source(request: dict[str, Any]) -> dict[str, Any]:
    source_bundle = request.get("source_bundle") or {}
    document_ids = source_bundle.get("document_ids") or request.get("document_ids") or []
    section_ids = source_bundle.get("section_ids") or request.get("section_ids") or []
    tag_group_ids = source_bundle.get("tag_group_ids") or request.get("tag_group_ids") or []
    tags = source_bundle.get("tags") or request.get("tags") or []
    title_hints = source_bundle.get("title_hints") or request.get("title_hints") or []
    return {
        "collection_id": source_bundle.get("collection_id") or request.get("collection_id"),
        "document_ids": [str(item).strip() for item in document_ids if str(item).strip()],
        "section_ids": [str(item).strip() for item in section_ids if str(item).strip()],
        "tag_group_ids": [str(item).strip() for item in tag_group_ids if str(item).strip()],
        "tags": [str(item).strip() for item in tags if str(item).strip()],
        "title_hints": [str(item).strip() for item in title_hints if str(item).strip()],
    }


def _fetch_context_chunks(request: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    """
    Retrieve similar chunks using doc_id and tags as semantic query text.
    Tags influence the embedded query but do not hard-filter chunks.
    """
    source = _normalize_request_source(request)
    doc_ids = source["document_ids"]
    tags = source["tags"]
    query_text = " ".join(tags) if tags else " ".join(source["title_hints"] or doc_ids)
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
                tags=None,
                min_importance=None,
                top_k=max(1, top_k),
            )
            logger.info(
                "Context chunks retrieved results=%s", len(results)
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
        source = _normalize_request_source(request)
        topics = source["tags"]
        docs = source["document_ids"]
        difficulty = request.get("difficulty")
        title = request.get("title")
        contexts = request.get("_contexts") or []
        context_lines = "\n".join(
            [
                f"- ({ctx.get('doc_id')}#{ctx.get('chunk_index')}) {ctx.get('text', '')[:400]}"
                for ctx in contexts[: max(1, count * 2)]
            ]
        )
        hint_line = ""
        if title:
            hint_line = f"\nRequested deck title: {title}\n"
        elif topics or docs:
            hint_line = f"\nFocus topics: {', '.join(topics or docs)}\n"
        context_block = f"{hint_line}\nUse these context snippets:\n{context_lines}\n" if context_lines else hint_line
        cards: List[dict[str, str]] = []
        seen_fronts: set[str] = set()
        MAX_ATTEMPTS = max(2, count * _FLASHCARD_MAX_ATTEMPT_MULTIPLIER)
        attempts = 0
        while len(cards) < count and attempts < MAX_ATTEMPTS:
            attempts += 1
            remaining = count - len(cards)
            batch = generator.generate(
                difficulty=difficulty,
                count=min(_FLASHCARD_BATCH_SIZE, remaining),
                avoid_fronts=[c.get("front") for c in cards],
                prompt_context=context_block,
            )
            if not batch:
                continue
            batch_added = False
            for card in batch:
                front = str(card.get("front") or "").strip()
                normalized_front = front.casefold()
                if not front or normalized_front in seen_fronts:
                    continue
                if _is_semantic_duplicate(card, cards):
                    continue
                cards.append(card)
                seen_fronts.add(normalized_front)
                batch_added = True
                if len(cards) >= count:
                    break
            if not batch_added:
                continue
        return cards
    except Exception as e:
        logger.warning("LLM flashcard generation failed: %s", str(e), exc_info=True)
        return []


@celery_app.task(name="flashcards.generate")
def generate_flashcards_task(job_id: str, request: dict[str, Any]) -> dict[str, Any]:
    """Generate placeholder flashcards for a job if missing; idempotent."""
    try:
        client = _get_client()
        key = _redis_key(job_id)
        existing_raw = client.get(key)
        existing = _deserialize_cards(existing_raw)
        source = _normalize_request_source(request)

        quantity = max(0, int(request.get("quantity") or 0))
        existing_count = len(existing)
        to_generate = max(0, quantity - existing_count)
        contexts = _fetch_context_chunks(request, max(to_generate, quantity))
        request_with_context = dict(request)
        request_with_context["_contexts"] = contexts
        if to_generate > 0 and not contexts:
            result = {"job_id": job_id, "generated": 0, "total": existing_count, "title": request.get("title")}
            emit_progress(
                job_id=job_id,
                doc_id=None,
                progress=100,
                status="COMPLETED",
                current_step="flashcard_generation",
                extra={"generated": 0, "total": existing_count, "reason": "no_context_hits", "title": request.get("title")},
            )
            logger.info("Flashcard generation skipped | job=%s reason=no_context_hits", job_id)
            _post_flashcard_finalize_callback(request, result, status_value="completed")
            return result
        if to_generate == 0:
            result = {"job_id": job_id, "generated": 0, "total": existing_count, "title": request.get("title")}
            emit_progress(
                job_id=job_id,
                doc_id=None,
                progress=100,
                status="COMPLETED",
                current_step="flashcard_generation",
                extra={"generated": 0, "total": existing_count, "title": request.get("title")},
            )
            _post_flashcard_finalize_callback(request, result, status_value="completed")
            return result

        # Seed progress so UI doesn't sit at 0 while generation spins up.
        emit_progress(
            job_id=job_id,
            doc_id=None,
            progress=10,
            status="RUNNING",
            current_step="flashcard_generation",
            extra={"to_generate": to_generate, "title": request.get("title")},
        )
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        llm_cards = _llm_prompt(request_with_context, to_generate)
        logger.info(
            "Flashcard generation start | job=%s user=%s title=%s doc_ids=%s tags=%s to_generate=%s llm_cards=%s",
            job_id,
            request.get("user_id"),
            request.get("title"),
            source["document_ids"],
            source["tags"],
            to_generate,
            len(llm_cards),
        )
        new_cards: list[dict[str, Any]] = []
        for idx in range(to_generate):
            card_id = str(uuid.uuid4())
            if idx < len(llm_cards):
                front = llm_cards[idx]["front"]
                back = llm_cards[idx]["back"]
            else:
                doc_ids = source["document_ids"]
                tags = source["tags"]
                front = f"Q{existing_count + idx + 1}: Explain concept for docs {', '.join(doc_ids)} with tags {', '.join(tags)}"
                back = "Placeholder answer. Replace with LLM-generated content."
            card = {
                "card_id": card_id,
                "user_id": request.get("user_id"),
                "job_id": job_id,
                "front": front,
                "back": back,
                "deck_id": request.get("deck_id"),
                "notes": request.get("notes"),
                "source_doc_id": (source["document_ids"] or [None])[0],
                "tags": source["tags"],
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
            new_cards.append(card)
            try:
                client.set(key, _serialize_cards(existing))
            except Exception:
                logger.warning("Failed to update Redis cache | job=%s", job_id, exc_info=True)
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
                total_requested = max(int(quantity or 0), existing_count + to_generate, 1)
                produced_so_far = existing_count + idx + 1
                # Allocate a final completion bump to 100 after persistence.
                progress_start = 10.0
                progress_end = 95.0
                progress_pct = progress_start + (produced_so_far / total_requested) * (progress_end - progress_start)
                emit_progress(
                    job_id=job_id,
                    doc_id=card["source_doc_id"],
                    progress=round(min(progress_pct, 100.0), 2),
                    status="RUNNING",
                    current_step="flashcard_generation",
                    extra={"generated": produced_so_far, "total": total_requested, "card_id": card_id, "title": request.get("title")},
                )
            except Exception:
                logger.warning("Failed to emit progress | job=%s card_id=%s", job_id, card_id, exc_info=True)

        client.set(key, _serialize_cards(existing))
        try:
            upsert_flashcards(DB_URL, existing)
        except Exception:
            # Log silently; Celery logger not wired here.
            logger.warning("Flashcard upsert failed | job=%s", job_id, exc_info=True)
        try:
            summarizer = LLMOutputSummarizer(model=OPENAI_MODEL)
            if _summary_enabled(request) and summarizer.is_active and new_cards:
                max_items = int(os.getenv("SUMMARY_FLASHCARDS_MAX_ITEMS", "40"))
                max_words = int(os.getenv("SUMMARY_FLASHCARDS_MAX_WORDS", "120"))
                max_chars = int(os.getenv("SUMMARY_FLASHCARDS_MAX_CHARS", "8000"))
                lines: list[str] = []
                total_chars = 0
                for card in new_cards[:max_items]:
                    front = (card.get("front") or "").strip()
                    back = (card.get("back") or "").strip()
                    if not front and not back:
                        continue
                    snippet = f"Front: {front}\nBack: {back}".strip()
                    if total_chars + len(snippet) > max_chars:
                        break
                    lines.append(snippet)
                    total_chars += len(snippet)
                if lines:
                    summary_text = summarizer.summarize_collection("\n\n".join(lines), label="flashcards", max_words=max_words)
                    if summary_text:
                        upsert_flashcard_summaries(
                            DB_URL,
                            [
                                {
                                    "user_id": new_cards[0].get("user_id"),
                                    "job_id": job_id,
                                    "summary": summary_text,
                                }
                            ],
                        )
        except Exception:
            logger.warning("Flashcard summary generation failed | job=%s", job_id, exc_info=True)
        emit_progress(
            job_id=job_id,
            doc_id=None,
            progress=100,
            status="COMPLETED",
            current_step="flashcard_generation",
            extra={"generated": to_generate, "total": len(existing), "title": request.get("title")},
        )
        logger.info("Flashcard generation complete | job=%s total=%s generated=%s", job_id, len(existing), to_generate)
        result = {"job_id": job_id, "generated": to_generate, "total": len(existing), "title": request.get("title")}
        _post_flashcard_finalize_callback(request, result, status_value="completed")
        return result
    except Exception as exc:
        emit_progress(
            job_id=job_id,
            doc_id=None,
            progress=100,
            status="FAILED",
            current_step="flashcard_generation",
            extra={"error": str(exc), "title": request.get("title")},
        )
        _post_flashcard_finalize_callback(
            request,
            {"job_id": job_id, "generated": 0, "total": 0, "title": request.get("title"), "error": str(exc)},
            status_value="failed",
            error_message=str(exc),
        )
        raise
