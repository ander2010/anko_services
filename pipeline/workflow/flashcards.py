from __future__ import annotations

import asyncio
import datetime as dt
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pipeline.db.flashcard_storage import (
    init_flashcard_db,
    load_flashcards_for_job,
    upsert_flashcards,
)
from pipeline.workflow.utils.progress import PROGRESS_DB_URL, flashcard_redis_key, get_progress_client
from pipeline.utils.logging_config import get_logger

logger = get_logger("pipeline.flashcards")

PROMPT_VERSION = "v1"  # Used for deterministic flashcard job ids.
LEARNING_STEPS_SECONDS = [60, 600]  # 1m, 10m
SESSION_MAX_WAIT_SECONDS = 600  # keep websocket alive for up to 10 minutes waiting for due cards


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


class RatingScale(Enum):
    HARD = 0
    GOOD = 1
    EASY = 2


class FlashcardWorkflow:
    """Shared flashcard state and helpers for websocket/session handling."""

    flashcard_store: dict[str, dict[str, Flashcard]] = {}
    flashcard_requests: dict[str, dict[str, Any]] = {}
    flashcard_tokens: dict[str, str] = {}
    flashcard_inflight: dict[str, dict] = {}
    flashcard_lock = asyncio.Lock()

    @staticmethod
    def derive_flashcard_job_id(request: dict[str, Any]) -> str:
        provided_job_id = str(request.get("job_id") or "").strip()
        if provided_job_id:
            return provided_job_id

        seed_data = {
            "user_id": request.get("user_id"),
            "document_ids": sorted(request.get("document_ids") or []),
            "tags": sorted(request.get("tags") or []),
            "difficulty": request.get("difficulty") or "",
            "prompt_version": PROMPT_VERSION,
        }
        seed = json.dumps(seed_data, sort_keys=True, separators=(",", ":"))
        return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

    @staticmethod
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

    @staticmethod
    def select_next_due_card(cards: dict[str, Flashcard]) -> Flashcard | None:
        now = dt.datetime.now(dt.timezone.utc)
        due_cards = [c for c in cards.values() if c.due_at and c.due_at <= now]
        if not due_cards:
            return None
        return min(due_cards, key=lambda c: c.due_at)

    @staticmethod
    def flashcard_stats(cards: dict[str, Flashcard]) -> tuple[int, int]:
        delivered_new = sum(1 for c in cards.values() if c.kind == "new")
        delivered_review = sum(1 for c in cards.values() if c.kind == "review")
        return delivered_new, delivered_review

    @staticmethod
    def next_due_seconds(cards: dict[str, Flashcard]) -> float | None:
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

    @staticmethod
    def _deserialize(raw: str | None) -> dict[str, Flashcard]:
        if not raw:
            return {}
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

    @classmethod
    async def load_flashcards(cls, job_id: str, user_id: str) -> dict[str, Flashcard]:
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
                await cls.save_flashcards(job_id, cards_dict)
            return cards_dict
        return cls._deserialize(raw)

    @classmethod
    async def save_flashcards(cls, job_id: str, cards: dict[str, Flashcard]) -> None:
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

    @classmethod
    async def wait_for_cards(cls, job_id: str, user_id: str, *, retries: int = 60, delay: float = 1.0) -> dict[str, Flashcard]:
        """Poll Redis/DB for cards after async generation."""
        for _ in range(retries):
            cards = await cls.load_flashcards(job_id, user_id)
            if cards:
                return cards
            await asyncio.sleep(delay)
        return {}


# Initialize flashcard DB on import for Celery/task use-cases.
init_flashcard_db(PROGRESS_DB_URL)
