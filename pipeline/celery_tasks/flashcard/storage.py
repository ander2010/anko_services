from __future__ import annotations

import datetime as dt
import json
from typing import Any, Iterable

import redis

from pipeline.db.flashcard_storage import load_flashcards_for_job, upsert_flashcards
from pipeline.utils.logging_config import get_logger

logger = get_logger(__name__)


class FlashcardStorage:
    """Storage helper for flashcards across Redis cache and SQL backend."""

    def __init__(self, redis_url: str, db_url: str) -> None:
        self.redis_url = redis_url
        self.db_url = db_url
        self._redis = redis.from_url(redis_url, decode_responses=True)

    def redis_key(self, job_id: str) -> str:
        return f"flashcards:cards:{job_id}"

    def load(self, user_id: str, job_id: str) -> list[dict[str, Any]]:
        key = self.redis_key(job_id)
        raw = self._redis.get(key)
        cards: list[dict[str, Any]] = []
        if raw:
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    cards = [c for c in data if isinstance(c, dict)]
            except Exception:
                cards = []
        if not cards:
            try:
                from_db = load_flashcards_for_job(self.db_url, user_id, job_id)
                cards = [self._model_to_dict(card) for card in from_db]
                if cards:
                    self.save(job_id, cards)
            except Exception:
                logger.warning("FlashcardStorage load from DB failed | job=%s", job_id, exc_info=True)
        return cards

    def save(self, job_id: str, cards: Iterable[dict[str, Any]]) -> None:
        cards_list = list(cards)
        key = self.redis_key(job_id)
        try:
            self._redis.set(key, json.dumps(cards_list, separators=(",", ":")))
        except Exception:
            logger.warning("FlashcardStorage save to Redis failed | job=%s", job_id, exc_info=True)
        try:
            upsert_flashcards(self.db_url, cards_list)
        except Exception:
            logger.warning("FlashcardStorage upsert to DB failed | job=%s", job_id, exc_info=True)

    @staticmethod
    def _model_to_dict(card: Any) -> dict[str, Any]:
        return {
            "card_id": card.card_id,
            "user_id": card.user_id,
            "job_id": card.job_id,
            "front": card.front,
            "back": card.back,
            "deck_id": getattr(card, "deck_id", None),
            "notes": getattr(card, "notes", None),
            "source_doc_id": getattr(card, "source_doc_id", None),
            "tags": getattr(card, "tags", []),
            "difficulty": getattr(card, "difficulty", None),
            "kind": getattr(card, "kind", "new"),
            "status": getattr(card, "status", "learning"),
            "learning_step_index": getattr(card, "learning_step_index", 0),
            "repetition": getattr(card, "repetition", 0),
            "interval_days": getattr(card, "interval_days", 0),
            "ease_factor": getattr(card, "ease_factor", 2.5),
            "due_at": FlashcardStorage._dt_to_iso(getattr(card, "due_at", None)),
            "first_seen_at": FlashcardStorage._dt_to_iso(getattr(card, "first_seen_at", None)),
            "created_at": FlashcardStorage._dt_to_iso(getattr(card, "created_at", None)),
        }

    @staticmethod
    def _dt_to_iso(val: Any) -> str | None:
        if isinstance(val, dt.datetime):
            return val.isoformat()
        try:
            if val:
                return dt.datetime.fromisoformat(str(val)).isoformat()
        except Exception:
            return None
        return None

