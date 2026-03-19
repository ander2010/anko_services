from __future__ import annotations

import datetime as dt
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import ProgrammingError, IntegrityError

from pipeline.db.models import Base, Flashcard, FlashcardReview, SummaryJob
from pipeline.db.session import create_engine_and_session, build_sqlite_url
from pipeline.utils.logging_config import get_logger

logger = get_logger("pipeline.flashcards.db")
SUMMARY_MAX_CHARS = 800


def _clamp_summary(text: str | None) -> str:
    if not text:
        return ""
    summary = text.strip()
    if len(summary) <= SUMMARY_MAX_CHARS:
        return summary
    return summary[:SUMMARY_MAX_CHARS].rstrip()

_engine = None
_SessionLocal: sessionmaker | None = None
_db_disabled = False


def init_flashcard_db(db_url: str) -> None:
    """Initialize engine and tables if not already initialized."""
    global _engine, _SessionLocal, _db_disabled
    if _db_disabled:
        return
    if _engine is not None and _SessionLocal is not None:
        return
    try:
        _engine, _SessionLocal = create_engine_and_session(db_url)
        # Schema creation disabled; tables must exist already.
    except ModuleNotFoundError as exc:
        logger.warning("Flashcard DB disabled; missing driver: %s", exc)
        _db_disabled = True
    except Exception as exc:
        _db_disabled = True
        logger.warning("Flashcard DB init failed; disabled persistence: %s", exc)


def _session() -> Session:
    if _SessionLocal is None or _db_disabled:
        raise RuntimeError("Flashcard DB not initialized")
    return _SessionLocal()


def upsert_flashcards(db_url: str, cards: Iterable[dict]) -> None:
    if not cards or _db_disabled:
        return
    init_flashcard_db(db_url)
    with _session() as session:
        try:
            for item in cards:
                card_id = item.get("card_id")
                if not card_id:
                    continue
                # Preserve existing fields and drop explicit None for selected keys to avoid nulling on updates.
                existing = session.execute(
                    select(Flashcard).where(Flashcard.card_id == card_id)
                ).scalar_one_or_none()
                if existing:
                    item["id"] = existing.id
                else:
                    item.pop("id", None)
                for key in (
                    "deck_id",
                    "notes",
                    "back_image",
                    "back_image_height",
                    "back_image_width",
                    "back_image_size_bytes",
                    "back_image_original_size_bytes",
                    "back_image_was_optimized",
                    "source_doc_id",
                    "difficulty",
                    "user_id",
                    "job_id",
                    "front",
                    "back",
                    "tags",
                ):
                    if item.get(key) is None:
                        if existing:
                            item[key] = getattr(existing, key, None)
                        else:
                            item.pop(key, None)
                session.merge(_dict_to_model(item))
            session.commit()
        except IntegrityError:
            session.rollback()
            for item in cards:
                card_id = item.get("card_id")
                if not card_id:
                    continue
                existing = session.execute(
                    select(Flashcard).where(Flashcard.card_id == card_id)
                ).scalar_one_or_none()
                if existing:
                    item["id"] = existing.id
                else:
                    item.pop("id", None)
                for key in (
                    "deck_id",
                    "notes",
                    "back_image",
                    "back_image_height",
                    "back_image_width",
                    "back_image_size_bytes",
                    "back_image_original_size_bytes",
                    "back_image_was_optimized",
                    "source_doc_id",
                    "difficulty",
                    "user_id",
                    "job_id",
                    "front",
                    "back",
                    "tags",
                ):
                    if item.get(key) is None:
                        if existing:
                            item[key] = getattr(existing, key, None)
                        else:
                            item.pop(key, None)
                session.merge(_dict_to_model(item))
            session.commit()


def upsert_flashcard_summaries(db_url: str, summaries: Iterable[dict]) -> None:
    if not summaries or _db_disabled:
        return
    init_flashcard_db(db_url)
    with _session() as session:
        try:
            for item in summaries:
                job_id = item.get("job_id")
                if not job_id:
                    continue
                existing = session.execute(
                    select(SummaryJob).where(
                        SummaryJob.job_id == job_id,
                        SummaryJob.item_type == "flashcards",
                    )
                ).scalar_one_or_none()
                if existing:
                    item["id"] = existing.id
                else:
                    item.pop("id", None)
                session.merge(_summary_dict_to_model(item))
            session.commit()
        except IntegrityError:
            session.rollback()
            for item in summaries:
                job_id = item.get("job_id")
                if not job_id:
                    continue
                existing = session.execute(
                    select(SummaryJob).where(
                        SummaryJob.job_id == job_id,
                        SummaryJob.item_type == "flashcards",
                    )
                ).scalar_one_or_none()
                if existing:
                    item["id"] = existing.id
                else:
                    item.pop("id", None)
                session.merge(_summary_dict_to_model(item))
            session.commit()


def insert_review(db_url: str, review: dict) -> None:
    if _db_disabled:
        return
    init_flashcard_db(db_url)
    with _session() as session:
        card_key = review.get("card_id")
        if not card_key:
            return
        flashcard_id = session.execute(
            select(Flashcard.id).where(Flashcard.card_id == card_key)
        ).scalar_one_or_none()
        if flashcard_id is None:
            logger.warning("Flashcard id not found for review | card_id=%s", card_key)
            return
        rec = FlashcardReview(
            card_id=flashcard_id,
            user_id=review.get("user_id"),
            job_id=review.get("job_id"),
            rating=review.get("rating"),
            time_to_answer_ms=review.get("time_to_answer_ms"),
            notes=review.get("notes"),
            created_at=review.get("created_at") if isinstance(review.get("created_at"), dt.datetime) else dt.datetime.fromisoformat(review["created_at"]) if review.get("created_at") else dt.datetime.now(dt.timezone.utc),
        )
        session.add(rec)
        session.commit()


def load_flashcards_for_job(db_url: str, user_id: str, job_id: str) -> list[Flashcard]:
    init_flashcard_db(db_url)
    with _session() as session:
        stmt = select(Flashcard).where(Flashcard.user_id == user_id, Flashcard.job_id == job_id)
        return session.execute(stmt).scalars().all()


def _dict_to_model(item: dict) -> Flashcard:
    card = Flashcard(card_id=item.get("card_id"))
    card_id_val = item.get("id")
    if card_id_val is not None:
        card.id = card_id_val
    card.user_id = item.get("user_id")
    card.job_id = item.get("job_id")
    card.front = item.get("front", "")
    card.back = item.get("back", "")
    card.deck_id = item.get("deck_id")
    card.notes = item.get("notes")
    card.back_image = item.get("back_image")
    card.back_image_height = item.get("back_image_height")
    card.back_image_width = item.get("back_image_width")
    card.back_image_size_bytes = item.get("back_image_size_bytes")
    card.back_image_original_size_bytes = item.get("back_image_original_size_bytes")
    card.back_image_was_optimized = item.get("back_image_was_optimized")
    card.source_doc_id = item.get("source_doc_id")
    card.tags = item.get("tags") or []
    card.difficulty = item.get("difficulty")
    card.kind = item.get("kind", "new")
    card.status = item.get("status", "learning")
    card.learning_step_index = item.get("learning_step_index", 0)
    card.repetition = item.get("repetition", 0)
    card.interval_days = item.get("interval_days", 0)
    card.ease_factor = item.get("ease_factor", 2.5)
    due_at = item.get("due_at")
    card.due_at = due_at if isinstance(due_at, dt.datetime) else dt.datetime.fromisoformat(due_at) if due_at else None
    first_seen_at = item.get("first_seen_at")
    card.first_seen_at = first_seen_at if isinstance(first_seen_at, dt.datetime) else dt.datetime.fromisoformat(first_seen_at) if first_seen_at else None
    return card


def _summary_dict_to_model(item: dict) -> SummaryJob:
    record = SummaryJob(job_id=item.get("job_id"), item_type="flashcards")
    record_id = item.get("id")
    if record_id is not None:
        record.id = record_id
    record.user_id = item.get("user_id")
    record.summary = _clamp_summary(item.get("summary"))
    now = dt.datetime.now(dt.timezone.utc)
    if record_id is None:
        record.created_at = now
    record.updated_at = now
    return record
