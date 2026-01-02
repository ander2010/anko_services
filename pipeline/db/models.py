from __future__ import annotations

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func, Float
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()
JSONType = SQLITE_JSON


class Document(Base):
    __tablename__ = "documents"

    document_id = Column(String, primary_key=True)
    source_path = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("chunk_id", name="uix_chunk_id"),
        UniqueConstraint("document_id", "chunk_index", name="uix_doc_chunk_index"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_id = Column(String, nullable=False)
    text = Column(Text, nullable=True)
    embedding = Column(SQLITE_JSON, nullable=False, default=list)
    meta = Column(SQLITE_JSON, nullable=False, default=dict)
    question_ids = Column(SQLITE_JSON, nullable=False, default=list)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class QAPair(Base):
    __tablename__ = "qa_pairs"
    __table_args__ = (UniqueConstraint("document_id", "qa_index", name="uix_doc_qa_index"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False)
    qa_index = Column(Integer, nullable=False)
    question = Column(Text, nullable=True)
    correct_response = Column(Text, nullable=True)
    context = Column(Text, nullable=True)
    meta = Column(SQLITE_JSON, nullable=False, default=dict)
    job_id = Column(String, nullable=True)
    chunk_id = Column(String, nullable=True)
    chunk_index = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class Flashcard(Base):
    __tablename__ = "flashcards"
    __table_args__ = (UniqueConstraint("card_id", name="uix_flashcard_card_id"),)

    card_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    job_id = Column(String, ForeignKey("flashcard_jobs.job_id", ondelete="CASCADE"), nullable=False)
    front = Column(Text, nullable=False)
    back = Column(Text, nullable=False)
    source_doc_id = Column(String, nullable=True)
    tags = Column(JSONType, nullable=False, default=list)
    difficulty = Column(String, nullable=True)
    kind = Column(String, nullable=False, default="new")
    status = Column(String, nullable=False, default="learning")
    learning_step_index = Column(Integer, nullable=False, default=0)
    repetition = Column(Integer, nullable=False, default=0)
    interval_days = Column(Integer, nullable=False, default=0)
    ease_factor = Column(Float, nullable=False, default=2.5)
    due_at = Column(DateTime, nullable=True)
    first_seen_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class FlashcardReview(Base):
    __tablename__ = "flashcard_reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    card_id = Column(String, ForeignKey("flashcards.card_id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String, nullable=False)
    job_id = Column(String, nullable=False)
    rating = Column(Integer, nullable=False)
    was_correct = Column(Integer, nullable=True)
    time_to_answer_ms = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class FlashcardJob(Base):
    __tablename__ = "flashcard_jobs"

    job_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    requested_new = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False, default="queued")
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class Notification(Base):
    __tablename__ = "notifications"

    job_id = Column(String, primary_key=True)
    meta = Column(SQLITE_JSON, nullable=False, default=dict)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class Tag(Base):
    __tablename__ = "tags"
    __table_args__ = (UniqueConstraint("document_id", "tag", name="uix_doc_tag"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False)
    tag = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    user_id = Column(String, nullable=True)
    job_id = Column(String, nullable=True)
    question = Column(Text, nullable=True)
    answer = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
