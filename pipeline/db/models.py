from __future__ import annotations

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()


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
