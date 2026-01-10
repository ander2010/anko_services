from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from pipeline.db.models import Base, Chunk, ConversationMessage, Document, Notification, QAPair, Section
from pipeline.db.async_session import build_sqlite_async_url, create_async_engine_and_session
from pipeline.utils.logging_config import get_logger
from pipeline.utils.types import ChunkEmbedding

logger = get_logger(__name__)


class AsyncSQLAlchemyStore:
    """Async SQLAlchemy-backed persistence layer mirroring the VectorStore API."""

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.db_url = build_sqlite_async_url(self.db_path) if not str(db_path).startswith("sqlite") else str(db_path)
        self.engine, self.SessionLocal = create_async_engine_and_session(self.db_url)

    async def init_models(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    # Utilities
    @staticmethod
    def _build_chunk_id(document_id: str, idx: int, chunk: ChunkEmbedding) -> str:
        metadata = chunk.metadata or {}
        tags = metadata.get("tags") or []
        page = metadata.get("page") or ""
        normalized_text = (chunk.text or "").strip()
        payload = f"{document_id}|{idx}|{page}|{','.join(sorted(map(str, tags)))}|{normalized_text}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    # Document helpers
    async def document_exists(self, document_id: str) -> bool:
        async with self.SessionLocal() as session:
            return await session.get(Document, document_id) is not None

    async def upsert_document(self, document_id: str, source_path: str, job_id: str | None = None) -> None:
        async with self.SessionLocal() as session:
            doc = await session.get(Document, document_id)
            if not doc:
                raise ValueError(f"Document {document_id} not found; cannot upsert.")
            if job_id is not None:
                doc.job_id = job_id
            await session.commit()

    # Chunk helpers
    async def store_chunks(self, document_id: str, chunk_vectors: Sequence[ChunkEmbedding]) -> None:
        async with self.SessionLocal() as session:
            await session.execute(delete(Chunk).where(Chunk.document_id == document_id))
            chunk_rows: list[Chunk] = []
            for idx, chunk in enumerate(chunk_vectors):
                chunk_id = self._build_chunk_id(document_id, idx, chunk)
                metadata = {**(chunk.metadata or {})}
                metadata.setdefault("chunk_id", chunk_id)
                question_ids = metadata.get("question_ids", [])
                if "question_ids" not in metadata:
                    metadata["question_ids"] = question_ids
                chunk_rows.append(
                    Chunk(
                        document_id=document_id,
                        chunk_index=idx,
                        chunk_id=chunk_id,
                        text=chunk.text,
                        embedding=list(chunk.embedding),
                        meta=metadata,
                        question_ids=question_ids,
                    )
                )
            session.add_all(chunk_rows)
            await session.commit()

    async def update_chunk_question_ids(self, document_id: str, updates: dict[str, list]) -> None:
        if not updates:
            return
        async with self.SessionLocal() as session:
            for chunk_id, additions in updates.items():
                if not additions:
                    continue
                stmt = select(Chunk).where(Chunk.document_id == document_id, Chunk.chunk_id == chunk_id)
                result = await session.execute(stmt)
                chunk_row: Chunk | None = result.scalar_one_or_none()
                if not chunk_row:
                    continue
                existing = list(chunk_row.question_ids or [])
                for item in additions:
                    if item not in existing:
                        existing.append(item)
                chunk_row.question_ids = existing
                metadata = dict(chunk_row.meta or {})
                metadata["question_ids"] = existing
                chunk_row.meta = metadata
            await session.commit()

    async def load_chunks(self, document_id: str) -> List[ChunkEmbedding]:
        async with self.SessionLocal() as session:
            stmt = select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index.asc())
            rows: Iterable[Chunk] = (await session.execute(stmt)).scalars().all()
            out: List[ChunkEmbedding] = []
            for row in rows:
                metadata = row.meta or {}
                metadata.setdefault("chunk_index", row.chunk_index)
                if row.question_ids and "question_ids" not in metadata:
                    metadata["question_ids"] = row.question_ids
                out.append(ChunkEmbedding(text=row.text, embedding=list(row.embedding or []), metadata=metadata))
            return out

    # Query helpers
    async def query_similar_chunks(self, query_embedding: Sequence[float], *, document_ids: str | Sequence[str] | None = None, tags: Sequence[str] | None = None, min_importance: float | None = None, top_k: int = 5) -> List[Tuple[str, int, ChunkEmbedding, float]]:
        if top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        query_vec = np.array(query_embedding, dtype=float)
        query_norm = np.linalg.norm(query_vec)
        if query_vec.size == 0 or np.isclose(query_norm, 0.0):
            return []
        query_unit = query_vec / query_norm

        required_tags = {tag.lower() for tag in (tags or [])}
        doc_list: list[str] = []
        if document_ids is None:
            doc_list = []
        elif isinstance(document_ids, str):
            doc_list = [document_ids]
        else:
            doc_list = [str(doc).strip() for doc in document_ids if str(doc).strip()]

        async with self.SessionLocal() as session:
            stmt = select(Chunk)
            if doc_list:
                stmt = stmt.where(Chunk.document_id.in_(doc_list))
            rows: Iterable[Chunk] = (await session.execute(stmt)).scalars().all()

            results: List[Tuple[str, int, ChunkEmbedding, float]] = []
            for row in rows:
                metadata = row.metadata or {}
                try:
                    stored_tags = {str(tag).lower() for tag in (metadata.get("tags") or [])}
                except TypeError:
                    stored_tags = set()
                if required_tags and not stored_tags.intersection(required_tags):
                    continue
                if min_importance is not None:
                    importance_val = metadata.get("importance")
                    try:
                        if importance_val is None or float(importance_val) < min_importance:
                            continue
                    except (TypeError, ValueError):
                        continue
                raw_vector = np.array(row.embedding or [], dtype=float)
                chunk_norm = np.linalg.norm(raw_vector)
                if raw_vector.size == 0 or np.isclose(chunk_norm, 0.0):
                    continue
                normed_vector = raw_vector / chunk_norm
                similarity = float(np.clip(np.dot(query_unit, normed_vector), -1.0, 1.0))
                chunk_metadata = dict(metadata)
                chunk_metadata.setdefault("chunk_index", row.chunk_index)
                if row.question_ids and "question_ids" not in chunk_metadata:
                    chunk_metadata["question_ids"] = row.question_ids
                chunk = ChunkEmbedding(text=row.text, embedding=raw_vector.tolist(), metadata=chunk_metadata)
                results.append((row.document_id, row.chunk_index, chunk, similarity))

            results.sort(key=lambda item: item[3], reverse=True)
            return results[:top_k]

    # QA helpers
    async def store_qa_pairs(self, document_id: str, qa_pairs: Sequence[dict], *, job_id: str | None = None) -> None:
        async with self.SessionLocal() as session:
            if job_id:
                await session.execute(delete(QAPair).where(QAPair.job_id == job_id))
            else:
                await session.execute(delete(QAPair).where(QAPair.document_id == document_id))
            qa_rows: list[QAPair] = []
            for idx, item in enumerate(qa_pairs):
                qa_rows.append(
                    QAPair(
                        document_id=document_id,
                        qa_index=idx,
                        question=item.get("question", ""),
                        correct_response=item.get("correct_response", ""),
                        context=item.get("context", ""),
                        meta=item.get("metadata", {}),
                        job_id=item.get("job_id") or job_id,
                        chunk_id=item.get("chunk_id"),
                        chunk_index=item.get("chunk_index"),
                    )
                )
            session.add_all(qa_rows)
            await session.commit()

    async def load_qa_pairs(self, document_id: str) -> List[dict]:
        async with self.SessionLocal() as session:
            stmt = select(QAPair).where(QAPair.document_id == document_id).order_by(QAPair.qa_index.asc())
            rows: Iterable[QAPair] = (await session.execute(stmt)).scalars().all()
            return [
                {
                    "section": idx + 1,
                    "question": row.question,
                    "correct_response": row.correct_response,
                    "context": row.context,
                    "metadata": row.meta or {},
                    "job_id": row.job_id,
                    "chunk_id": row.chunk_id,
                    "chunk_index": row.chunk_index,
                }
                for idx, row in enumerate(rows)
            ]

    # Utility
    async def list_documents(self) -> List[Tuple[str, str]]:
        async with self.SessionLocal() as session:
            stmt = select(Document.document_id, Document.source_path).order_by(Document.created_at.desc())
            res = await session.execute(stmt)
            return res.all()

    # Notifications / tags
    async def upsert_notification(self, job_id: str, metadata: dict) -> None:
        if not job_id:
            return
        async with self.SessionLocal() as session:
            existing = await session.get(Notification, job_id)
            if existing:
                existing.meta = metadata or {}
            else:
                session.add(Notification(job_id=job_id, meta=metadata or {}))
            await session.commit()

    async def store_tags(self, document_id: str, tags: Sequence[str], job_id: str | None = None) -> None:
        if not document_id:
            return
        deduped = sorted({str(tag).strip() for tag in (tags or []) if str(tag).strip()})
        async with self.SessionLocal() as session:
            await session.execute(delete(Section).where(Section.document_id == document_id))
            if deduped:
                session.add_all(
                    [
                        Section(document_id=document_id, job_id=job_id, title=tag, content=tag, order=idx)
                        for idx, tag in enumerate(deduped, start=1)
                    ]
                )
            await session.commit()

    async def store_conversation_message(self, session_id: str, user_id: str | None, job_id: str | None, question: str, answer: str) -> None:
        if not session_id:
            return
        async with self.SessionLocal() as session:
            session.add(ConversationMessage(session_id=session_id, user_id=user_id, job_id=job_id, question=question, answer=answer))
            await session.commit()

    async def load_notification(self, job_id: str) -> dict | None:
        async with self.SessionLocal() as session:
            row = await session.get(Notification, job_id)
            if not row:
                return None
            return row.meta or {}
