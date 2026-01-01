from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import psycopg
from psycopg.rows import dict_row

from pipeline.utils.logging_config import get_logger
from pipeline.utils.types import ChunkEmbedding

logger = get_logger(__name__)


@dataclass
class PostgresVectorStore:
    dsn: str

    def __post_init__(self) -> None:
        # Autocommit simplifies lifecycle for short-lived tasks
        self._conn = psycopg.connect(self.dsn, row_factory=dict_row, autocommit=True)
        self._chunk_cache: dict[str, List[tuple]] = {}
        self._install_schema()

    def close(self) -> None:
        self._conn.close()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = %s AND column_name = %s
                """,
                (table, column),
            )
            if cur.fetchone():
                return
            cur.execute(f'ALTER TABLE {table} ADD COLUMN {column} {definition}')

    def _install_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    document_id TEXT,
                    chunk_index INTEGER,
                    chunk_id TEXT,
                    text TEXT,
                    embedding JSONB,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (document_id, chunk_index),
                    UNIQUE (chunk_id)
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS qa_pairs (
                    document_id TEXT,
                    qa_index INTEGER,
                    question TEXT,
                    correct_response TEXT,
                    context TEXT,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    job_id TEXT,
                    chunk_id TEXT,
                    chunk_index INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (document_id, qa_index)
                );
                """
            )
            self._ensure_column("chunks", "question_ids", "JSON DEFAULT '[]'::jsonb")
            self._ensure_column("qa_pairs", "job_id", "TEXT")
            self._ensure_column("qa_pairs", "chunk_id", "TEXT")
            self._ensure_column("qa_pairs", "chunk_index", "INTEGER")
            self._ensure_column("chunks", "created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            self._ensure_column("chunks", "updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            self._ensure_column("qa_pairs", "created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            self._ensure_column("qa_pairs", "updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS notifications (
                    job_id TEXT PRIMARY KEY,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS tags (
                    document_id TEXT,
                    tag TEXT,
                    PRIMARY KEY (document_id, tag),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
                );
                """
            )
            self._ensure_column("tags", "created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            self._ensure_column("tags", "updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT,
                    user_id TEXT,
                    job_id TEXT,
                    question TEXT,
                    answer TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_conversation_session ON conversation_messages(session_id)")

    # Document helpers
    def document_exists(self, document_id: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute("SELECT 1 FROM documents WHERE document_id = %s LIMIT 1", (document_id,))
            return cur.fetchone() is not None

    def upsert_document(self, document_id: str, source_path: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (document_id, source_path)
                VALUES (%s, %s)
                ON CONFLICT (document_id) DO UPDATE SET source_path = EXCLUDED.source_path
                """,
                (document_id, source_path),
            )

    # Chunk helpers
    def store_chunks(self, document_id: str, chunk_vectors: Sequence[ChunkEmbedding]) -> None:
        def build_chunk_id(idx: int, chunk: ChunkEmbedding) -> str:
            metadata = chunk.metadata or {}
            tags = metadata.get("tags") or []
            page = metadata.get("page") or ""
            normalized_text = (chunk.text or "").strip()
            payload = f"{document_id}|{idx}|{page}|{','.join(sorted(map(str, tags)))}|{normalized_text}"
            return hashlib.sha1(payload.encode("utf-8")).hexdigest()

        logger.info("Storing %s chunks for %s (Postgres)", len(chunk_vectors), document_id)
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))

            inserts: List[tuple] = []
            cache_entries: List[tuple] = []

            for idx, chunk in enumerate(chunk_vectors):
                chunk_id = build_chunk_id(idx, chunk)
                metadata = {**(chunk.metadata or {})}
                metadata.setdefault("chunk_id", chunk_id)
                question_ids = metadata.get("question_ids", [])
                if "question_ids" not in metadata:
                    metadata["question_ids"] = question_ids

                vector = np.array(chunk.embedding, dtype=float)
                norm = np.linalg.norm(vector)
                normed_vector = vector / norm if vector.size and not np.isclose(norm, 0.0) else None

                inserts.append(
                    (
                        document_id,
                        idx,
                        chunk_id,
                        chunk.text,
                        json.dumps(chunk.embedding, ensure_ascii=False),
                        json.dumps(metadata, ensure_ascii=False),
                        json.dumps(question_ids, ensure_ascii=False),
                    )
                )
                cache_entries.append((idx, chunk.text, metadata, normed_vector, vector.tolist()))

            cur.executemany(
                """
                INSERT INTO chunks (document_id, chunk_index, chunk_id, text, embedding, metadata, question_ids)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
                """,
                inserts,
            )

        self._chunk_cache[document_id] = cache_entries

    def load_chunks(self, document_id: str) -> List[ChunkEmbedding]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_index, text, embedding, metadata, question_ids
                FROM chunks
                WHERE document_id = %s
                ORDER BY chunk_index ASC
                """,
                (document_id,),
            )
            rows = cur.fetchall()

        chunk_embeddings: List[ChunkEmbedding] = []
        for row in rows:
            metadata = row["metadata"] or {}
            embedding = row["embedding"] or []
            question_ids = row.get("question_ids") or []
            if question_ids and isinstance(metadata, dict) and "question_ids" not in metadata:
                metadata["question_ids"] = question_ids
            chunk_embeddings.append(ChunkEmbedding(text=row["text"], embedding=embedding, metadata=metadata))
        return chunk_embeddings

    # Query helpers
    def query_similar_chunks(self, query_embedding: Sequence[float], *, document_ids: str | Sequence[str] | None = None, tags: Sequence[str] | None = None, min_importance: float | None = None, top_k: int = 5) -> List[Tuple[str, int, ChunkEmbedding, float]]:
        if top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        query_vec = np.array(query_embedding, dtype=float)
        query_norm = np.linalg.norm(query_vec)
        if query_vec.size == 0 or np.isclose(query_norm, 0.0):
            return []
        query_unit = query_vec / query_norm

        results: List[Tuple[str, int, ChunkEmbedding, float]] = []
        required_tags = {tag.lower() for tag in (tags or [])}
        doc_list: list[str] = []
        if document_ids is None:
            doc_list = []
        elif isinstance(document_ids, str):
            doc_list = [document_ids]
        else:
            doc_list = [str(doc).strip() for doc in document_ids if str(doc).strip()]

        cached_entries = self._chunk_cache.get(doc_list[0]) if len(doc_list) == 1 else None
        if cached_entries is not None:
            entries = [(doc_list[0], *entry) for entry in cached_entries]
        else:
            sql = "SELECT document_id, chunk_index, text, embedding, metadata, question_ids FROM chunks"
            params: list = []
            if doc_list:
                placeholders = ",".join(["%s"] * len(doc_list))
                sql += f" WHERE document_id IN ({placeholders})"
                params.extend(doc_list)
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            entries = []
            for row in rows:
                metadata = row["metadata"] or {}
                chunk_idx = row["chunk_index"]
                if isinstance(metadata, dict):
                    metadata.setdefault("chunk_index", chunk_idx)
                question_ids = row.get("question_ids") or []
                if question_ids and "question_ids" not in metadata:
                    metadata["question_ids"] = question_ids
                raw_vector = np.array(row["embedding"] or [], dtype=float)
                norm = np.linalg.norm(raw_vector)
                normed_vector = raw_vector / norm if raw_vector.size and not np.isclose(norm, 0.0) else None
                entries.append((row["document_id"], row["chunk_index"], row["text"], metadata, normed_vector, raw_vector.tolist()))

            if len(doc_list) == 1:
                self._chunk_cache[doc_list[0]] = [(idx, text, metadata, normed, raw) for _, idx, text, metadata, normed, raw in entries]

        for doc_id, idx, text, metadata, normed_vector, raw_vector in entries:
            stored_tags_raw = metadata.get("tags") or []
            try:
                stored_tags = {str(tag).lower() for tag in stored_tags_raw}
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

            if normed_vector is None:
                continue

            similarity = float(np.clip(np.dot(query_unit, normed_vector), -1.0, 1.0))
            chunk = ChunkEmbedding(text=text, embedding=raw_vector, metadata=metadata)
            results.append((doc_id, idx, chunk, similarity))

        results.sort(key=lambda item: item[3], reverse=True)
        top_results = results[:top_k]
        logger.info("Query similar chunks (Postgres) docs=%s tags=%s min_importance=%s -> %s hits", doc_list, tags, min_importance, len(top_results))
        return top_results

    # QA helpers
    def store_qa_pairs(self, document_id: str, qa_pairs: Sequence[dict], *, job_id: str | None = None) -> None:
        with self._conn.cursor() as cur:
            if job_id:
                cur.execute("DELETE FROM qa_pairs WHERE job_id = %s", (job_id,))
            else:
                cur.execute("DELETE FROM qa_pairs WHERE document_id = %s", (document_id,))
            cur.executemany(
                """
                INSERT INTO qa_pairs (document_id, qa_index, question, correct_response, context, metadata, job_id, chunk_id, chunk_index)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                """,
                [
                    (
                        document_id,
                        idx,
                        item.get("question", ""),
                        item.get("correct_response", ""),
                        item.get("context", ""),
                        json.dumps(item.get("metadata", {}), ensure_ascii=False),
                        item.get("job_id") or job_id,
                        item.get("chunk_id"),
                        item.get("chunk_index"),
                    )
                    for idx, item in enumerate(qa_pairs)
                ],
            )

    def load_qa_pairs(self, document_id: str) -> List[dict]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT qa_index, question, correct_response, context, metadata, job_id, chunk_id, chunk_index
                FROM qa_pairs
                WHERE document_id = %s
                ORDER BY qa_index ASC
                """,
                (document_id,),
            )
            rows = cur.fetchall()

        return [
            {
                "section": idx + 1,
                "question": row["question"],
                "correct_response": row["correct_response"],
                "context": row["context"],
                "metadata": row["metadata"] or {},
                "job_id": row.get("job_id"),
                "chunk_id": row.get("chunk_id"),
                "chunk_index": row.get("chunk_index"),
            }
            for idx, row in enumerate(rows)
        ]

    def update_chunk_question_ids(self, document_id: str, updates: dict[str, list]) -> None:
        """Merge question_ids into specific chunks without rewriting all chunks."""
        if not updates:
            return
        with self._conn.cursor() as cur:
            for chunk_id, additions in updates.items():
                if not additions:
                    continue
                cur.execute("SELECT question_ids FROM chunks WHERE document_id = %s AND chunk_id = %s", (document_id, chunk_id))
                row = cur.fetchone()
                existing_ids: list = []
                if row and row.get("question_ids") is not None:
                    try:
                        existing_ids = row["question_ids"] or []
                    except Exception:
                        existing_ids = []
                merged = list(existing_ids)
                for item in additions:
                    if item not in merged:
                        merged.append(item)
                cur.execute(
                    """
                    UPDATE chunks
                    SET question_ids = %s::jsonb,
                        metadata = coalesce(metadata, '{}'::jsonb) || jsonb_build_object('question_ids', %s::jsonb),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE document_id = %s AND chunk_id = %s
                    """,
                    (json.dumps(merged, ensure_ascii=False), json.dumps(merged, ensure_ascii=False), document_id, chunk_id),
                )

    # Utility
    def list_documents(self) -> List[Tuple[str, str]]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT document_id, source_path FROM documents ORDER BY created_at DESC")
            rows = cur.fetchall()
        return [(row["document_id"], row["source_path"]) for row in rows]

    # Notifications / tags
    def upsert_notification(self, job_id: str, metadata: dict) -> None:
        if not job_id:
            return
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO notifications (job_id, metadata, created_at, updated_at)
                VALUES (%s, %s::jsonb, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (job_id) DO UPDATE SET metadata = EXCLUDED.metadata, updated_at = CURRENT_TIMESTAMP
                """,
                (job_id, json.dumps(metadata or {}, ensure_ascii=False)),
            )

    def store_tags(self, document_id: str, tags: Sequence[str]) -> None:
        if not document_id:
            return
        deduped = sorted({str(tag).strip() for tag in (tags or []) if str(tag).strip()})
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM tags WHERE document_id = %s", (document_id,))
            if deduped:
                cur.executemany(
                    "INSERT INTO tags (document_id, tag) VALUES (%s, %s) ON CONFLICT (document_id, tag) DO NOTHING",
                    [(document_id, tag) for tag in deduped],
                )

    def store_conversation_message(self, session_id: str, user_id: str | None, job_id: str | None, question: str, answer: str) -> None:
        if not session_id:
            return
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversation_messages (session_id, user_id, job_id, question, answer) VALUES (%s, %s, %s, %s, %s)",
                (session_id, user_id, job_id, question, answer),
            )

    def load_notification(self, job_id: str) -> dict | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT metadata FROM notifications WHERE job_id = %s", (job_id,))
            row = cur.fetchone()
        if not row:
            return None
        try:
            return row["metadata"] or {}
        except Exception:
            return {}

    def find_qa_by_question_id(self, question_id: str) -> tuple[str, dict] | None:
        """Lookup QA by question_id stored in meta; returns (document_id, qa_dict) or None."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_id, qa_index, question, correct_response, context, metadata, job_id, chunk_id, chunk_index
                FROM qa_pairs
                WHERE metadata ->> 'question_id' = %s
                LIMIT 1
                """,
                (question_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        qa_dict = {
            "section": row["qa_index"] + 1,
            "question": row["question"],
            "correct_response": row["correct_response"],
            "context": row["context"],
            "metadata": row["metadata"] or {},
            "job_id": row.get("job_id"),
            "chunk_id": row.get("chunk_id"),
            "chunk_index": row.get("chunk_index"),
        }
        return row["document_id"], qa_dict
