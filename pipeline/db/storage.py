from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np

from pipeline.db.store import SQLAlchemyStore
from pipeline.utils.logging_config import get_logger
from pipeline.workflow.postgres_storage import PostgresVectorStore
from pipeline.utils.types import ChunkEmbedding

logger = get_logger(__name__)


@dataclass
class VectorStore:
    """Simple wrapper managing connection lifecycle and schema creation."""

    db_path: Path

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._install_schema()
        # In-memory cache of normalized vectors keyed by document_id to avoid repeated JSON decoding on proximity queries.
        self._chunk_cache: dict[str, List[tuple]] = {}

    def close(self) -> None:
        self._conn.close()

    def _install_schema(self) -> None:
        cursor = self._conn.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id INTEGER PRIMARY KEY,
                source_path TEXT,
                job_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chunks (
                document_id INTEGER,
                chunk_index INTEGER,
                chunk_id TEXT,
                text TEXT,
                embedding TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (document_id, chunk_index),
                FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS qa_pairs (
                document_id INTEGER,
                qa_index INTEGER,
                question TEXT,
                correct_response TEXT,
                context TEXT,
                metadata TEXT DEFAULT '{}',
                job_id TEXT,
                chunk_id TEXT,
                chunk_index INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (document_id, job_id),
                FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS notifications (
                job_id TEXT PRIMARY KEY,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                job_id TEXT,
                title TEXT,
                content TEXT,
                "order" INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
            );
            """
        )
        self._conn.commit()
        self._ensure_column("chunks", "metadata", "TEXT DEFAULT '{}' ")
        self._ensure_column("chunks", "chunk_id", "TEXT")
        self._ensure_column("chunks", "question_ids", "TEXT DEFAULT '[]'")
        self._ensure_column("chunks", "created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        self._ensure_column("chunks", "updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        self._conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id)")
        self._ensure_column("qa_pairs", "metadata", "TEXT DEFAULT '{}'")
        self._ensure_column("qa_pairs", "job_id", "TEXT")
        self._ensure_column("qa_pairs", "chunk_id", "TEXT")
        self._ensure_column("qa_pairs", "chunk_index", "INTEGER")
        self._ensure_column("qa_pairs", "created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        self._ensure_column("qa_pairs", "updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        self._ensure_column("notifications", "metadata", "TEXT DEFAULT '{}'")
        self._ensure_column("notifications", "created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        self._ensure_column("notifications", "updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        self._ensure_column("sections", "job_id", "TEXT")
        self._ensure_column("sections", "created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        self._ensure_column("sections", "updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT,
                job_id TEXT,
                question TEXT,
                answer TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_conversation_session ON conversation_messages(session_id)")
        self._ensure_column("documents", "job_id", "TEXT")

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        cursor = self._conn.execute(f"PRAGMA table_info({table})")
        if column in {row[1] for row in cursor.fetchall()}:
            return
        self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        self._conn.commit()

    # Document helpers
    def document_exists(self, document_id: str) -> bool:
        cursor = self._conn.execute("SELECT 1 FROM documents WHERE document_id = ?", (document_id,))
        return cursor.fetchone() is not None

    def upsert_document(self, document_id: str, source_path: str, job_id: str | None = None) -> None:
        if not self.document_exists(document_id):
            raise ValueError(f"Document {document_id} not found; cannot upsert.")
        if job_id is None:
            return
        self._conn.execute(
            """
            UPDATE documents
            SET job_id = COALESCE(?, job_id)
            WHERE document_id = ?
            """,
            (job_id, document_id),
        )
        self._conn.commit()

    # Chunk helpers
    def store_chunks(self, document_id: str, chunk_vectors: Sequence[ChunkEmbedding]) -> None:
        def build_chunk_id(idx: int, chunk: ChunkEmbedding) -> str:
            # Deterministic identifier for a chunk derived from document, text, page, and tags.
            metadata = chunk.metadata or {}
            tags = metadata.get("tags") or []
            page = metadata.get("page") or ""
            normalized_text = (chunk.text or "").strip()
            payload = f"{document_id}|{idx}|{page}|{','.join(sorted(map(str, tags)))}|{normalized_text}"
            return hashlib.sha1(payload.encode("utf-8")).hexdigest()

        logger.info("Storing %s chunks for %s", len(chunk_vectors), document_id)
        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            inserts: list = []
            cache_entries: List[tuple] = []
            for idx, chunk in enumerate(chunk_vectors):
                chunk_id = build_chunk_id(idx, chunk)
                metadata = {**(chunk.metadata or {})}
                metadata.setdefault("chunk_id", chunk_id)
                question_ids = metadata.get("question_ids", [])
                if "question_ids" not in metadata:
                    metadata["question_ids"] = question_ids
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
                vector = np.array(chunk.embedding, dtype=float)
                norm = np.linalg.norm(vector)
                normed_vector = vector / norm if vector.size and not np.isclose(norm, 0.0) else None
                cache_entries.append((idx, chunk.text, metadata, normed_vector, vector.tolist()))
            self._conn.executemany(
                "INSERT INTO chunks (document_id, chunk_index, chunk_id, text, embedding, metadata, question_ids) VALUES (?, ?, ?, ?, ?, ?, ?)",
                inserts,
            )
            # Refresh cache with normalized vectors for faster future queries.
            self._chunk_cache[document_id] = cache_entries

    def update_chunk_question_ids(self, document_id: str, updates: dict[str, list]) -> None:
        """Merge question_ids into specific chunks without rewriting all chunks."""
        if not updates:
            return
        cursor = self._conn.cursor()
        for chunk_id, additions in updates.items():
            if not additions:
                continue
            cursor.execute("SELECT question_ids FROM chunks WHERE document_id = ? AND chunk_id = ?", (document_id, chunk_id))
            row = cursor.fetchone()
            existing_ids: list = []
            if row and row[0]:
                try:
                    existing_ids = json.loads(row[0]) or []
                except json.JSONDecodeError:
                    existing_ids = []
            merged = list(existing_ids)
            for item in additions:
                if item not in merged:
                    merged.append(item)
            cursor.execute(
                "UPDATE chunks SET question_ids = ?, updated_at = CURRENT_TIMESTAMP WHERE document_id = ? AND chunk_id = ?",
                (json.dumps(merged, ensure_ascii=False), document_id, chunk_id),
            )
        self._conn.commit()

    def load_chunks(self, document_id: str) -> List[ChunkEmbedding]:
        cursor = self._conn.execute(
            """
            SELECT chunk_index, text, embedding, metadata, question_ids
            FROM chunks
            WHERE document_id = ?
            ORDER BY chunk_index ASC
            """,
            (document_id,),
        )
        rows = cursor.fetchall()
        chunk_embeddings: List[ChunkEmbedding] = []
        for chunk_index, text, embedding_json, metadata_json, question_ids_json in rows:
            metadata = json.loads(metadata_json) if metadata_json else {}
            try:
                question_ids = json.loads(question_ids_json) if question_ids_json else []
            except json.JSONDecodeError:
                question_ids = []
            if question_ids and isinstance(metadata, dict) and "question_ids" not in metadata:
                metadata["question_ids"] = question_ids
            if isinstance(metadata, dict):
                metadata.setdefault("chunk_index", chunk_index)
            chunk_embeddings.append(ChunkEmbedding(text=text, embedding=json.loads(embedding_json), metadata=metadata))
        return chunk_embeddings

    # Query helpers
    def query_similar_chunks(self, query_embedding: Sequence[float], *, document_ids: str | Sequence[str] | None = None, tags: Sequence[str] | None = None, min_importance: float | None = None, top_k: int = 5) -> List[Tuple[str, int, ChunkEmbedding, float]]:
        """Return the most similar chunks ranked by cosine proximity."""
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

        if len(doc_list) == 1 and doc_list[0] in self._chunk_cache:
            doc_id = doc_list[0]
            cached_entries = self._chunk_cache[doc_id]
            for idx, text, metadata, normed_vector, raw_vector in cached_entries:
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
        else:
            sql = "SELECT document_id, chunk_index, text, embedding, metadata, question_ids FROM chunks"
            params: list = []
            if doc_list:
                placeholders = ",".join("?" for _ in doc_list)
                sql += f" WHERE document_id IN ({placeholders})"
                params.extend(doc_list)
            cursor = self._conn.execute(sql, params)
            cache_entries: List[tuple] | None = [] if len(doc_list) == 1 else None
            for doc_id, idx, text, embedding_json, metadata_json, question_ids_json in cursor.fetchall():
                metadata = json.loads(metadata_json) if metadata_json else {}
                metadata.setdefault("chunk_index", idx)
                try:
                    question_ids = json.loads(question_ids_json) if question_ids_json else []
                except json.JSONDecodeError:
                    question_ids = []
                if question_ids and "question_ids" not in metadata:
                    metadata["question_ids"] = question_ids
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
                raw_vector = np.array(json.loads(embedding_json), dtype=float)
                chunk_norm = np.linalg.norm(raw_vector)
                if raw_vector.size == 0 or np.isclose(chunk_norm, 0.0):
                    continue
                normed_vector = raw_vector / chunk_norm
                similarity = float(np.clip(np.dot(query_unit, normed_vector), -1.0, 1.0))
                chunk = ChunkEmbedding(text=text, embedding=raw_vector.tolist(), metadata=metadata)
                results.append((doc_id, idx, chunk, similarity))
                if cache_entries is not None:
                    cache_entries.append((idx, text, metadata, normed_vector, raw_vector.tolist()))
            if cache_entries is not None and doc_list:
                self._chunk_cache[doc_list[0]] = cache_entries

        results.sort(key=lambda item: item[3], reverse=True)
        top_results = results[:top_k]
        doc_label = doc_list if doc_list else "ALL"
        logger.info("Query similar chunks for doc=%s tags=%s min_importance=%s -> %s hits", doc_label, tags, min_importance, len(top_results))
        return top_results

    # QA helpers
    def store_qa_pairs(self, document_id: str, qa_pairs: Sequence[dict], *, job_id: str | None = None) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM qa_pairs WHERE job_id = ?", (job_id,))
            logger.info("Storing %s QA pairs for %s under job_id=%s", len(qa_pairs), document_id, job_id)
            self._conn.executemany(
                """
                INSERT INTO qa_pairs (document_id, qa_index, question, correct_response, context, metadata, job_id, chunk_id, chunk_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        cursor = self._conn.execute(
            """
            SELECT qa_index, question, correct_response, context, metadata, job_id, chunk_id, chunk_index
            FROM qa_pairs
            WHERE document_id = ?
            ORDER BY qa_index ASC
            """,
            (document_id,),
        )
        rows = cursor.fetchall()
        return [
            {
                "section": idx + 1,
                "question": question,
                "correct_response": correct_response,
                "context": context,
                "metadata": json.loads(metadata_json) if metadata_json else {},
                "job_id": job_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
            }
            for idx, question, correct_response, context, metadata_json, job_id, chunk_id, chunk_index in rows
        ]

    # Utility
    def list_documents(self) -> List[Tuple[str, str]]:
        cursor = self._conn.execute("SELECT document_id, source_path FROM documents ORDER BY created_at DESC")
        return cursor.fetchall()

    # Notifications / tags
    def upsert_notification(self, job_id: str, metadata: dict) -> None:
        if not job_id:
            return
        payload = json.dumps(metadata or {}, ensure_ascii=False)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO notifications (job_id, metadata, created_at, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(job_id) DO UPDATE SET metadata = excluded.metadata, updated_at = CURRENT_TIMESTAMP
                """,
                (job_id, payload),
            )

    def upsert_notifications(self, items: Sequence[tuple[str, dict]]) -> None:
        if not items:
            return
        deduped = [(job_id, json.dumps(metadata or {}, ensure_ascii=False)) for job_id, metadata in items if job_id]
        if not deduped:
            return
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO notifications (job_id, metadata, created_at, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(job_id) DO UPDATE SET metadata = excluded.metadata, updated_at = CURRENT_TIMESTAMP
                """,
                deduped,
            )

    def store_tags(self, document_id: str, tags: Sequence[str], job_id: str | None = None) -> None:
        if not document_id:
            return
        deduped = sorted({str(tag).strip() for tag in (tags or []) if str(tag).strip()})
        with self._conn:
            self._conn.execute("DELETE FROM sections WHERE document_id = ?", (document_id,))
            if deduped:
                self._conn.executemany(
                    "INSERT INTO sections (document_id, job_id, title, content, \"order\") VALUES (?, ?, ?, ?, ?)",
                    [(document_id, job_id, tag, tag, idx) for idx, tag in enumerate(deduped, start=1)],
                )

    def store_conversation_message(self, session_id: str, user_id: str | None, job_id: str | None, question: str, answer: str) -> None:
        if not session_id:
            return
        with self._conn:
            self._conn.execute(
                "INSERT INTO conversation_messages (session_id, user_id, job_id, question, answer) VALUES (?, ?, ?, ?, ?)",
                (session_id, user_id, job_id, question, answer),
            )

    def load_notification(self, job_id: str) -> dict | None:
        cursor = self._conn.execute("SELECT metadata FROM notifications WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0]) if row[0] else {}
        except json.JSONDecodeError:
            return {}

    def find_qa_by_question_id(self, question_id: str) -> tuple[str, dict] | None:
        """Lookup QA by question_id stored in metadata; returns (document_id, qa_dict) or None."""
        cursor = self._conn.execute(
            """
            SELECT document_id, qa_index, question, correct_response, context, metadata, job_id, chunk_id, chunk_index
            FROM qa_pairs
            WHERE json_extract(metadata, '$.question_id') = ?
            LIMIT 1
            """,
            (question_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        try:
            metadata = json.loads(row[5]) if row[5] else {}
        except json.JSONDecodeError:
            metadata = {}
        qa_dict = {
            "section": int(row[1]) + 1,
            "question": row[2],
            "correct_response": row[3],
            "context": row[4],
            "metadata": metadata,
            "job_id": row[6],
            "chunk_id": row[7],
            "chunk_index": row[8],
        }
        return row[0], qa_dict


def _is_postgres(path: Union[str, Path]) -> bool:
    value = str(path)
    return value.startswith("postgres://") or value.startswith("postgresql://")


@contextmanager
def open_store(db_path: Union[Path, str]):
    """Context manager ensuring the connection is closed. Supports SQLite or Postgres by passing a connection URL."""
    if _is_postgres(db_path):
        store = PostgresVectorStore(str(db_path))
    else:
        store = SQLAlchemyStore(db_path)
    try:
        yield store
    finally:
        store.close()
