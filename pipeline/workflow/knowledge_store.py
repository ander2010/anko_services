from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

from pipeline.utils.logging_config import get_logger
from pipeline.db.storage import open_store
from pipeline.utils.types import ChunkEmbedding

logger = get_logger(__name__)


@dataclass
class VectorEmbeddingsIndex:
    store: Any

    def save(self, document_id: str, chunk_embeddings: Sequence[ChunkEmbedding]) -> None:
        self.store.store_chunks(document_id, chunk_embeddings)

    def load(self, document_id: str) -> List[ChunkEmbedding]:
        return self.store.load_chunks(document_id)


@dataclass
class MetadataIndex:
    store: Any

    def save(self, document_id: str, qa_pairs: Sequence[dict], *, job_id: str | None = None) -> None:
        self.store.store_qa_pairs(document_id, qa_pairs, job_id=job_id)

    def load(self, document_id: str) -> List[dict]:
        return self.store.load_qa_pairs(document_id)


class ImportanceFilter:
    """Simple importance-based filtering helper."""

    def filter_embeddings(self, embeddings: Sequence[ChunkEmbedding], min_importance: float | None = None) -> List[ChunkEmbedding]:
        if min_importance is None:
            return list(embeddings)

        filtered: List[ChunkEmbedding] = []
        for embedding in embeddings:
            importance_val = embedding.metadata.get("importance") if getattr(embedding, "metadata", None) else None
            if importance_val is None:
                filtered.append(embedding)
                continue
            try:
                importance = float(importance_val)
            except (TypeError, ValueError):
                filtered.append(embedding)
                continue
            if importance >= min_importance:
                filtered.append(embedding)
        return filtered


class LocalKnowledgeStore:
    """Concrete knowledge store backed by SQLite. Acts as the local RAII / RAG repository for embeddings and metadata while exposing hooks for future remote stores."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path) if not str(db_path).startswith("postgres") else db_path
        self._store_cm = None
        self._store: Any = None
        self.vector_embeddings: VectorEmbeddingsIndex | None = None
        self.metadata_index: MetadataIndex | None = None
        self.importance_filter = ImportanceFilter()

    def __enter__(self) -> "LocalKnowledgeStore":
        self._store_cm = open_store(self.db_path)
        self._store = self._store_cm.__enter__()
        self.vector_embeddings = VectorEmbeddingsIndex(self._store)
        self.metadata_index = MetadataIndex(self._store)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._store_cm is not None:
            self._store_cm.__exit__(exc_type, exc, tb)
        self._store_cm = None
        self._store = None
        self.vector_embeddings = None
        self.metadata_index = None

    def _require_store(self) -> VectorStore:
        if self._store is None:
            raise RuntimeError("KnowledgeStore must be used within a context manager.")
        return self._store

    @staticmethod
    def _normalize_doc_id(document_id: str | int) -> int:
        try:
            return int(document_id)
        except Exception:
            raise ValueError(f"Invalid document_id; expected int, got {document_id!r}")

    def document_exists(self, document_id: str | int) -> bool:
        return self._require_store().document_exists(self._normalize_doc_id(document_id))

    def load_document(self, document_id: str) -> Tuple[List[ChunkEmbedding], List[dict]]:
        if not self.vector_embeddings or not self.metadata_index:
            raise RuntimeError("KnowledgeStore is not initialized.")
        return (self.vector_embeddings.load(document_id), self.metadata_index.load(document_id))

    def save_document(self, document_id: str | int, source_path: str, chunk_embeddings: Sequence[ChunkEmbedding], qa_pairs: Sequence[dict], *, allow_overwrite: bool = True, job_id: str | None = None) -> None:
        normalized_id = self._normalize_doc_id(document_id)
        logger.info("Saving document_id=%s chunks=%s qa_pairs=%s overwrite=%s", normalized_id, len(chunk_embeddings), len(qa_pairs), allow_overwrite)
        store = self._require_store()
        if not allow_overwrite and store.document_exists(normalized_id):
            raise ValueError(f"Document id '{normalized_id}' already exists; choose a new id or set allow_overwrite=True.")
        if not store.document_exists(normalized_id):
            raise ValueError(f"Document id '{normalized_id}' not found; cannot process.")
        store.upsert_document(normalized_id, source_path, job_id=job_id)
        if not self.vector_embeddings or not self.metadata_index:
            raise RuntimeError("KnowledgeStore is not initialized.")
        self.vector_embeddings.save(normalized_id, chunk_embeddings)
        self.metadata_index.save(normalized_id, qa_pairs, job_id=job_id)

    def append_chunks(self, document_id: str | int, source_path: str, chunk_embeddings: Sequence[ChunkEmbedding]) -> None:
        """Append chunk embeddings to an existing document without rewriting existing chunks."""
        store = self._require_store()
        normalized_id = self._normalize_doc_id(document_id)
        if not store.document_exists(normalized_id):
            raise ValueError(f"Document id '{normalized_id}' not found; cannot process.")
        store.upsert_document(normalized_id, source_path)
        append_method = getattr(store, "append_chunks", None)
        if not append_method:
            raise RuntimeError("append_chunks not supported by this store")
        append_method(normalized_id, chunk_embeddings)

    def count_chunks(self, document_id: str | int) -> int:
        store = self._require_store()
        counter = getattr(store, "count_chunks", None)
        if not counter:
            return 0
        try:
            return int(counter(self._normalize_doc_id(document_id)))
        except Exception:
            return 0

    def update_chunk_question_ids(self, document_id: str, updates: dict[str, list]) -> None:
        """Merge question IDs into chunk metadata without rewriting all embeddings."""
        store = self._require_store()
        store.update_chunk_question_ids(document_id, updates)

    def save_notification(self, job_id: str, metadata: dict) -> None:
        store = self._require_store()
        store.upsert_notification(job_id, metadata)

    def save_notifications(self, items: Sequence[tuple[str, dict]]) -> None:
        """Persist multiple notifications in one connection/transaction."""
        store = self._require_store()
        batch_method = getattr(store, "upsert_notifications", None)
        if batch_method:
            batch_method(items)
        else:
            for job_id, metadata in items:
                store.upsert_notification(job_id, metadata)

    def save_tags(self, document_id: str, tags: Sequence[str], job_id: str | None = None) -> None:
        store = self._require_store()
        store.store_tags(self._normalize_doc_id(document_id), tags, job_id=job_id)

    def save_conversation_message(self, session_id: str, user_id: str | None, job_id: str | None, question: str, answer: str) -> None:
        store = self._require_store()
        store.store_conversation_message(session_id, user_id, job_id, question, answer)

    def load_notification(self, job_id: str) -> dict | None:
        store = self._require_store()
        return store.load_notification(job_id)

    def filter_embeddings_by_importance(self, embeddings: Sequence[ChunkEmbedding], min_importance: float | None) -> List[ChunkEmbedding]:
        return self.importance_filter.filter_embeddings(embeddings, min_importance)

    def query_similar_chunks(self, query_embedding: Sequence[float], *, document_ids: str | Sequence[str] | None = None, tags: Sequence[str] | None = None, min_importance: float | None = None, top_k: int = 5):
        store = self._require_store()
        doc_ids: list[str] = []
        if document_ids is None:
            doc_ids = []
        elif isinstance(document_ids, str):
            doc_ids = [document_ids]
        else:
            doc_ids = [str(doc).strip() for doc in document_ids if str(doc).strip()]

        results: list[tuple] = []
        if not doc_ids:
            results.extend(store.query_similar_chunks(query_embedding, document_ids=None, tags=tags, min_importance=min_importance, top_k=top_k))
        else:
            for doc in doc_ids:
                results.extend(store.query_similar_chunks(query_embedding, document_ids=doc, tags=tags, min_importance=min_importance, top_k=top_k))
        return results

    def find_question_by_id(self, question_id: str) -> tuple[str, dict] | None:
        """Return (document_id, qa_dict) for a given question_id, or None if not found."""
        store = self._require_store()
        finder = getattr(store, "find_qa_by_question_id", None)
        if not finder:
            return None
        return finder(question_id)
