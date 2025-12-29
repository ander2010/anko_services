from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from pipeline.logging_config import get_logger
from pipeline.storage import VectorStore, open_store
from pipeline.types import ChunkEmbedding

logger = get_logger(__name__)


@dataclass
class VectorEmbeddingsIndex:
    store: VectorStore

    def save(self, document_id: str, chunk_embeddings: Sequence[ChunkEmbedding]) -> None:
        self.store.store_chunks(document_id, chunk_embeddings)

    def load(self, document_id: str) -> List[ChunkEmbedding]:
        return self.store.load_chunks(document_id)


@dataclass
class MetadataIndex:
    store: VectorStore

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
        self._store: VectorStore | None = None
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

    def document_exists(self, document_id: str) -> bool:
        return self._require_store().document_exists(document_id)

    def load_document(self, document_id: str) -> Tuple[List[ChunkEmbedding], List[dict]]:
        if not self.vector_embeddings or not self.metadata_index:
            raise RuntimeError("KnowledgeStore is not initialized.")
        return (self.vector_embeddings.load(document_id), self.metadata_index.load(document_id))

    def save_document(self, document_id: str, source_path: str, chunk_embeddings: Sequence[ChunkEmbedding], qa_pairs: Sequence[dict], *, allow_overwrite: bool = True, job_id: str | None = None) -> None:
        logger.info("Saving document_id=%s chunks=%s qa_pairs=%s overwrite=%s", document_id, len(chunk_embeddings), len(qa_pairs), allow_overwrite)
        store = self._require_store()
        if not allow_overwrite and store.document_exists(document_id):
            raise ValueError(f"Document id '{document_id}' already exists; choose a new id or set allow_overwrite=True.")
        store.upsert_document(document_id, source_path)
        if not self.vector_embeddings or not self.metadata_index:
            raise RuntimeError("KnowledgeStore is not initialized.")
        self.vector_embeddings.save(document_id, chunk_embeddings)
        self.metadata_index.save(document_id, qa_pairs, job_id=job_id)

    def update_chunk_question_ids(self, document_id: str, updates: dict[str, list]) -> None:
        """Merge question IDs into chunk metadata without rewriting all embeddings."""
        store = self._require_store()
        store.update_chunk_question_ids(document_id, updates)

    def filter_embeddings_by_importance(self, embeddings: Sequence[ChunkEmbedding], min_importance: float | None) -> List[ChunkEmbedding]:
        return self.importance_filter.filter_embeddings(embeddings, min_importance)

    def query_similar_chunks(self, query_embedding: Sequence[float], *, document_id: str | None = None, tags: Sequence[str] | None = None, min_importance: float | None = None, top_k: int = 5):
        store = self._require_store()
        return store.query_similar_chunks(query_embedding, document_id=document_id, tags=tags, min_importance=min_importance, top_k=top_k)
