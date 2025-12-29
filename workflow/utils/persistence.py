from __future__ import annotations

from typing import Sequence

from pipeline.knowledge_store import LocalKnowledgeStore
from pipeline.logging_config import get_logger
from pipeline.types import ChunkEmbedding

logger = get_logger(__name__)


def save_document(db_path, document_id: str, source_path: str, embeddings: Sequence[ChunkEmbedding], qa_pairs, *, allow_overwrite: bool = True, job_id=None):
    """Persist embeddings and QA pairs with consistent logging and error handling."""
    try:
        with LocalKnowledgeStore(db_path) as store:
            store.save_document(document_id, source_path, embeddings, qa_pairs, allow_overwrite=allow_overwrite, job_id=job_id)
            logger.info("Persisted document | job=%s doc=%s db=%s embeddings=%s qa_pairs=%s", job_id, document_id, db_path, len(embeddings), len(qa_pairs))
    except Exception:
        logger.warning("Failed to persist document | job=%s doc=%s db=%s", job_id, document_id, db_path, exc_info=True)


def update_chunk_question_ids(db_path, document_id: str, updates: dict):
    try:
        with LocalKnowledgeStore(db_path) as store:
            store.update_chunk_question_ids(document_id, updates)
    except Exception:
        logger.warning("Failed to update chunk question ids | doc=%s", document_id, exc_info=True)
