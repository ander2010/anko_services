from __future__ import annotations

from threading import RLock
from typing import List, Sequence

from sentence_transformers import SentenceTransformer

from pipeline.utils.logging_config import get_logger
from pipeline.utils.types import ChunkCandidate, ChunkEmbedding

logger = get_logger(__name__)


class Chunkvectorizer:
    """Encodes chunk text into embeddings."""

    _MODEL_CACHE: dict[str, SentenceTransformer] = {}
    _MODEL_LOCKS: dict[str, RLock] = {}

    @classmethod
    def _model_lock(cls, model_name: str) -> RLock:
        lock = cls._MODEL_LOCKS.get(model_name)
        if lock is None:
            lock = RLock()
            cls._MODEL_LOCKS[model_name] = lock
        return lock

    @classmethod
    def preload(cls, model_name: str) -> SentenceTransformer:
        """Load a model into the per-process cache ahead of task execution."""
        with cls._model_lock(model_name):
            if model_name not in cls._MODEL_CACHE:
                cls._MODEL_CACHE[model_name] = SentenceTransformer(model_name)
                logger.info("Preloaded embedding model %s", model_name)
            else:
                logger.info("Embedding model %s already preloaded", model_name)
            return cls._MODEL_CACHE[model_name]

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._lock = self._model_lock(model_name)
        with self._lock:
            if model_name in self._MODEL_CACHE:
                self._model = self._MODEL_CACHE[model_name]
                logger.info("Reusing cached embedding model %s", model_name)
            else:
                self._model = SentenceTransformer(model_name)
                self._MODEL_CACHE[model_name] = self._model
                logger.info("Loaded embedding model %s", model_name)

    def vectorize(self, chunks: Sequence[ChunkCandidate]) -> List[ChunkEmbedding]:
        if not chunks:
            return []

        with self._lock:
            vectors = self._model.encode([chunk.text for chunk in chunks], convert_to_numpy=True, normalize_embeddings=True)
        return [
            ChunkEmbedding(text=chunk.text, embedding=vector.tolist(), metadata=getattr(chunk, "metadata", None) or {})
            for chunk, vector in zip(chunks, vectors)
        ]

    def encode_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        with self._lock:
            vectors = self._model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]
