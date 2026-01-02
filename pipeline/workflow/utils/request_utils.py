from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Sequence

from keybert import KeyBERT

from pipeline.workflow.utils.request_models import ProcessRequest, ProcessType
from pipeline.workflow.vectorizer import Chunkvectorizer

MAX_CONTEXT_CHUNKS = 15
CONTEXT_TOKEN_LIMIT = int(os.getenv("ASK_CONTEXT_TOKEN_LIMIT", "1800"))
_KEYBERT_CACHE: dict[str, KeyBERT] = {}


def derive_job_id(request: ProcessRequest) -> str:
    if request.job_id:
        return request.job_id

    if request.process == ProcessType.GENERATE_QUESTION:
        tags = request.tags or []
        if isinstance(tags, str):
            tags = [tags]

        query_texts_raw = request.query_text
        if isinstance(query_texts_raw, str):
            query_texts = [query_texts_raw]
        else:
            try:
                query_texts = [str(text).strip() for text in (query_texts_raw or []) if str(text).strip()]
            except TypeError:
                query_texts = []

        seed_data = {
            "process": request.process.value,
            "doc_id": request.doc_id,
            "theme": request.theme,
            "quantity_question": request.quantity_question,
            "question_format": request.question_format,
            "tags": sorted(str(tag) for tag in tags if str(tag)),
            "query_text": sorted(query_texts),
        }
        seed = json.dumps(seed_data, sort_keys=True, separators=(",", ":"))
    else:
        seed = f"{request.doc_id}:{request.process.value}"

    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def derive_variant_job_id(question_id: str, quantity: int, difficulty: str, question_format: str) -> str:
    seed_data = {
        "question_id": question_id,
        "quantity": quantity,
        "difficulty": difficulty,
        "question_format": question_format,
    }
    seed = json.dumps(seed_data, sort_keys=True, separators=(",", ":"))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def average_embedding_vectors(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    first = vectors[0] or []
    if not first:
        return []
    length = len(first)
    sums = [0.0] * length
    count = 0
    for vec in vectors:
        if len(vec) != length:
            continue
        sums = [a + float(b) for a, b in zip(sums, vec)]
        count += 1
    if count == 0:
        return []
    return [val / count for val in sums]


def _estimate_tokens(text: str | None) -> int:
    return len((text or "").split())


def trim_chunks_to_budget(chunks: list[dict], question: str, token_budget: int = CONTEXT_TOKEN_LIMIT) -> list[dict]:
    """Sort chunks by importance/similarity and keep those that fit within the token budget (including question tokens)."""
    if not chunks:
        return []
    budget = max(0, token_budget - _estimate_tokens(question))
    scored = []
    for ch in chunks:
        meta = ch.get("metadata") or {}
        importance = meta.get("importance")
        try:
            importance = float(importance) if importance is not None else None
        except (TypeError, ValueError):
            importance = None
        similarity = ch.get("similarity")
        try:
            similarity = float(similarity) if similarity is not None else None
        except (TypeError, ValueError):
            similarity = None
        score = importance if importance is not None else (similarity if similarity is not None else 0.0)
        tokens = meta.get("tokens")
        try:
            tokens = int(tokens) if tokens is not None else None
        except (TypeError, ValueError):
            tokens = None
        if tokens is None:
            tokens = _estimate_tokens(ch.get("text"))
        scored.append((score, tokens, ch))

    scored.sort(key=lambda item: item[0], reverse=True)
    kept: list[dict] = []
    used = 0
    for _score, tokens, ch in scored:
        if tokens <= 0:
            continue
        if used + tokens > budget:
            continue
        kept.append(ch)
        used += tokens
    if not kept and scored:
        kept.append(scored[0][2])
    return kept


def get_keyword_model(model_name: str) -> KeyBERT:
    if model_name not in _KEYBERT_CACHE:
        _KEYBERT_CACHE[model_name] = KeyBERT(model=model_name)
    return _KEYBERT_CACHE[model_name]


def embed_question(question: str, model_name: str) -> list[float]:
    vectorizer = Chunkvectorizer(model_name)
    kw_model = get_keyword_model(model_name)
    keywords = [kw for kw, _score in kw_model.extract_keywords(question, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=8)]
    texts = [question] + keywords
    vectors = vectorizer._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    if len(vectors) > 1:
        return average_embedding_vectors([vec.tolist() for vec in vectors if hasattr(vec, "tolist")])
    return vectors[0].tolist() if len(vectors) else []


def apply_external_options(settings, request: ProcessRequest):
    options = request.options
    if options.ocr_language:
        settings.lang = options.ocr_language
    if options.chunk_size:
        settings.max_chunk_tokens = options.chunk_size
    if options.embedding_model:
        settings.embedding_model = options.embedding_model
    if options.importance_threshold is not None:
        settings.importance_threshold = options.importance_threshold
    if options.ga_format:
        settings.qa_format = options.ga_format
    if options.max_chunks is not None:
        settings.max_chunks = options.max_chunks
    return settings


def merge_settings(base: dict, overrides: dict | None = None) -> dict:
    merged = dict(base or {})
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                merged[key] = value
    for key, value in list(merged.items()):
        if isinstance(value, Path):
            merged[key] = str(value)
    return merged

