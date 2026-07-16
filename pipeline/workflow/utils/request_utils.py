from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Sequence

from pipeline.workflow.utils.request_models import AssessmentGenerationRequest, ProcessRequest, ProcessType
from pipeline.workflow.vectorizer import Chunkvectorizer

MAX_CONTEXT_CHUNKS = 15
CONTEXT_TOKEN_LIMIT = int(os.getenv("ASK_CONTEXT_TOKEN_LIMIT", "1800"))


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


def derive_assessment_job_id(request: AssessmentGenerationRequest | dict) -> str:
    if isinstance(request, AssessmentGenerationRequest):
        payload = request.to_worker_payload()
    else:
        payload = dict(request or {})

    provided_job_id = str(payload.get("job_id") or "").strip()
    if provided_job_id:
        return provided_job_id

    source_bundle = payload.get("source_bundle") or {}
    document_ids = source_bundle.get("document_ids") or payload.get("document_ids") or []
    section_ids = source_bundle.get("section_ids") or payload.get("section_ids") or []
    tag_group_ids = source_bundle.get("tag_group_ids") or payload.get("tag_group_ids") or []
    tags = source_bundle.get("tags") or payload.get("tags") or []
    query_texts_raw = payload.get("query_text")
    if isinstance(query_texts_raw, str):
        query_texts = [query_texts_raw]
    else:
        try:
            query_texts = [str(text).strip() for text in (query_texts_raw or []) if str(text).strip()]
        except TypeError:
            query_texts = []

    seed_data = {
        "process": "generate_question",
        "battery_id": payload.get("battery_id"),
        "collection_id": source_bundle.get("collection_id") or payload.get("collection_id"),
        "document_ids": sorted(str(item) for item in document_ids if str(item).strip()),
        "section_ids": sorted(str(item) for item in section_ids if str(item).strip()),
        "tag_group_ids": sorted(str(item) for item in tag_group_ids if str(item).strip()),
        "title": payload.get("title"),
        "theme": payload.get("theme"),
        "quantity_question": payload.get("quantity_question"),
        "question_format": payload.get("question_format"),
        "difficulty": payload.get("difficulty"),
        "tags": sorted(str(tag) for tag in tags if str(tag).strip()),
        "query_text": sorted(query_texts),
        "prompt_version": payload.get("prompt_version") or "v1",
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


def embed_question(question: str, model_name: str) -> list[float]:
    vectorizer = Chunkvectorizer(model_name)
    vectors = vectorizer.encode_texts([question])
    return vectors[0] if vectors else []


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

