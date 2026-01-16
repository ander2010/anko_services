from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Sequence

import numpy as np
from keybert import KeyBERT

from pipeline.utils.logging_config import get_logger
from pipeline.workflow.llm import LLMQuestionGenerator

logger = get_logger(__name__)


def collect_tags_from_payload(chunks: Sequence[dict] | None, embeddings: Sequence[dict] | None) -> List[str]:
    tag_set = set()
    for chunk in chunks or []:
        for tag in chunk.get("tags", []) or []:
            tag_set.add(str(tag))
    for emb in embeddings or []:
        meta = emb.get("metadata") or {}
        for tag in meta.get("tags", []) or []:
            tag_set.add(str(tag))
    return sorted(tag_set)


def _normalize_vector(vec: Sequence[float]) -> np.ndarray | None:
    arr = np.array(vec, dtype=float)
    if arr.size == 0:
        return None
    norm = np.linalg.norm(arr)
    if np.isclose(norm, 0.0):
        return None
    return arr / norm


def _sample_even(items: Sequence, limit: int) -> list:
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return list(items)
    step = max(1, len(items) // limit)
    return [items[i] for i in range(0, len(items), step)][:limit]


def filter_tags_by_embedding(tags: Sequence[str], embeddings: Sequence[dict] | None, *, top_k: int | None = None, min_cosine: float = 0.1, min_support: int = 1, max_embeddings: int = 5000) -> List[str]:
    """Score tags by how well their supporting embeddings align with the document centroid and return the best ones.

    - Builds a document centroid from all chunk embeddings.
    - For each tag, averages the normalized embeddings of chunks that contain the tag.
    - Scores tags by cosine similarity to the document centroid, boosted by log(support).
    """
    if top_k is None:
        top_k = random.randint(15, 25)
    unique_tags = [t for t in dict.fromkeys(tags or []) if t]
    if not unique_tags:
        return []

    sampled_embeddings = embeddings or []
    if max_embeddings and len(sampled_embeddings) > max_embeddings:
        sampled_embeddings = _sample_even(sampled_embeddings, max_embeddings)

    normed_embeddings: list[np.ndarray] = []
    for emb in sampled_embeddings:
        vec = emb.get("embedding") or emb.get("vector") or []
        normed = _normalize_vector(vec)
        if normed is not None:
            normed_embeddings.append(normed)
    if not normed_embeddings:
        return unique_tags[:top_k]

    doc_centroid = np.mean(normed_embeddings, axis=0)
    doc_centroid_norm = np.linalg.norm(doc_centroid)
    if np.isclose(doc_centroid_norm, 0.0):
        return unique_tags[:top_k]
    doc_centroid /= doc_centroid_norm

    tag_vectors: Dict[str, list[np.ndarray]] = {tag: [] for tag in unique_tags}
    for emb in sampled_embeddings:
        meta = emb.get("metadata") or {}
        tags_in_meta = meta.get("tags") or []
        vec = emb.get("embedding") or emb.get("vector") or []
        normed = _normalize_vector(vec)
        if normed is None or not tags_in_meta:
            continue
        for tag in tags_in_meta:
            if tag in tag_vectors:
                tag_vectors[tag].append(normed)

    scored: list[tuple[str, float]] = []
    for tag, vecs in tag_vectors.items():
        support = len(vecs)
        if support < min_support:
            continue
        tag_centroid = np.mean(vecs, axis=0)
        tag_norm = np.linalg.norm(tag_centroid)
        if np.isclose(tag_norm, 0.0):
            continue
        tag_centroid /= tag_norm
        cos = float(np.clip(np.dot(doc_centroid, tag_centroid), -1.0, 1.0))
        if cos < min_cosine:
            continue
        score = cos * (1.0 + math.log1p(support))
        scored.append((tag, score))

    if not scored:
        return unique_tags[:top_k]

    scored.sort(key=lambda item: item[1], reverse=True)
    return [tag for tag, _ in scored[:top_k]]


def extract_keywords_keybert(
    chunks: Sequence[dict] | None,
    *,
    top_n: int = 20,
    diversity: float = 0.4,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_chars: int | None = None,
    max_chunks: int | None = None,
    per_page_chars: int | None = None,
    sample_pages: int | None = None,
) -> List[str]:
    """Use KeyBERT (Ebert-style) to extract salient keywords across chunk text, bounded for large docs.

    Returns a deduped list of keywords ordered by score; falls back to empty list on failure.
    """
    max_chars = max_chars or int(os.getenv("KEYBERT_MAX_CHARS", 50000))
    max_chunks = max_chunks or int(os.getenv("KEYBERT_MAX_CHUNKS", 500))
    per_page_chars = per_page_chars or int(os.getenv("KEYBERT_PER_PAGE_CHARS", 800))
    sample_pages = sample_pages or int(os.getenv("KEYBERT_SAMPLE_PAGES", 200))

    pages: Dict[int, list[str]] = {}
    linear: list[str] = []
    for chunk in chunks or []:
        text = (chunk.get("text") or "").strip()
        if not text or len(text) < 8:
            continue
        meta = chunk.get("metadata") or {}
        page = meta.get("page")
        try:
            page_int = int(page) if page is not None else None
        except Exception:
            page_int = None
        if page_int is not None:
            pages.setdefault(page_int, [])
            if sum(len(t) for t in pages[page_int]) < per_page_chars:
                pages[page_int].append(text[:per_page_chars])
        else:
            linear.append(text)

    texts: list[str] = []
    if pages:
        sorted_pages = sorted(pages.keys())
        sampled_page_keys = _sample_even(sorted_pages, sample_pages)
        for key in sampled_page_keys:
            for snippet in pages.get(key, []):
                texts.append(snippet)
                if len(texts) >= max_chunks or sum(len(t) for t in texts) >= max_chars:
                    break
            if len(texts) >= max_chunks or sum(len(t) for t in texts) >= max_chars:
                break
    else:
        total_chars = 0
        for text in linear:
            if len(texts) >= max_chunks or total_chars >= max_chars:
                break
            snippet = text[:per_page_chars]
            texts.append(snippet)
            total_chars += len(snippet)

    if not texts:
        return []

    try:
        kw_model = KeyBERT(model=model_name)
        doc_text = "\n".join(texts)
        keywords = kw_model.extract_keywords(
            doc_text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            use_mmr=True,
            diversity=diversity,
            top_n=top_n,
        )
        return [phrase for phrase, _ in keywords if phrase]
    except Exception:
        logger.warning("KeyBERT keyword extraction failed; continuing without Ebert filter", exc_info=True)
        return []


def ensure_llm_active_warning(llm_generator: LLMQuestionGenerator | None) -> None:
    if llm_generator and not llm_generator.is_active:
        logger.warning("LLM is inactive (missing API key); tagging will use fallback.")


def infer_tags_with_llm(llm_generator: LLMQuestionGenerator | None, text: str, fallback_max: int = 5, warn: bool = True) -> List[str]:
    if llm_generator and llm_generator.is_active:
        try:
            return llm_generator.tag_text(text)
        except Exception:
            logger.warning("LLM tagging failed; falling back to heuristic", exc_info=warn)
    if warn:
        logger.warning("LLM inactive; using heuristic tags")
    return [part.strip() for part in text.split(",") if part.strip()][:fallback_max]
