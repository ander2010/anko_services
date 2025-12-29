from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from celery_app import celery_app  # type: ignore
from workflow.chunking import Chunker
from workflow.importance import ImportanceScorer
from workflow.llm import LLMImportanceClient
from pipeline.logging_config import get_logger
from pipeline.metadata import MetadataEnricher
from pipeline.normalization import TextNormalizer
from workflow.progress import emit_progress  # type: ignore
from pipeline.types import OCRPageResult, ChunkEmbedding
from pipeline.workflow_core import Chunkvectorizer

logger = get_logger(__name__)


_VECTORIZER_CACHE: dict[str, Chunkvectorizer] = {}


def get_vectorizer(model_name: str) -> Chunkvectorizer:
    """Reuse embedding models across tasks to avoid reloading per task."""
    if model_name not in _VECTORIZER_CACHE:
        _VECTORIZER_CACHE[model_name] = Chunkvectorizer(model_name)
    return _VECTORIZER_CACHE[model_name]


# Preload the default model once on worker startup.
_ = get_vectorizer("sentence-transformers/all-MiniLM-L6-v2")


def deserialize_sections(sections: List[dict]) -> List[OCRPageResult]:
    """Deserialize a list of section-like items (dicts or OCRPageResult objects) into OCRPageResult."""
    out: List[OCRPageResult] = []
    for item in sections or []:
        if hasattr(item, "get"):
            page_val = item.get("page", 0)
            raw_val = item.get("raw_text", "")
            cleaned_val = item.get("cleaned_text", item.get("raw_text", ""))
            conf_val = item.get("confidence", 0.0)
        else:
            page_val = getattr(item, "page", 0)
            raw_val = getattr(item, "raw_text", "")
            cleaned_val = getattr(item, "cleaned_text", getattr(item, "raw_text", ""))
            conf_val = getattr(item, "confidence", 0.0)

        out.append(
            OCRPageResult(
                page=int(page_val),
                raw_text=str(raw_val),
                cleaned_text=str(cleaned_val),
                confidence=float(conf_val or 0.0),
            )
        )
    return out


@celery_app.task(name="pipeline.embedding-compute")
def embedding_task(payload: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    """Compute embeddings (and upstream chunking) for a PDF using Celery."""
    path = Path(payload.get("file_path") or payload.get("file path") or "")
    job_id = payload.get("job_id") or settings.get("job_id")
    doc_id = payload.get("doc_id") or settings.get("document_id") or path.stem

    logger.info("Generating embeddings for %s", path)

    sections = deserialize_sections(payload.get("sections", []))
    emit_progress(job_id=job_id, doc_id=doc_id, progress=40, step_progress=0, status="CHUNKING", current_step="chunking", extra={"pages": len(sections)})

    normalizer = TextNormalizer()
    paragraphs = normalizer.segment_into_paragraphs(sections, min_chars=int(settings.get("min_paragraph_chars", 40)))

    chunker = Chunker()
    chunk_candidates = chunker.adaptive_chunk_paragraphs(paragraphs, max_tokens=int(settings.get("max_chunk_tokens", 220)), overlap=int(settings.get("chunk_overlap", 40)))
    chunk_candidates = chunker.enforce_chunk_quality(chunk_candidates, min_tokens=int(settings.get("min_chunk_tokens", 40)), max_tokens=int(settings.get("max_chunk_tokens", 220)))

    total_candidates = len(chunk_candidates) or 1
    for idx, chunk in enumerate(chunk_candidates, 1):
        step_pct = round((idx / total_candidates) * 100, 2)
        overall = round(40.0 + (step_pct / 100.0) * 20.0, 2)  # chunking spans 40->60
        emit_progress(job_id=job_id, doc_id=doc_id, progress=overall, step_progress=step_pct, status="CHUNKING", current_step="chunking", extra={"chunk_index": idx, "total_chunks": total_candidates})

    llm_client: Optional[Any] = None
    try:
        llm_client = LLMImportanceClient(api_key=settings.get("openai_api_key"), model=settings.get("openai_model", "gpt-4o-mini"))
    except Exception:
        logger.debug("LLMImportanceClient unavailable; continuing without it")

    scorer = ImportanceScorer(relevance_threshold=float(settings.get("importance_threshold", 0.4)), llm_client=llm_client)
    scored_chunks = scorer.score_chunks(chunk_candidates)
    scored_chunks = [c for c in scored_chunks if c.relevance]

    page_confidence = payload.get("page_confidence") or {}
    enriched_chunks = MetadataEnricher().enrich(scored_chunks, page_confidence)

    max_chunks = settings.get("max_chunks")
    if max_chunks is not None:
        try:
            max_chunks_int = int(max_chunks)
            enriched_chunks = enriched_chunks[:max_chunks_int]
        except (TypeError, ValueError):
            logger.warning("Invalid max_chunks: %r", max_chunks)

    model_name = settings.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    vectorizer = get_vectorizer(model_name)

    total_embeddings = len(enriched_chunks) or 1
    for idx, _ in enumerate(enriched_chunks, 1):
        step_pct = round((idx / total_embeddings) * 100, 2)
        overall = round(60.0 + (step_pct / 100.0) * 20.0, 2)  # embedding spans 60->80
        emit_progress(job_id=job_id, doc_id=doc_id, progress=overall, step_progress=step_pct, status="EMBEDDING", current_step="embedding", extra={"chunk_index": idx, "total_chunks": total_embeddings})

    embeddings = vectorizer.vectorize([ec for ec in enriched_chunks])

    # Final progress update
    emit_progress(job_id=job_id, doc_id=doc_id, progress=80, step_progress=100, status="EMBEDDING", current_step="embedding", extra={"total_chunks": total_embeddings})

    return {
        "enriched_chunks": [asdict(ec) for ec in enriched_chunks],
        "embeddings": [asdict(ChunkEmbedding(text=e.text, embedding=list(e.embedding), metadata=e.metadata or {})) for e in embeddings],
        "job_id": job_id,
        "document_id": doc_id,
        "file_path": str(path),
        "page_confidence": page_confidence,
    }
