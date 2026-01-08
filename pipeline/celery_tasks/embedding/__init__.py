from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from celery_app import celery_app  # type: ignore
from redis import Redis
from pipeline.workflow.chunking import Chunker
from pipeline.workflow.importance import ImportanceScorer
from pipeline.workflow.llm import LLMImportanceClient
from pipeline.utils.logging_config import get_logger
from pipeline.workflow.metadata import MetadataEnricher
from pipeline.workflow.utils.normalization import TextNormalizer
from pipeline.workflow.utils.progress import emit_progress, PROGRESS_REDIS_URL  # type: ignore
from pipeline.utils.types import OCRPageResult, ChunkEmbedding
from pipeline.workflow.vectorizer import Chunkvectorizer

logger = get_logger(__name__)


class EmbeddingTaskService:
    """Encapsulates chunking and embedding computation for Celery workers."""

    _vectorizer_cache: dict[str, Chunkvectorizer] = {}
    _progress_redis = None

    @classmethod
    def _get_redis(cls):
        if cls._progress_redis is None:
            cls._progress_redis = Redis.from_url(PROGRESS_REDIS_URL, decode_responses=True)
        return cls._progress_redis

    @classmethod
    def _update_units(cls, job_id: str, doc_id: str, stage: str, count: int, *, total_pages: int | None = None) -> float:
        """Update per-stage counters and return monotonic overall percent."""
        if not job_id:
            return 0.0
        try:
            CHUNK_STAGES = 3  # embed, persist, tag
            MIN_PROGRESS = 10.0  # after validate/prepare

            r = cls._get_redis()
            units_key = f"job:{job_id}:units"
            # increment totals by stage
            if stage == "embed" and count > 0:
                r.hincrby(units_key, "done_embed", count)
            elif stage == "persist" and count > 0:
                r.hincrby(units_key, "done_persist", count)
            elif stage == "tag" and count > 0:
                r.hincrby(units_key, "done_tag", count)
            elif stage == "ocr":
                if total_pages is not None:
                    try:
                        existing_tp = int(r.hget(units_key, "total_pages") or 0)
                    except Exception:
                        existing_tp = 0
                    r.hset(units_key, mapping={"total_pages": max(existing_tp, int(total_pages))})
                if count > 0:
                    r.hincrby(units_key, "done_ocr", count)

            data = r.hgetall(units_key)
            total_chunks = int(data.get("total_chunks", 0) or 0)
            total_pages_val = int(data.get("total_pages", 0) or 0)
            done_embed = int(data.get("done_embed", 0) or 0)
            done_persist = int(data.get("done_persist", 0) or 0)
            done_tag = int(data.get("done_tag", 0) or 0)
            done_ocr = int(data.get("done_ocr", 0) or 0)

            total_chunks = max(1, total_chunks)
            total_units = total_chunks * CHUNK_STAGES
            done_units = min(total_units, done_embed + done_persist + done_tag)

            # Progress driven by chunk pipeline (embed/persist/tag) from 10 -> 95.
            chunk_pct = (done_units / total_units) if total_units else 0.0
            chunk_progress = MIN_PROGRESS + chunk_pct * 85.0  # reserve last 5 for final tagging completion

            # OCR contribution to reach the initial 10% (optional).
            ocr_progress = 0.0
            if total_pages_val > 0:
                ocr_progress = min(1.0, done_ocr / total_pages_val) * MIN_PROGRESS

            progress_val = max(ocr_progress, chunk_progress)
            if done_tag >= total_chunks and total_chunks > 0:
                progress_val = 100.0

            try:
                base = float(r.hget(f"job:{job_id}:progress", "progress") or 0.0)
            except Exception:
                base = 0.0
            progress = round(min(100.0, max(base, progress_val)), 2)
            r.hset(f"job:{job_id}:progress", mapping={"progress": progress})
            return progress
        except Exception:
            return 0.0

    @classmethod
    def _stage_pct(cls, job_id: str, stage: str) -> float:
        """Return per-stage completion percent."""
        try:
            r = cls._get_redis()
            data = r.hgetall(f"job:{job_id}:units")
            total_chunks = int(data.get("total_chunks", 0) or 0)
            total_pages_val = int(data.get("total_pages", 0) or 0)
            denom = total_chunks if total_chunks > 0 else total_pages_val
            denom = max(1, denom)
            done = int(data.get(f"done_{stage}", 0) or 0)
            return round(min(100.0, max(0.0, (done / denom) * 100.0)), 2)
        except Exception:
            return 0.0

    @classmethod
    def get_vectorizer(cls, model_name: str) -> Chunkvectorizer:
        """Reuse embedding models across tasks to avoid reloading per task."""
        if model_name not in cls._vectorizer_cache:
            cls._vectorizer_cache[model_name] = Chunkvectorizer(model_name)
        return cls._vectorizer_cache[model_name]

    @staticmethod
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

    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings or {}

    def compute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Compute embeddings (and upstream chunking) for a PDF using Celery."""
        path = Path(payload.get("file_path") or payload.get("file path") or "")
        job_id = payload.get("job_id") or self.settings.get("job_id")
        doc_id = payload.get("doc_id") or self.settings.get("document_id") or path.stem
        batch_index = int(payload.get("batch_index") or 1)
        total_batches = int(payload.get("total_batches") or 1)

        logger.info("Generating embeddings for %s", path)

        sections = self.deserialize_sections(payload.get("sections", []))
        # Emit the current overall progress (do not reset to 0).
        current_overall = self._update_units(job_id or "", doc_id or "", "embed", 0)
        emit_progress(
            job_id=job_id,
            doc_id=doc_id,
            progress=current_overall,
            status="CHUNKING",
            current_step="chunking",
            extra={"pages": len(sections), "batch": batch_index, "total_batches": total_batches},
        )

        normalizer = TextNormalizer()
        paragraphs = normalizer.segment_into_paragraphs(sections, min_chars=int(self.settings.get("min_paragraph_chars", 40)))

        chunker = Chunker()
        chunk_candidates = chunker.adaptive_chunk_paragraphs(paragraphs, max_tokens=int(self.settings.get("max_chunk_tokens", 220)), overlap=int(self.settings.get("chunk_overlap", 40)))
        chunk_candidates = chunker.enforce_chunk_quality(chunk_candidates, min_tokens=int(self.settings.get("min_chunk_tokens", 40)), max_tokens=int(self.settings.get("max_chunk_tokens", 220)))

        llm_client: Optional[Any] = None
        try:
            llm_client = LLMImportanceClient(api_key=self.settings.get("openai_api_key"), model=self.settings.get("openai_model", "gpt-4o-mini"))
        except Exception:
            logger.debug("LLMImportanceClient unavailable; continuing without it")

        scorer = ImportanceScorer(relevance_threshold=float(self.settings.get("importance_threshold", 0.4)), llm_client=llm_client)
        scored_chunks = scorer.score_chunks(chunk_candidates)
        scored_chunks = [c for c in scored_chunks if c.relevance]

        page_confidence = payload.get("page_confidence") or {}
        enriched_chunks = MetadataEnricher().enrich(scored_chunks, page_confidence)

        max_chunks = self.settings.get("max_chunks")
        if max_chunks is not None:
            try:
                max_chunks_int = int(max_chunks)
                enriched_chunks = enriched_chunks[:max_chunks_int]
            except (TypeError, ValueError):
                logger.warning("Invalid max_chunks: %r", max_chunks)

        model_name = self.settings.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        vectorizer = self.get_vectorizer(model_name)

        embeddings = vectorizer.vectorize([ec for ec in enriched_chunks])

        # Record total chunks for progress once per batch.
        try:
            r = self._get_redis()
            units_key = f"job:{job_id}:units"
            r.hincrby(units_key, "total_chunks", len(enriched_chunks))
        except Exception:
            pass

        # Emit progress per chunk so users see steady growth instead of a single jump.
        last_overall = self._update_units(job_id or "", doc_id or "", "embed", 0)
        for idx, _chunk in enumerate(enriched_chunks, 1):
            last_overall = self._update_units(job_id or "", doc_id or "", "embed", 1)
            emit_progress(
                job_id=job_id,
                doc_id=doc_id,
                progress=last_overall,
                status="EMBEDDING",
                current_step="embedding",
                extra={"chunk_index": idx, "total_chunks": len(enriched_chunks), "batch": batch_index, "total_batches": total_batches},
            )

        return {
            "enriched_chunks": [asdict(ec) for ec in enriched_chunks],
            "embeddings": [asdict(ChunkEmbedding(text=e.text, embedding=list(e.embedding), metadata=e.metadata or {})) for e in embeddings],
            "job_id": job_id,
            "document_id": doc_id,
            "file_path": str(path),
            "page_confidence": page_confidence,
            "batch_index": batch_index,
            "total_batches": total_batches,
        }


# Preload the default model once on worker startup.
_ = EmbeddingTaskService.get_vectorizer("sentence-transformers/all-MiniLM-L6-v2")


@celery_app.task(name="pipeline.embedding-compute")
def embedding_task(payload: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    """Compute embeddings (and upstream chunking) for a PDF using Celery."""
    return EmbeddingTaskService(settings).compute(payload)
