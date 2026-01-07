from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Sequence

from celery_app import celery_app  # type: ignore
from openai import OpenAI
from redis import Redis
from pipeline.workflow.knowledge_store import LocalKnowledgeStore
from pipeline.utils.logging_config import get_logger
from pipeline.utils.types import ChunkCandidate, ChunkEmbedding
from pipeline.workflow.qa import QAComposer
from pipeline.workflow.vectorizer import Chunkvectorizer
from pipeline.workflow.conversation import append_message, format_history
from pipeline.workflow.llm import LLMQuestionGenerator
from pipeline.workflow.utils.progress import emit_progress, PROGRESS_REDIS_URL
from pipeline.workflow.utils.persistence import save_conversation_message, save_document, save_notification, save_tags
from pipeline.workflow.utils.settings import normalize_settings
from pipeline.workflow.utils.tags import collect_tags_from_payload, ensure_llm_active_warning, infer_tags_with_llm


logger = get_logger(__name__)


class LLMTaskService:
    """Encapsulates all LLM-related Celery task logic."""

    def __init__(self, settings: dict):
        self.settings = normalize_settings(settings or {})
        self.db_path = self.settings.get("db_path", "hope/vector_store.db")
        self._progress_redis = None

    def _get_redis(self):
        if self._progress_redis is None:
            self._progress_redis = Redis.from_url(PROGRESS_REDIS_URL, decode_responses=True)
        return self._progress_redis

    def _update_units(self, job_id: str, doc_id: str, stage: str, count: int, *, total_pages: int | None = None) -> float:
        """Update per-stage counters (including OCR) and return overall percent."""
        if not job_id or count <= 0 and stage != "ocr":
            return 0.0
        try:
            OCR_WEIGHT = 0.5
            CHUNK_STAGES = 3

            r = self._get_redis()
            units_key = f"job:{job_id}:units"
            if stage == "embed" and count > 0:
                r.hincrby(units_key, "total_chunks", count)
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
            done_ocr = int(data.get("done_ocr", 0) or 0)
            done_embed = int(data.get("done_embed", 0) or 0)
            done_persist = int(data.get("done_persist", 0) or 0)
            done_tag = int(data.get("done_tag", 0) or 0)
            computed_total_work = total_pages_val * OCR_WEIGHT + max(total_pages_val, total_chunks, 1) * CHUNK_STAGES
            try:
                existing_total_work = float(data.get("total_work", 0) or 0)
            except Exception:
                existing_total_work = 0.0
            total_work = max(1.0, computed_total_work, existing_total_work)
            r.hset(units_key, mapping={"total_work": total_work})
            done_work = (done_ocr * OCR_WEIGHT) + done_embed + done_persist + done_tag
            raw_pct = min(100.0, max(0.0, (done_work / total_work) * 100.0))

            try:
                base = float(r.hget(f"job:{job_id}:progress", "progress") or 0.0)
            except Exception:
                base = 0.0
            progress = round(min(100.0, max(base, raw_pct)), 2)
            r.hset(f"job:{job_id}:progress", mapping={"progress": progress})
            return progress
        except Exception:
            return 0.0

    # ---------------------
    # Shared helpers
    # ---------------------
    @staticmethod
    def _deserialize_chunks(chunks: List[dict]) -> List[ChunkCandidate]:
        return [
            ChunkCandidate(
                page=int(item.get("page", 0)),
                text=item.get("text", ""),
                tokens=int(item.get("tokens", 0) or 0),
                importance=float(item.get("importance", 0.0) or 0.0),
                relevance=bool(item.get("relevance", True)),
                concept_type=item.get("concept_type", "Explanation"),
                tags=item.get("tags", []),
                difficulty=item.get("difficulty", "medium"),
                metadata=item.get("metadata", {}),
            )
            for item in chunks or []
        ]

    @staticmethod
    def _deserialize_embeddings(items: List[dict]) -> List[ChunkEmbedding]:
        return [
            ChunkEmbedding(
                text=item.get("text", ""),
                embedding=item.get("embedding", []),
                metadata=item.get("metadata", {}),
            )
            for item in items or []
        ]

    @staticmethod
    def _embeddings_to_candidates(embeddings: Sequence[ChunkEmbedding], theme: Optional[str] = None, difficulty: Optional[str] = None) -> List[ChunkCandidate]:
        candidates: List[ChunkCandidate] = []
        for emb in embeddings:
            meta = emb.metadata or {}
            tags = list(meta.get("tags") or [])
            if theme:
                tags = list(dict.fromkeys(tags + [theme]))
            candidates.append(
                ChunkCandidate(
                    page=int(meta.get("page", 0) or 0),
                    text=emb.text,
                    tokens=int(meta.get("tokens", 0) or 0) if isinstance(meta, dict) else 0,
                    importance=float(meta.get("importance", 0.0) or 0.0),
                    relevance=True,
                    concept_type=meta.get("concept_type", "Explanation"),
                    tags=tags,
                    difficulty=difficulty or meta.get("difficulty", "medium"),
                    metadata=meta,
                )
            )
        return candidates

    @staticmethod
    def _average_embedding_vectors(vectors: Sequence[Sequence[float]]) -> List[float]:
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

    @staticmethod
    def _openai_client(settings: dict) -> OpenAI:
        api_key = settings.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for LLM calls")
        return OpenAI(api_key=api_key)

    @staticmethod
    def _format_context_chunks(chunks: Sequence[dict]) -> str:
        """Build a human-readable context block grouped by document, with metadata for clarity."""
        grouped: Dict[str, list[tuple[int, dict]]] = {}
        for idx, chunk in enumerate(chunks or [], 1):
            meta = chunk.get("metadata") or {}
            doc_id = chunk.get("document_id") or meta.get("document_id") or "unknown"
            grouped.setdefault(doc_id, []).append((idx, chunk))

        sections: List[str] = []
        for doc_id, items in grouped.items():
            # Sort within a doc by importance/similarity descending
            def score(item: tuple[int, dict]) -> float:
                meta = (item[1].get("metadata") or {})
                imp = meta.get("importance")
                sim = item[1].get("similarity") or meta.get("similarity")
                try:
                    return float(imp if imp is not None else sim if sim is not None else 0.0)
                except (TypeError, ValueError):
                    return 0.0

            doc_lines: List[str] = [f"Document: {doc_id}"]
            for idx, chunk in sorted(items, key=score, reverse=True):
                text = str(chunk.get("text", "") or "").strip()
                if not text:
                    continue
                meta = chunk.get("metadata") or {}
                page = meta.get("page")
                tags = meta.get("tags") or []
                imp = meta.get("importance")
                sim = chunk.get("similarity") or meta.get("similarity")
                try:
                    score_val = float(imp if imp is not None else sim if sim is not None else 0.0)
                except (TypeError, ValueError):
                    score_val = 0.0
                meta_parts: list[str] = []
                if page is not None:
                    meta_parts.append(f"page: {page}")
                if tags:
                    meta_parts.append(f"tags: {', '.join(map(str, tags))}")
                meta_parts.append(f"score: {score_val:.3f}")
                meta_block = "metadata: " + "; ".join(meta_parts)
                doc_lines.append(f"- chunk {idx} | {meta_block}\n  context: {text}")
            sections.append("\n".join(doc_lines))
        return "\n\n".join(sections)

    # ---------------------
    # Task handlers
    # ---------------------
    def persist_document(self, payload: dict) -> dict:
        settings = self.settings
        job_id = payload.get("job_id") or settings.get("job_id")
        doc_id = payload.get("doc_id") or payload.get("document_id") or settings.get("document_id")

        qa_pairs: List[dict] = payload.get("qa_pairs") or []

        if settings.get("persist_local"):
            embeddings = self._deserialize_embeddings(payload.get("embeddings", []))
            save_document(self.db_path, doc_id or settings.get("document_id", "celery-doc"), payload.get("file_path", ""), embeddings, qa_pairs, allow_overwrite=settings.get("allow_overwrite", True), job_id=job_id)
            logger.info("Persisted   | job=%s doc=%s embeddings=%s qa_pairs=%s", job_id, doc_id, len(embeddings), len(qa_pairs))
        else:
            logger.info("Persistence skipped | job=%s doc=%s persist_local=%s", job_id, doc_id, settings.get("persist_local"))

        tags_sorted = collect_tags_from_payload(payload.get("enriched_chunks"), payload.get("embeddings"))
        extra = {
            "tags": tags_sorted,
            "tag_set": tags_sorted,
            "qa_pairs": len(qa_pairs),
            "chunks": len(payload.get("enriched_chunks", [])),
            "embeddings": len(payload.get("embeddings", [])),
        }

        emit_progress(job_id=job_id, doc_id=doc_id, progress=85, status="PERSISTED", current_step="persist", extra=extra)

        return payload

    def persist_document_batch(self, payload: dict) -> dict:
        """Append embeddings for a batch to the knowledge store without rewriting existing chunks."""
        settings = self.settings
        job_id = payload.get("job_id") or settings.get("job_id")
        doc_id = payload.get("doc_id") or payload.get("document_id") or settings.get("document_id")
        batch_index = int(payload.get("batch_index") or 1)
        total_batches = int(payload.get("total_batches") or 1)

        embeddings = self._deserialize_embeddings(payload.get("embeddings", []))
        source_path = payload.get("file_path", "")

        if not settings.get("persist_local"):
            logger.info("Persist batch skipped | job=%s doc=%s persist_local=%s", job_id, doc_id, settings.get("persist_local"))
            return {"job_id": job_id, "doc_id": doc_id, "file_path": source_path}

        if not doc_id:
            logger.warning("Persist batch skipped | missing doc_id")
            return payload

        persisted_count = 0
        try:
            with LocalKnowledgeStore(self.db_path) as store:
                store.append_chunks(doc_id, source_path, embeddings)
            persisted_count = len(embeddings)
            logger.info("Persisted batch | job=%s doc=%s embeddings=%s", job_id, doc_id, persisted_count)
        except Exception:
            logger.warning("Failed to persist batch | job=%s doc=%s", job_id, doc_id, exc_info=True)

        overall_progress = self._update_units(job_id or "", doc_id or "", "persist", persisted_count)

        emit_progress(
            job_id=job_id,
            doc_id=doc_id,
            progress=overall_progress,
            status="PERSISTED",
            current_step="persist",
            extra={"batch": batch_index, "total_batches": total_batches},
        )

        return {
            "job_id": job_id,
            "doc_id": doc_id,
            "file_path": source_path,
            "batch_index": batch_index,
            "total_batches": total_batches,
        }

    def tag_chunks(self, payload: dict) -> dict:
        settings = self.settings
        job_id = payload.get("job_id") or settings.get("job_id")
        doc_id = payload.get("doc_id") or settings.get("document_id")

        chunks = payload.get("enriched_chunks") or []
        embeddings = payload.get("embeddings") or []

        # If no chunks/embeddings provided, load from the knowledge store to support streaming pipelines.
        if (not chunks or not embeddings) and doc_id:
            try:
                with LocalKnowledgeStore(self.db_path) as ks:
                    loaded_embeddings, _ = ks.load_document(doc_id)
                embeddings = [
                    {"text": emb.text, "embedding": list(emb.embedding or []), "metadata": dict(emb.metadata or {})}
                    for emb in loaded_embeddings
                ]
                chunks = [
                    {"text": emb.text, "metadata": dict(emb.metadata or {})}
                    for emb in loaded_embeddings
                ]
                logger.info("Loaded chunks from store for tagging | doc=%s count=%s", doc_id, len(chunks))
            except Exception:
                logger.warning("Failed to load chunks from store for tagging | doc=%s", doc_id, exc_info=True)

        total_chunks = len(chunks)
        if total_chunks == 0:
            overall = self._update_units(job_id or "", doc_id or "", "tag", 0)
            emit_progress(job_id=job_id, doc_id=doc_id, progress=max(95, overall), status="TAGGING", current_step="tagging", extra={"chunk_index": 0, "total_chunks": 0})
            logger.info("Tag skipped | job=%s doc=%s reason=no_chunks", job_id, doc_id)
            return {"doc_id": doc_id, "job_id": job_id, "enriched_chunks": [], "embeddings": [], "tags": []}

        logger.info("Tag start  | job=%s doc=%s chunks=%s embeddings=%s", job_id, doc_id, len(chunks), len(embeddings))

        llm_generator = LLMQuestionGenerator(api_key=settings.get("openai_api_key"), model=settings.get("openai_model", "gpt-4o-mini"))
        ensure_llm_active_warning(llm_generator)

        inactive_logged = False
        for idx, chunk in enumerate(chunks, 1):
            text = chunk.get("text") or ""
            inferred_tags = infer_tags_with_llm(llm_generator, text, warn=False)
            if not inferred_tags and not inactive_logged:
                logger.warning("LLM tagging inactive; skipping inferred tags for remaining chunks (job=%s doc=%s)", job_id, doc_id)
                inactive_logged = True

            existing_tags = set(str(tag) for tag in (chunk.get("tags") or []))
            tags = sorted(existing_tags.union(inferred_tags))
            chunk["tags"] = tags
            meta = chunk.get("metadata") or {}
            meta["tags"] = tags
            chunk["metadata"] = meta

            # propagate tags to the matching embedding (if lengths align)
            if idx - 1 < len(embeddings):
                emb_meta = embeddings[idx - 1].get("metadata") or {}
                emb_meta["tags"] = tags
                embeddings[idx - 1]["metadata"] = emb_meta

            overall = self._update_units(job_id or "", doc_id or "", "tag", 1)
            emit_progress(job_id=job_id, doc_id=doc_id, progress=overall, status="TAGGING", current_step="tagging", extra={"chunk_index": idx, "total_chunks": total_chunks})
            logger.info("Tag progress | job=%s doc=%s chunk=%s/%s tags=%s", job_id, doc_id, idx, total_chunks, tags)

        if embeddings:
            save_document(self.db_path, doc_id or settings.get("document_id", "celery-doc"), payload.get("file_path", ""), self._deserialize_embeddings(embeddings), payload.get("qa_pairs", []), allow_overwrite=settings.get("allow_overwrite", True), job_id=job_id)

        tags_sorted = collect_tags_from_payload(chunks, embeddings)
        save_tags(self.db_path, doc_id or settings.get("document_id", "celery-doc"), tags_sorted)
        save_notification(
            self.db_path,
            job_id or "",
            {
                "status": "COMPLETED",
                "doc_id": doc_id,
                "tags": tags_sorted,
                "chunks": len(chunks),
                "embeddings": len(embeddings),
            },
        )
        emit_progress(job_id=job_id, doc_id=doc_id, progress=100, status="TAGGED", current_step="tagging", extra={"tags": tags_sorted, "chunks": len(chunks), "embeddings": len(embeddings)})
        emit_progress(job_id=job_id, doc_id=doc_id, progress=100, status="COMPLETED", current_step="done", extra={"tags": tags_sorted, "chunks": len(chunks), "embeddings": len(embeddings)})
        logger.info("Tag done    | job=%s doc=%s chunks=%s tags=%s", job_id, doc_id, len(chunks), len(tags_sorted))

        payload["enriched_chunks"] = chunks
        payload["embeddings"] = embeddings
        payload["tags"] = tags_sorted

        return payload

    def generate_questions(self, payload: dict) -> dict:
        settings = self.settings
        doc_id = payload.get("doc_id")
        if not doc_id:
            raise ValueError("doc_id is required for generate_questions_task")

        job_id = payload.get("job_id") or settings.get("job_id")
        logger.info("GenQ start | job=%s doc=%s db=%s", job_id, doc_id, self.db_path)

        with LocalKnowledgeStore(self.db_path) as knowledge_store:
            embeddings, _ = knowledge_store.load_document(doc_id)

        if not embeddings:
        emit_progress(job_id=job_id, doc_id=doc_id, progress=100, status="NO_EMBEDDINGS", current_step="load_embeddings", extra={"embeddings": 0, "process": "generate_question"})
            logger.warning("GenQ aborted| job=%s doc=%s reason=no_embeddings", job_id, doc_id)
            return {"doc_id": doc_id, "qa_pairs": [], "count": 0, "error": "no embeddings found for document"}

        emit_progress(job_id=job_id, doc_id=doc_id, progress=15, status="GENERATING_QUESTIONS", current_step="load_embeddings", extra={"embeddings": len(embeddings), "process": "generate_question"})
        logger.info("GenQ loaded| job=%s doc=%s embeddings=%s", job_id, doc_id, len(embeddings))

        query_texts_raw = payload.get("query_text")
        if isinstance(query_texts_raw, str):
            query_texts = [query_texts_raw]
        else:
            try:
                query_texts = [str(text).strip() for text in (query_texts_raw or []) if str(text).strip()]
            except TypeError:
                query_texts = []

        top_k = payload.get("top_k")
        if top_k is not None:
            try:
                top_k = max(1, int(top_k))
            except (TypeError, ValueError):
                top_k = None

        quantity = payload.get("quantity_question")
        if top_k is None and quantity:
            try:
                top_k = max(1, int(quantity))
            except (TypeError, ValueError):
                top_k = None

        if top_k is None:
            top_k = min(len(embeddings), 10) if embeddings else 5
        logger.info("GenQ select| job=%s doc=%s top_k=%s queries=%s tags=%s min_importance=%s", job_id, doc_id, top_k, query_texts, payload.get("tags"), payload.get("min_importance"))

        # If query texts provided, attempt a similarity search; otherwise pick top_k
        merged: Dict[tuple[str, int], tuple[str, int, ChunkEmbedding, float]] = {}
        try:
            if query_texts:
                vectorizer = Chunkvectorizer(settings.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
                try:
                    query_vectors = vectorizer._model.encode(query_texts, convert_to_numpy=True, normalize_embeddings=True)
                except Exception:
                    query_vectors = []
                # Run similarity queries inside a knowledge store context
                with LocalKnowledgeStore(self.db_path) as knowledge_store:
                    for query_text, vector in zip(query_texts, query_vectors or []):
                        single_results = knowledge_store.query_similar_chunks(vector.tolist(), document_ids=[doc_id], tags=payload.get("tags") or None, min_importance=payload.get("min_importance", settings.get("importance_threshold")), top_k=top_k)
                        if not single_results and payload.get("tags"):
                            single_results = knowledge_store.query_similar_chunks(vector.tolist(), document_ids=[doc_id], tags=None, min_importance=payload.get("min_importance", settings.get("importance_threshold")), top_k=top_k)
                        for docid, idx, chunk, similarity in single_results:
                            key = (docid, idx)
                            if key not in merged or similarity > merged[key][3]:
                                merged[key] = (docid, idx, chunk, similarity)
            logger.info("GenQ merged | job=%s doc=%s merged_hits=%s query_texts=%s", job_id, doc_id, len(merged), bool(query_texts))
        except Exception:
            merged = {}

        if merged:
            selected_embeddings = [item[2] for item in merged.values()]
            logger.info("GenQ source | job=%s doc=%s source=merged", job_id, doc_id)
        else:
            fallback_embedding = self._average_embedding_vectors([emb.embedding or [] for emb in embeddings])
            if fallback_embedding:
                with LocalKnowledgeStore(self.db_path) as knowledge_store:
                    results = knowledge_store.query_similar_chunks(fallback_embedding, document_ids=[doc_id], tags=None, min_importance=payload.get("min_importance", settings.get("importance_threshold")), top_k=top_k)
                selected_embeddings = [item[2] for item in results] if results else embeddings[:top_k]
                logger.info("GenQ source | job=%s doc=%s source=fallback results=%s", job_id, doc_id, len(selected_embeddings))
            else:
                selected_embeddings = embeddings[:top_k]
                logger.info("GenQ source | job=%s doc=%s source=topk", job_id, doc_id)

        emit_progress(job_id=job_id, doc_id=doc_id, progress=40, status="GENERATING_QUESTIONS", current_step="select_chunks", extra={"selected_chunks": len(selected_embeddings), "top_k": top_k, "process": "generate_question"})
        logger.info("GenQ chunks| job=%s doc=%s selected=%s top_k=%s", job_id, doc_id, len(selected_embeddings), top_k)

        candidates = self._embeddings_to_candidates(selected_embeddings, theme=payload.get("theme"), difficulty=payload.get("difficulty"))
        logger.info("GenQ cand  | job=%s doc=%s candidates=%s theme=%s difficulty=%s", job_id, doc_id, len(candidates), payload.get("theme"), payload.get("difficulty"))

        worker_count = int(settings.get("ga_workers", settings.get("qa_workers", 4)))
        ga_generator = LLMQuestionGenerator(api_key=settings.get("openai_api_key"), model=settings.get("openai_model", "gpt-4o-mini"))
        try:
            ga_composer = QAComposer(ga_generator=ga_generator, ga_workers=worker_count, theme_hint=payload.get("theme"), difficulty_hint=payload.get("difficulty"), target_questions=payload.get("quantity_question"))
        except Exception:
            ga_composer = QAComposer(ga_generator=ga_generator, ga_workers=worker_count)

        ga_progress = {"count": 0}

        def ga_progress_cb(item: Any, *_args: Any, **_kwargs: Any) -> None:
            if not job_id:
                return
            ga_progress["count"] += 1
            emit_progress(job_id=job_id, doc_id=doc_id, progress=85, status="QA_GENERATING", current_step="qa", extra={"count": ga_progress["count"]})
            logger.info("GenQ prog  | job=%s doc=%s qa_progress=%s question=%s", job_id, doc_id, ga_progress["count"], (item.get("question") if isinstance(item, dict) else None))

        qa_pairs = ga_composer.generate(candidates, max_answer_words=int(settings.get("qa_answer_length", 60)), ga_format=payload.get("question_format") or settings.get("qa_format"), progress_cb=ga_progress_cb)
        logger.info("GenQ QA    | job=%s doc=%s pairs=%s", job_id, doc_id, len(qa_pairs))

        try:
            with LocalKnowledgeStore(self.db_path) as knowledge_store:
                existing_embeddings, _ = knowledge_store.load_document(doc_id)

                # Build chunk index lookup for existing embeddings
                chunk_index_lookup: Dict[str, int] = {}
                for i, emb in enumerate(existing_embeddings):
                    meta = emb.metadata or {}
                    cid = meta.get("chunk_id")
                    if cid is not None and "chunk_index" in meta:
                        chunk_index_lookup[str(cid)] = meta.get("chunk_index")

                # Backfill chunk_index into associated QA entries and enrich metadata
                for qa in qa_pairs:
                    meta = qa.get("metadata") or {}
                    chunk_ids = meta.get("chunk_ids") or qa.get("chunk_ids") or []
                    cid = meta.get("chunk_id") or (chunk_ids[0] if chunk_ids else None)
                    if cid and str(cid) in chunk_index_lookup:
                        meta["chunk_index"] = chunk_index_lookup[str(cid)]
                        qa["metadata"] = meta
                    if cid and "chunk_id" not in meta:
                        meta["chunk_id"] = cid
                    meta["job_id"] = job_id
                    qa.setdefault("job_id", job_id)
                    if cid:
                        qa.setdefault("chunk_id", cid)
                        qa.setdefault("chunk_index", chunk_index_lookup.get(str(cid)))
                    if "question_id" not in meta:
                        meta["question_id"] = str(uuid.uuid4())
                    # Propagate useful fields into metadata
                    if "tags" in qa and "tags" not in meta:
                        meta["tags"] = qa.get("tags")
                    if "format" in qa and "format" not in meta:
                        meta["format"] = qa.get("format")
                    if "pages" in qa and "pages" not in meta:
                        meta["pages"] = qa.get("pages")
                    if "page" in qa and "page" not in meta:
                        meta["page"] = qa.get("page")
                    if "chunk_ids" in qa and "chunk_ids" not in meta:
                        meta["chunk_ids"] = qa.get("chunk_ids")
                    qa["metadata"] = meta

                # Attach job and chunk references to QA metadata and build chunk->question map
                chunk_question_map: Dict[str, List[str]] = {}
                for qa in qa_pairs:
                    meta = qa.get("metadata") or {}
                    chunk_id_for_map = meta.get("chunk_id")
                    question_id = meta.get("question_id")
                    if not chunk_id_for_map or not question_id:
                        continue
                    chunk_question_map.setdefault(str(chunk_id_for_map), []).append(str(question_id))

                # Prepare question_id updates only for involved chunks
                updates: Dict[str, list] = {}
                for cid, additions in chunk_question_map.items():
                    if not additions:
                        continue
                    updates[cid] = additions

                # Persist QA pairs
                metadata_index = getattr(knowledge_store, "metadata_index", None)
                if metadata_index is not None:
                    metadata_index.save(doc_id, qa_pairs, job_id=job_id)
                else:
                    store = getattr(knowledge_store, "_store", None)
                    if store and hasattr(store, "store_qa_pairs"):
                        store.store_qa_pairs(doc_id, qa_pairs, job_id=job_id)
                    else:
                        raise RuntimeError("No metadata index/store available to persist QA pairs")

                # Update chunk question_ids for affected chunks
                knowledge_store.update_chunk_question_ids(doc_id, updates)
        except Exception:
            logger.warning("Failed to persist generated QA for %s", doc_id, exc_info=True)

        # Collect tags
        tag_set = set()
        for emb in selected_embeddings:
            meta = emb.metadata or {}
            for tag in meta.get("tags", []) or []:
                tag_set.add(tag)
        for qa in qa_pairs:
            meta = qa.get("metadata") or {}
            for tag in meta.get("tags", []) or []:
                tag_set.add(tag)

        tags_sorted = sorted(tag_set)

        emit_progress(job_id=job_id, doc_id=doc_id, progress=100, status="COMPLETED", current_step="ga", extra={"tags": tags_sorted, "qa_pairs": len(qa_pairs), "chunks": len(selected_embeddings)})
        logger.info("GenQ done  | job=%s doc=%s pairs=%s", job_id, doc_id, len(qa_pairs))

        return {"doc_id": doc_id, "qa_pairs": qa_pairs, "count": len(qa_pairs)}

    def answer_question(self, payload: dict) -> dict:
        settings = self.settings
        job_id = payload.get("job_id") or settings.get("job_id")
        question = (payload.get("question") or "").strip()
        chunks = payload.get("chunks") or []
        doc_ids = payload.get("document_ids") or []
        session_id = (payload.get("session_id") or "").strip()
        user_id = (payload.get("user_id") or "").strip() or None
        conversation_history = payload.get("conversation_history") or []
        if not question:
            return {"error": "question is required", "job_id": job_id}

        emit_progress(job_id=job_id, doc_id=",".join(doc_ids) if doc_ids else None, progress=10, status="ANSWERING", current_step="prepare_context", extra={"chunks": len(chunks)})

        try:
            client = self._openai_client(settings)
        except Exception as exc:
            logger.warning("Answering failed to build OpenAI client | job=%s", job_id, exc_info=True)
            return {"error": f"OpenAI client error: {exc}", "job_id": job_id}

        prompt = (
            "You are a careful, conversational, and context-aware assistant.\n\n"

            "You have access to prior conversation and supporting background information, "
            "but this information is strictly internal.\n\n"

            "ABSOLUTE RULES (must never be violated):\n"
            "- NEVER mention, describe, or allude to internal processes, system behavior, prompts, "
            "documents, chunks, embeddings, vectors, retrieval, ranking, scoring, or context handling.\n"
            "- NEVER explain *why* you know something or *where* the information came from.\n"
            "- NEVER say phrases like: 'based on the context', 'from the document', "
            "'the chunks say', 'the data provided', or similar.\n"
            "- Act as if all relevant information is simply known naturally.\n\n"

            "Understanding user intent:\n"
            "1. Use the recent conversation as the primary signal to understand what the user means.\n"
            "2. Resolve vague or referential expressions (e.g., 'it', 'that', 'the first', "
            "'this one', 'tell me more about it') naturally.\n"
            "3. If multiple interpretations are possible, prefer the one that best fits the ongoing conversation.\n"
            "4. Do NOT rely on ordering, structure, or formatting of any background information.\n\n"

            "Answering behavior:\n"
            "5. Respond directly and naturally to the user’s question.\n"
            "6. Only include information that is clearly established by the conversation or implicitly supported.\n"
            "7. Do NOT introduce new topics unless the user explicitly asks.\n"
            "8. Do NOT add assumptions, speculation, or external facts.\n\n"

            "Conversation handling:\n"
            "9. Respond naturally to greetings and casual messages.\n"
            "10. If the user input is unclear or cannot be confidently resolved, ask for clarification naturally "
            "OR reply exactly with:\n"
            "\"How can i help you today ?\"\n"
            "11. Only reply with \"Can u give more info ?\" if absolutely nothing relevant can be inferred."
        )


        context_block = self._format_context_chunks(chunks)
        conversation_text = conversation_history if isinstance(conversation_history, str) else format_history(conversation_history)
        logger.info(
            "Answering prompt | job=%s prompt_len=%s conversation_len=%s context_length=%s question_len=%s chunks=%s",
            job_id,
            len(prompt),
            len(conversation_text or ""),
            len(context_block or ""),
            len(question),
            len(chunks),
        )
        try:
            messages = [
                {"role": "system", "content": prompt},
            ]
            if conversation_text:
                messages.append({"role": "system", "content": f"Recent conversation so far:\n{conversation_text}"})
            messages.extend(
                [
                    {"role": "system", "content": f"Context:\n{context_block}"},
                    {"role": "user", "content": question},
                ]
            )
            logger.info("Answering messages | job=%s messages_count=%s", job_id, len(messages))
            completion = client.chat.completions.create(
                model=settings.get("openai_model", "gpt-4o-mini"),
                temperature=0.2,
                messages=messages,
            )
            answer = completion.choices[0].message.content if completion.choices else ""
        except Exception as exc:
            logger.warning("Answering failed | job=%s", job_id, exc_info=True)
            logger.info("Falling back to direct answer task for job=%s", job_id)
            try:
                return self.direct_answer(payload)
            except Exception:
                return {"error": f"LLM answer failed: {exc}", "job_id": job_id}
        logger.info("Answering done | job=%s answer_length=%s", job_id, len(answer or ""))
        if answer.lower().strip() in ["how can i help you today ?"]:
            logger.info("Answer indicates insufficient context or off-topic | job=%s answer=%s", job_id, answer or "")
            return self.direct_answer(payload)
        emit_progress(job_id=job_id, doc_id=",".join(doc_ids) if doc_ids else None, progress=100, status="COMPLETED", current_step="answer", extra={"chunks": len(chunks), "answer": answer or ""})

        try:
            append_message(session_id, user_id, question, answer or "")
        except Exception:
            logger.warning("Failed to persist conversation history | job=%s session=%s", job_id, session_id, exc_info=True)
        try:
            save_conversation_message(self.db_path, session_id, user_id, job_id, question, answer or "")
        except Exception:
            logger.warning("Failed to persist conversation to DB | job=%s session=%s", job_id, session_id, exc_info=True)

        return {
            "job_id": job_id,
            "question": question,
            "answer": answer or "",
            "chunks_used": len(chunks),
            "document_ids": doc_ids,
        }

    def direct_answer(self, payload: dict) -> dict:
        settings = self.settings
        job_id = payload.get("job_id") or settings.get("job_id")
        question = (payload.get("question") or "").strip()
        chunks = payload.get("chunks") or []
        session_id = (payload.get("session_id") or "").strip()
        user_id = (payload.get("user_id") or "").strip() or None
        conversation_history = payload.get("conversation_history") or []
        if not question:
            return {"error": "question is required", "job_id": job_id}

        try:
            client = self._openai_client(settings)
        except Exception as exc:
            logger.warning("Direct answer failed to build OpenAI client | job=%s", job_id, exc_info=True)
            return {"error": f"OpenAI client error: {exc}", "job_id": job_id}

        emit_progress(job_id=job_id, doc_id=None, progress=20, step_progress=0, status="ANSWERING", current_step="direct_answer")

        base_prompt = "Answer the user's question directly. If uncertain, say \"I don't know\"."
        conversation_text = conversation_history if isinstance(conversation_history, str) else format_history(conversation_history)
        try:
            messages = [{"role": "system", "content": base_prompt}]
            if conversation_text:
                messages.append({"role": "system", "content": f"Recent conversation so far:\n{conversation_text}"})
            messages.append({"role": "user", "content": question})
            completion = client.chat.completions.create(
                model=settings.get("openai_model", "gpt-4o-mini"),
                temperature=0.3,
                messages=messages,
            )
            answer = completion.choices[0].message.content if completion.choices else ""
        except Exception as exc:
            logger.warning("Direct answer failed | job=%s", job_id, exc_info=True)
            return {"error": f"LLM direct answer failed: {exc}", "job_id": job_id}
        logger.info("Answering done | job=%s answer_length=%s", job_id, len(answer or ""))
        emit_progress(
            job_id=job_id,
            doc_id=None,
            progress=100,
            step_progress=100,
            status="COMPLETED",
            current_step="direct_answer",
            extra={"answer": answer or ""},
        )

        try:
            append_message(session_id, user_id, question, answer or "")
        except Exception:
            logger.warning("Failed to persist conversation history | job=%s session=%s", job_id, session_id, exc_info=True)
        try:
            save_conversation_message(self.db_path, session_id, user_id, job_id, question, answer or "")
        except Exception:
            logger.warning("Failed to persist conversation to DB | job=%s session=%s", job_id, session_id, exc_info=True)

        return {"job_id": job_id, "question": question, "answer": answer or "", "chunks_used": len(chunks)}

    def generate_question_variants(self, payload: dict) -> dict:
        """Generate variant questions based on an existing question_id."""
        settings = self.settings
        job_id = payload.get("job_id") or settings.get("job_id")
        question_id = (payload.get("question_id") or "").strip()
        if not question_id:
            return {"error": "question_id is required", "job_id": job_id}

        try:
            quantity = max(1, int(payload.get("quantity", 10)))
        except (TypeError, ValueError):
            quantity = 10
        difficulty = (payload.get("difficulty") or "medium").strip() or "medium"
        question_format = (payload.get("question_format") or "variety").strip() or "variety"
        settings.setdefault("job_id", job_id)
        db_path = settings.get("db_path", "hope/vector_store.db")

        logger.info("Variant start | job=%s question_id=%s qty=%s difficulty=%s format=%s", job_id, question_id, quantity, difficulty, question_format)

        try:
            with LocalKnowledgeStore(db_path) as knowledge_store:
                found = knowledge_store.find_question_by_id(question_id)
                if not found:
                    return {"error": f"question_id {question_id} not found", "job_id": job_id}
                doc_id, qa_entry = found
                emit_progress(
                    job_id=job_id,
                    doc_id=doc_id,
                    progress=10,
                    step_progress=0,
                    status="QA_VARIANTS",
                    current_step="qa_variants",
                    extra={"parent_question_id": question_id, "quantity": quantity, "difficulty": difficulty, "question_format": question_format},
                )
                embeddings, _ = knowledge_store.load_document(doc_id)
        except Exception as exc:
            logger.warning("Variant lookup failed | job=%s question_id=%s", job_id, question_id, exc_info=True)
            emit_progress(job_id=job_id, doc_id=None, progress=100, step_progress=0, status="FAILED", current_step="qa_variants", extra={"error": str(exc)})
            return {"error": f"lookup failed: {exc}", "job_id": job_id}

        meta = qa_entry.get("metadata") or {}
        target_chunk_id = meta.get("chunk_id") or qa_entry.get("chunk_id")
        target_chunk_index = meta.get("chunk_index") if meta.get("chunk_index") is not None else qa_entry.get("chunk_index")

        target_embedding: ChunkEmbedding | None = None
        for emb in embeddings:
            emb_meta = emb.metadata or {}
            if target_chunk_id and emb_meta.get("chunk_id") == target_chunk_id:
                target_embedding = emb
                break
            if target_chunk_index is not None and emb_meta.get("chunk_index") == target_chunk_index:
                target_embedding = emb
                break
        if target_embedding is None and embeddings:
            target_embedding = embeddings[0]

        if target_embedding is None:
            return {"error": "no chunk embedding available for the provided question", "job_id": job_id}

        candidates = self._embeddings_to_candidates([target_embedding], difficulty=difficulty)
        tags = list((target_embedding.metadata or {}).get("tags") or [])

        ga_generator = LLMQuestionGenerator(api_key=settings.get("openai_api_key"), model=settings.get("openai_model", "gpt-4o-mini"))
        ga_composer = QAComposer(
            ga_generator=ga_generator,
            ga_workers=int(settings.get("ga_workers", settings.get("qa_workers", 4))),
            theme_hint=None,
            difficulty_hint=difficulty,
            target_questions=quantity,
        )

        ga_progress = {"count": 0}

        def ga_progress_cb(item: Any, *_args: Any, **_kwargs: Any) -> None:
            if not job_id:
                return
            ga_progress["count"] += 1
            emit_progress(job_id=job_id, doc_id=doc_id, progress=85, step_progress=ga_progress["count"], status="QA_VARIANTS", current_step="qa_variants", extra={"count": ga_progress["count"]})

        qa_pairs = ga_composer.generate(
            candidates,
            max_answer_words=int(settings.get("qa_answer_length", 60)),
            ga_format=question_format,
            progress_cb=ga_progress_cb,
        )

        chunk_question_map: Dict[str, List[str]] = {}
        for qa in qa_pairs:
            qa_meta = qa.get("metadata") or {}
            qa_meta.setdefault("parent_question_id", question_id)
            qa_meta.setdefault("job_id", job_id)
            if "question_id" not in qa_meta:
                qa_meta["question_id"] = str(uuid.uuid4())
            if target_chunk_id:
                qa_meta.setdefault("chunk_id", target_chunk_id)
            if target_chunk_index is not None and "chunk_index" not in qa_meta:
                qa_meta["chunk_index"] = target_chunk_index
            if tags and "tags" not in qa_meta:
                qa_meta["tags"] = tags
            qa["metadata"] = qa_meta
            if qa_meta.get("chunk_id") and qa_meta.get("question_id"):
                chunk_question_map.setdefault(str(qa_meta["chunk_id"]), []).append(str(qa_meta["question_id"]))

        try:
            with LocalKnowledgeStore(db_path) as knowledge_store:
                metadata_index = getattr(knowledge_store, "metadata_index", None)
                if metadata_index is not None:
                    metadata_index.save(doc_id, qa_pairs, job_id=job_id)
                else:
                    store = getattr(knowledge_store, "_store", None)
                    if store and hasattr(store, "store_qa_pairs"):
                        store.store_qa_pairs(doc_id, qa_pairs, job_id=job_id)
                    else:
                        raise RuntimeError("No metadata index/store available to persist QA variants")
                if chunk_question_map:
                    knowledge_store.update_chunk_question_ids(doc_id, chunk_question_map)
        except Exception as exc:
            logger.warning("Failed to persist question variants for %s", question_id, exc_info=True)
            emit_progress(job_id=job_id, doc_id=doc_id, progress=100, step_progress=0, status="FAILED", current_step="qa_variants", extra={"error": str(exc), "parent_question_id": question_id})
            return {"error": f"persist failed: {exc}", "job_id": job_id}

        emit_progress(job_id=job_id, doc_id=doc_id, progress=100, step_progress=100, status="COMPLETED", current_step="qa_variants", extra={"qa_pairs": len(qa_pairs), "parent_question_id": question_id})
        return {"job_id": job_id, "document_id": doc_id, "parent_question_id": question_id, "qa_pairs": qa_pairs, "count": len(qa_pairs)}


# ---------------------
# Celery task wrappers
# ---------------------
@celery_app.task(name="pipeline.persist.document")
def persist_document_task(payload: dict, settings: dict) -> dict:
    """Persist embeddings/QA pairs to the knowledge store without generating QA content."""
    return LLMTaskService(settings).persist_document(payload)


@celery_app.task(name="pipeline.persist.document.batch")
def persist_document_batch_task(payload: dict, settings: dict) -> dict:
    """Append embeddings for a batch to the knowledge store."""
    return LLMTaskService(settings).persist_document_batch(payload)


@celery_app.task(name="pipeline.finalize.batch_pipeline")
def finalize_batch_pipeline_task(batch_results: list, payload: dict, settings: dict) -> dict:
    """Finalize a batch-based pipeline run by tagging chunks from the store and emitting completion."""
    svc = LLMTaskService(settings)
    doc_id = payload.get("doc_id") or settings.get("document_id")
    job_id = payload.get("job_id") or settings.get("job_id")
    try:
        tagged = svc.tag_chunks({"doc_id": doc_id, "job_id": job_id})
        emit_progress(job_id=job_id, doc_id=doc_id, progress=100, step_progress=100, status="COMPLETED", current_step="done", extra={"batches": len(batch_results or [])})
        return tagged
    except Exception as exc:
        logger.warning("Finalize batch pipeline failed | job=%s doc=%s", job_id, doc_id, exc_info=True)
        emit_progress(job_id=job_id, doc_id=doc_id, progress=100, step_progress=0, status="FAILED", current_step="done", extra={"error": str(exc)})
        return {"error": str(exc), "job_id": job_id, "doc_id": doc_id}


@celery_app.task(name="pipeline.llm.tag")
def tag_chunks_task(payload: dict, settings: dict) -> dict:
    """Tag each enriched chunk using the LLM service and propagate tags to embeddings."""
    return LLMTaskService(settings).tag_chunks(payload)


@celery_app.task(name="pipeline.llm.generate_questions")
def generate_questions_task(payload: dict, settings: dict) -> dict:
    """Generate questions for an existing document using stored embeddings."""
    return LLMTaskService(settings).generate_questions(payload)


@celery_app.task(name="pipeline.llm.answer_question")
def answer_question_task(payload: dict, settings: dict) -> dict:
    """Answer a question using retrieved context chunks via OpenAI."""
    return LLMTaskService(settings).answer_question(payload)


@celery_app.task(name="pipeline.llm.direct_answer")
def direct_answer_task(payload: dict, settings: dict) -> dict:
    """Fallback answer when no context chunks are available."""
    return LLMTaskService(settings).direct_answer(payload)


@celery_app.task(name="pipeline.llm.generate_question_variants")
def generate_question_variants_task(payload: dict, settings: dict) -> dict:
    """Generate variant questions based on an existing question_id."""
    return LLMTaskService(settings).generate_question_variants(payload)
