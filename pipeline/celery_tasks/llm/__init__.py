from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Sequence

import requests
from celery_app import celery_app  # type: ignore
from openai import OpenAI
from redis import Redis
from pipeline.workflow.knowledge_store import LocalKnowledgeStore
from pipeline.workflow.document_intelligence import DocumentIntelligenceWorkflow
from pipeline.utils.logging_config import get_logger
from pipeline.utils.types import ChunkCandidate, ChunkEmbedding
from pipeline.workflow.qa import QAComposer
from pipeline.workflow.vectorizer import Chunkvectorizer
from pipeline.workflow.conversation import append_message, format_history
from pipeline.workflow.llm import LLMOutputSummarizer, LLMQuestionGenerator
from pipeline.workflow.utils.progress import emit_progress, PROGRESS_REDIS_URL
from pipeline.workflow.utils.persistence import save_conversation_message, save_document, save_question_summaries, save_summary, save_tags
from pipeline.workflow.utils.settings import normalize_settings
from pipeline.workflow.utils.tags import collect_tags_from_payload, ensure_llm_active_warning, filter_tags_by_embedding, infer_tags_with_llm


logger = get_logger(__name__)


def _summary_enabled(payload: dict[str, Any]) -> bool:
    metadata = payload.get("metadata") or {}
    if bool(metadata.get("skip_summary")):
        return False
    return True


def _post_battery_finalize_callback(payload: dict, result: dict, *, status_value: str, error_message: str | None = None) -> None:
    metadata = payload.get("metadata") or {}
    callback_url = str(metadata.get("callback_url") or "").strip()
    if not callback_url:
        return

    callback_token = str(metadata.get("callback_token") or os.getenv("INTERNAL_SERVICE_TOKEN", "andelef")).strip()
    source_bundle = payload.get("source_bundle") or {}
    callback_payload = {
        "token": callback_token,
        "job_id": payload.get("job_id"),
        "battery_id": payload.get("battery_id"),
        "status": status_value,
        "title": result.get("title") or payload.get("title"),
        "question_format": payload.get("question_format"),
        "source_bundle": source_bundle,
        "error": error_message or result.get("error"),
    }
    try:
        response = requests.post(
            callback_url,
            json=callback_payload,
            headers={"X-Internal-Token": callback_token} if callback_token else None,
            timeout=30,
        )
        logger.info(
            "Battery finalize callback | url=%s status=%s ok=%s job=%s",
            callback_url,
            response.status_code,
            response.ok,
            payload.get("job_id"),
        )
        response.raise_for_status()
    except Exception:
        logger.warning("Battery finalize callback failed | url=%s job=%s", callback_url, payload.get("job_id"), exc_info=True)


class LLMTaskService:
    """Encapsulates all LLM-related Celery task logic."""

    def __init__(self, settings: dict):
        self.settings = normalize_settings(settings or {})
        self.db_path = self.settings.get("db_path") or os.getenv("DB_URL", "hope/vector_store.db")
        self._progress_redis = None
        self.PREP_PROGRESS = 10.0
        self.STAGE_WEIGHTS = {"ocr": 10.0, "embed": 30.0, "persist": 30.0, "tag": 30.0}

    def _get_redis(self):
        if self._progress_redis is None:
            self._progress_redis = Redis.from_url(PROGRESS_REDIS_URL, decode_responses=True)
        return self._progress_redis

    def _update_units(self, job_id: str, doc_id: str, stage: str, count: int, *, total_pages: int | None = None) -> float:
        """Update per-stage counters and return monotonic overall percent."""
        if not job_id:
            return 0.0
        try:
            MIN_PROGRESS = self.PREP_PROGRESS

            r = self._get_redis()
            units_key = f"job:{job_id}:units"
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
            total_pages_val = max(1, total_pages_val)

            ocr_pct = min(1.0, done_ocr / total_pages_val) if total_pages_val else 0.0
            embed_pct = min(1.0, done_embed / total_chunks)
            persist_pct = min(1.0, done_persist / total_chunks)
            tag_pct = min(1.0, done_tag / total_chunks)

            raw_pct = MIN_PROGRESS
            raw_pct += self.STAGE_WEIGHTS["ocr"] * ocr_pct
            raw_pct += self.STAGE_WEIGHTS["embed"] * embed_pct
            raw_pct += self.STAGE_WEIGHTS["persist"] * persist_pct
            raw_pct += self.STAGE_WEIGHTS["tag"] * tag_pct
            raw_pct = min(100.0, raw_pct)

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
    def _build_summary_source(self, chunks: Sequence[dict]) -> str:
        max_chunks = int(self.settings.get("summary_max_chunks", 30))
        max_chars = int(self.settings.get("summary_max_chars", 18000))
        min_chars = int(self.settings.get("summary_min_chunk_chars", 80))
        context_window = int(self.settings.get("summary_context_window", 1))
        scored: list[tuple[float, int, int]] = []
        indexed_chunks: dict[int, str] = {}
        for idx, chunk in enumerate(chunks or []):
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            meta = chunk.get("metadata") or {}
            try:
                importance = float(meta.get("importance", chunk.get("importance", 0.0)) or 0.0)
            except (TypeError, ValueError):
                importance = 0.0
            try:
                page = int(meta.get("page") or chunk.get("page") or 0)
            except (TypeError, ValueError):
                page = 0
            index_key = meta.get("chunk_index")
            if index_key is None:
                index_key = chunk.get("chunk_index")
            try:
                index_key = int(index_key) if index_key is not None else idx
            except (TypeError, ValueError):
                index_key = idx
            indexed_chunks[index_key] = text
            if len(text) >= min_chars:
                scored.append((importance, page, index_key))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        seed_indices = [index_key for _importance, _page, index_key in scored[: max(1, max_chunks)]]
        expanded_indices: set[int] = set()
        for index_key in seed_indices:
            for offset in range(-context_window, context_window + 1):
                expanded_indices.add(index_key + offset)
        ordered_indices = [idx for idx in sorted(expanded_indices) if idx in indexed_chunks]
        selected: list[str] = []
        total_chars = 0
        for index_key in ordered_indices:
            if total_chars >= max_chars:
                break
            snippet = indexed_chunks[index_key][: max_chars - total_chars]
            selected.append(snippet)
            total_chars += len(snippet)
        logger.info(
            "Summary source built | chunks=%s selected=%s chars=%s max_chars=%s window=%s",
            len(chunks or []),
            len(selected),
            total_chars,
            max_chars,
            context_window,
        )
        return "\n\n".join(selected)

    def _summarize_qa_pairs(self, qa_pairs: Sequence[dict], *, doc_id: int, job_id: str | None) -> list[dict]:
        if not qa_pairs or not job_id:
            return []
        summarizer = LLMOutputSummarizer(
            api_key=self.settings.get("openai_api_key"),
            model=self.settings.get("openai_model", "gpt-4o-mini"),
        )
        if not summarizer.is_active:
            return []
        max_items = int(self.settings.get("summary_questions_max_items", 30))
        max_words = int(self.settings.get("summary_questions_max_words", 120))
        max_chars = int(self.settings.get("summary_questions_max_chars", 8000))
        lines: list[str] = []
        total_chars = 0
        for qa in qa_pairs[:max_items]:
            question = (qa.get("question") or "").strip()
            answer = (qa.get("correct_response") or "").strip()
            if not question:
                continue
            snippet = f"Q: {question}\nA: {answer}".strip()
            if not snippet:
                continue
            if total_chars + len(snippet) > max_chars:
                break
            lines.append(snippet)
            total_chars += len(snippet)
        if not lines:
            return []
        summary_text = summarizer.summarize_collection("\n\n".join(lines), label="questions", max_words=max_words)
        if not summary_text:
            return []
        return [
            {
                "job_id": job_id,
                "summary": summary_text,
            }
        ]

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
    def _require_doc_id(payload: dict, settings: dict) -> int:
        doc = payload.get("doc_id") or payload.get("document_id") or settings.get("document_id")
        if doc is None:
            raise ValueError("document_id is required and must be an integer")
        try:
            return int(doc)
        except Exception:
            raise ValueError(f"Invalid document_id '{doc}'; must be an integer")

    @staticmethod
    def _resolve_generation_document_ids(payload: dict, settings: dict) -> list[str]:
        source_bundle = payload.get("source_bundle") or {}
        raw_doc_ids = source_bundle.get("document_ids") or payload.get("document_ids") or payload.get("doc_ids") or []
        if not raw_doc_ids:
            single = payload.get("doc_id") or payload.get("document_id") or settings.get("document_id")
            if single not in (None, ""):
                raw_doc_ids = [single]

        doc_ids: list[str] = []
        seen: set[str] = set()
        for raw in raw_doc_ids:
            text = str(raw).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            doc_ids.append(text)
        if not doc_ids:
            raise ValueError("At least one document_id is required for generate_questions_task")
        return doc_ids

    @staticmethod
    def _annotate_embedding(embedding: ChunkEmbedding, *, document_id: str, chunk_index: int | None = None, similarity: float | None = None) -> ChunkEmbedding:
        metadata = dict(embedding.metadata or {})
        metadata["document_id"] = str(document_id)
        if chunk_index is not None and metadata.get("chunk_index") is None:
            metadata["chunk_index"] = chunk_index
        if similarity is not None:
            metadata["similarity"] = float(similarity)
        return ChunkEmbedding(text=embedding.text, embedding=list(embedding.embedding or []), metadata=metadata)

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
    def _build_fallback_qa_pairs(
        candidates: Sequence[ChunkCandidate],
        *,
        question_format: str,
        target_questions: int,
    ) -> List[dict[str, Any]]:
        if not candidates:
            return []

        normalized_format = str(question_format or "multiple_choice").strip().lower()
        desired = max(1, min(int(target_questions or 1), max(1, len(candidates) * 3)))
        generic_distractors = [
            "Clinical diagnosis",
            "Insurance coverage",
            "Appointment scheduling",
            "Emergency triage",
            "Medication dosage",
        ]

        pairs: List[dict[str, Any]] = []
        for candidate in candidates:
            tags = [str(tag).strip() for tag in (candidate.tags or []) if str(tag).strip()]
            answer = tags[0] if tags else "Payment processing"
            metadata = dict(candidate.metadata or {})
            context_text = str(candidate.text or "").strip()
            if normalized_format in {"multiple_choice", "multi_choice", "single_choice", "singlechoice", "single"}:
                option_pool = [answer]
                option_pool.extend(tag for tag in tags[1:] if tag.casefold() != answer.casefold())
                option_pool.extend(generic_distractors)
                options: list[str] = []
                seen_options: set[str] = set()
                for option in option_pool:
                    cleaned = str(option).strip()
                    if not cleaned:
                        continue
                    key = cleaned.casefold()
                    if key in seen_options:
                        continue
                    seen_options.add(key)
                    options.append(cleaned)
                    if len(options) >= 4:
                        break
                pairs.append(
                    {
                        "question": "Which topic best matches this document excerpt?",
                        "correct_response": answer,
                        "context": context_text,
                        "metadata": {
                            **metadata,
                            "type": "multiple_choice",
                            "format": "multiple_choice",
                            "question_format": "multiple_choice",
                            "options": options,
                        },
                        "tags": tags,
                        "page": getattr(candidate, "page", None),
                    }
                )
            elif normalized_format in {"true_false", "truefalse", "tf"}:
                statement = f"This excerpt is mainly about {answer}."
                pairs.append(
                    {
                        "question": statement,
                        "correct_response": "True",
                        "context": context_text,
                        "metadata": {
                            **metadata,
                            "type": "true_false",
                            "format": "true_false",
                            "question_format": "true_false",
                            "options": ["True", "False"],
                        },
                        "tags": tags,
                        "page": getattr(candidate, "page", None),
                    }
                )
            else:
                pairs.append(
                    {
                        "question": "What is the main topic of this document excerpt?",
                        "correct_response": answer,
                        "context": context_text,
                        "metadata": {
                            **metadata,
                            "type": "single_choice",
                            "format": "single_choice",
                            "question_format": "single_choice",
                            "options": [answer, "Other"],
                        },
                        "tags": tags,
                        "page": getattr(candidate, "page", None),
                    }
                )

            if len(pairs) >= desired:
                break

        return pairs[:desired]

    @staticmethod
    def _diversify_embeddings_by_page(embeddings: Sequence[ChunkEmbedding], max_items: int) -> List[ChunkEmbedding]:
        """Prefer one chunk per document/page first, then fill remaining slots in original order."""
        if max_items <= 0:
            return []
        if len(embeddings) <= max_items:
            return list(embeddings)

        buckets: Dict[tuple[str, int], List[ChunkEmbedding]] = {}
        order: List[tuple[str, int]] = []
        for emb in embeddings:
            meta = emb.metadata or {}
            try:
                page = int(meta.get("page", 0) or 0)
            except (TypeError, ValueError):
                page = 0
            doc_id = str(meta.get("document_id") or "")
            bucket_key = (doc_id, page)
            if bucket_key not in buckets:
                buckets[bucket_key] = []
                order.append(bucket_key)
            buckets[bucket_key].append(emb)

        diversified: List[ChunkEmbedding] = []
        while len(diversified) < max_items:
            added = False
            for bucket_key in order:
                bucket = buckets.get(bucket_key) or []
                if not bucket:
                    continue
                diversified.append(bucket.pop(0))
                added = True
                if len(diversified) >= max_items:
                    break
            if not added:
                break
        return diversified[:max_items]

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
        doc_id = self._require_doc_id(payload, settings)

        qa_pairs: List[dict] = payload.get("qa_pairs") or []

        if settings.get("persist_local"):
            embeddings = self._deserialize_embeddings(payload.get("embeddings", []))
            save_document(self.db_path, doc_id, payload.get("file_path", ""), embeddings, qa_pairs, allow_overwrite=settings.get("allow_overwrite", True), job_id=job_id)
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
        doc_id = self._require_doc_id(payload, settings)
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
        doc_id = self._require_doc_id(payload, settings)

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

        # Pre-count already tagged chunks to avoid retagging and to keep progress monotonic.
        already_tagged = 0
        for chunk in chunks:
            existing_tags = set(str(tag) for tag in (chunk.get("tags") or []))
            if not existing_tags:
                meta_tags = set(str(tag) for tag in (chunk.get("metadata") or {}).get("tags", []) or [])
                existing_tags |= meta_tags
            if existing_tags:
                already_tagged += 1
                tags = sorted(existing_tags)
                chunk["tags"] = tags
                meta = chunk.get("metadata") or {}
                meta["tags"] = tags
                chunk["metadata"] = meta
        if already_tagged > 0:
            try:
                r = self._get_redis()
                units_key = f"job:{job_id}:units"
                current_tagged = int(r.hget(units_key, "done_tag") or 0)
            except Exception:
                current_tagged = 0
            delta = max(0, already_tagged - current_tagged)
            if delta:
                self._update_units(job_id or "", doc_id or "", "tag", delta)

        done_so_far = already_tagged
        # Only invoke the LLM when a chunk has no tags yet; otherwise respect existing tags and just propagate.
        inactive_logged = False
        for idx, chunk in enumerate(chunks, 1):
            text = chunk.get("text") or ""
            existing_tags = set(str(tag) for tag in (chunk.get("tags") or []))
            inferred_tags: list[str] = []
            if not existing_tags:
                inferred_tags = infer_tags_with_llm(llm_generator, text, warn=False)
                if not inferred_tags and not inactive_logged:
                    logger.warning("LLM tagging inactive; skipping inferred tags for remaining chunks (job=%s doc=%s)", job_id, doc_id)
                    inactive_logged = True

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

            # Only count progress when we actually tag a previously untagged chunk.
            if inferred_tags or not existing_tags:
                done_so_far += 1
                overall = self._update_units(job_id or "", doc_id or "", "tag", 1)
                emit_progress(job_id=job_id, doc_id=doc_id, progress=overall, status="TAGGING", current_step="tagging", extra={"chunk_index": done_so_far, "total_chunks": total_chunks})
                logger.info("Tag progress | job=%s doc=%s chunk=%s/%s tags=%s", job_id, doc_id, done_so_far, total_chunks, tags)

        if embeddings:
            save_document(self.db_path, doc_id, payload.get("file_path", ""), self._deserialize_embeddings(embeddings), payload.get("qa_pairs", []), allow_overwrite=settings.get("allow_overwrite", True), job_id=job_id)

        tags_sorted = collect_tags_from_payload(chunks, embeddings)
        summary_text = ""
        summary_source = self._build_summary_source(chunks)
        if summary_source:
            summary_words = int(settings.get("summary_max_words", 320))
            summary_text = llm_generator.summarize_text(summary_source, max_words=summary_words)
            if summary_text:
                logger.info(
                    "Doc summary generated | doc=%s chars=%s max_words=%s",
                    doc_id,
                    len(summary_text),
                    summary_words,
                )
                save_summary(self.db_path, doc_id, summary_text)
        min_tags = int(settings.get("summary_min_tags", 15))
        max_tags = int(settings.get("summary_max_tags", 35))
        doc_tags = llm_generator.tag_document(summary_text, min_tags=min_tags, max_tags=max_tags) if summary_text else []
        if doc_tags:
            logger.info("Doc tags generated | doc=%s tags=%s", doc_id, len(doc_tags))
            tags_filtered = doc_tags
        else:
            candidates = tags_sorted
            phrase_candidates = [tag for tag in candidates if len(str(tag).split()) >= 2]
            if phrase_candidates:
                candidates = phrase_candidates
            tags_filtered = filter_tags_by_embedding(candidates, embeddings, min_support=2)
        section_descriptions = DocumentIntelligenceWorkflow.build_section_descriptions(
            section_titles=tags_filtered,
            document_summary=summary_text,
            chunks=chunks,
        )
        save_tags(self.db_path, doc_id, tags_filtered, job_id=job_id, descriptions=section_descriptions)
        emit_progress(job_id=job_id, doc_id=doc_id, progress=100, status="TAGGED", current_step="tagging", extra={"tags": tags_filtered, "chunks": len(chunks), "embeddings": len(embeddings)})
        emit_progress(job_id=job_id, doc_id=doc_id, progress=100, status="COMPLETED", current_step="done", extra={"tags": tags_filtered, "chunks": len(chunks), "embeddings": len(embeddings)})
        logger.info("Tag done    | job=%s doc=%s chunks=%s tags=%s", job_id, doc_id, len(chunks), len(tags_filtered))
        logger.info("Tag list    | job=%s doc=%s tags_full=%s", job_id, doc_id, tags_filtered)

        payload["enriched_chunks"] = chunks
        payload["embeddings"] = embeddings
        payload["tags"] = tags_filtered

        return payload

    def generate_questions(self, payload: dict) -> dict:
        settings = self.settings
        document_ids = self._resolve_generation_document_ids(payload, settings)
        progress_doc_id = ",".join(document_ids)
        primary_doc_id = document_ids[0]
        job_id = payload.get("job_id") or settings.get("job_id")
        logger.info("GenQ start | job=%s docs=%s db=%s", job_id, document_ids, self.db_path)

        embeddings_by_doc: Dict[str, List[ChunkEmbedding]] = {}
        embeddings: List[ChunkEmbedding] = []
        with LocalKnowledgeStore(self.db_path) as knowledge_store:
            for doc_id in document_ids:
                loaded_embeddings, _ = knowledge_store.load_document(doc_id)
                annotated = [
                    self._annotate_embedding(
                        emb,
                        document_id=str(doc_id),
                        chunk_index=(emb.metadata or {}).get("chunk_index", idx),
                    )
                    for idx, emb in enumerate(loaded_embeddings)
                ]
                if annotated:
                    embeddings_by_doc[str(doc_id)] = annotated
                    embeddings.extend(annotated)

        available_doc_ids = list(embeddings_by_doc.keys())
        if not embeddings:
            emit_progress(job_id=job_id, doc_id=progress_doc_id, progress=100, status="NO_EMBEDDINGS", current_step="load_embeddings", extra={"embeddings": 0, "process": "generate_question", "document_ids": document_ids, "title": payload.get("title")})
            logger.warning("GenQ aborted| job=%s docs=%s reason=no_embeddings", job_id, document_ids)
            return {"document_ids": document_ids, "qa_pairs": [], "count": 0, "error": "no embeddings found for requested documents"}

        emit_progress(job_id=job_id, doc_id=progress_doc_id, progress=15, status="GENERATING_QUESTIONS", current_step="load_embeddings", extra={"embeddings": len(embeddings), "process": "generate_question", "document_ids": available_doc_ids, "title": payload.get("title")})
        logger.info("GenQ loaded| job=%s docs=%s embeddings=%s", job_id, available_doc_ids, len(embeddings))

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
        logger.info("GenQ select| job=%s docs=%s top_k=%s queries=%s tags=%s min_importance=%s", job_id, available_doc_ids, top_k, query_texts, payload.get("tags"), payload.get("min_importance"))

        # If query texts provided, attempt a similarity search; otherwise pick top_k
        merged: Dict[tuple[str, int], tuple[str, int, ChunkEmbedding, float]] = {}
        try:
            if query_texts:
                vectorizer = Chunkvectorizer(settings.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
                try:
                    query_vectors = vectorizer.encode_texts(query_texts)
                except Exception:
                    query_vectors = []
                # Run similarity queries inside a knowledge store context
                with LocalKnowledgeStore(self.db_path) as knowledge_store:
                    for query_text, vector in zip(query_texts, query_vectors or []):
                        single_results = knowledge_store.query_similar_chunks(vector, document_ids=available_doc_ids, tags=payload.get("tags") or None, min_importance=payload.get("min_importance", settings.get("importance_threshold")), top_k=top_k)
                        if not single_results and payload.get("tags"):
                            single_results = knowledge_store.query_similar_chunks(vector, document_ids=available_doc_ids, tags=None, min_importance=payload.get("min_importance", settings.get("importance_threshold")), top_k=top_k)
                        for docid, idx, chunk, similarity in single_results:
                            key = (str(docid), idx)
                            annotated_chunk = self._annotate_embedding(chunk, document_id=str(docid), chunk_index=idx, similarity=float(similarity))
                            if key not in merged or similarity > merged[key][3]:
                                merged[key] = (str(docid), idx, annotated_chunk, float(similarity))
            logger.info("GenQ merged | job=%s docs=%s merged_hits=%s query_texts=%s", job_id, available_doc_ids, len(merged), bool(query_texts))
        except Exception:
            merged = {}

        if merged:
            selected_embeddings = [item[2] for item in merged.values()]
            logger.info("GenQ source | job=%s docs=%s source=merged", job_id, available_doc_ids)
        else:
            fallback_embedding = self._average_embedding_vectors([emb.embedding or [] for emb in embeddings])
            if fallback_embedding:
                with LocalKnowledgeStore(self.db_path) as knowledge_store:
                    results = knowledge_store.query_similar_chunks(fallback_embedding, document_ids=available_doc_ids, tags=payload.get("tags") or None, min_importance=payload.get("min_importance", settings.get("importance_threshold")), top_k=top_k)
                selected_embeddings = [
                    self._annotate_embedding(item[2], document_id=str(item[0]), chunk_index=item[1], similarity=float(item[3]))
                    for item in results
                ] if results else embeddings[:top_k]
                logger.info("GenQ source | job=%s docs=%s source=fallback results=%s", job_id, available_doc_ids, len(selected_embeddings))
            else:
                selected_embeddings = embeddings[:top_k]
                logger.info("GenQ source | job=%s docs=%s source=topk", job_id, available_doc_ids)

        selected_embeddings = self._diversify_embeddings_by_page(selected_embeddings, top_k)
        logger.info("GenQ diversify | job=%s docs=%s diversified=%s top_k=%s", job_id, available_doc_ids, len(selected_embeddings), top_k)

        emit_progress(job_id=job_id, doc_id=progress_doc_id, progress=40, status="GENERATING_QUESTIONS", current_step="select_chunks", extra={"selected_chunks": len(selected_embeddings), "top_k": top_k, "process": "generate_question", "document_ids": available_doc_ids, "title": payload.get("title")})
        logger.info("GenQ chunks| job=%s docs=%s selected=%s top_k=%s", job_id, available_doc_ids, len(selected_embeddings), top_k)

        candidates = self._embeddings_to_candidates(selected_embeddings, theme=payload.get("theme"), difficulty=payload.get("difficulty"))
        logger.info("GenQ cand  | job=%s docs=%s candidates=%s theme=%s difficulty=%s", job_id, available_doc_ids, len(candidates), payload.get("theme"), payload.get("difficulty"))

        worker_count = int(settings.get("ga_workers", settings.get("qa_workers", 4)))
        ga_generator = LLMQuestionGenerator(api_key=settings.get("openai_api_key"), model=settings.get("openai_model", "gpt-4o-mini"))
        try:
            ga_composer = QAComposer(ga_generator=ga_generator, ga_workers=worker_count, theme_hint=payload.get("theme"), difficulty_hint=payload.get("difficulty"), target_questions=payload.get("quantity_question"))
        except Exception:
            ga_composer = QAComposer(ga_generator=ga_generator, ga_workers=worker_count)

        ga_progress = {"count": 0}
        total_target = int(payload.get("quantity_question") or 0)
        # QA generation follows "load_embeddings" (15%) and "select_chunks" (40%).
        # Start at 40 so progress doesn't appear stuck until 40% of items are produced.
        progress_start = 40.0
        progress_end = 100.0

        def ga_progress_cb(item: Any, *_args: Any, **_kwargs: Any) -> None:
            if not job_id:
                return
            ga_progress["count"] += 1
            question_text = (item.get("question") if isinstance(item, dict) else "") or ""
            preview = question_text.strip().replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:157] + "..."
            extra = {"count": ga_progress["count"]}
            if total_target:
                extra["total"] = total_target
            if preview:
                extra["question_preview"] = preview
            if total_target:
                effective_count = min(ga_progress["count"], total_target)
                pct = progress_start + (effective_count / total_target) * (progress_end - progress_start)
            else:
                pct = progress_end
            extra["document_ids"] = available_doc_ids
            extra["title"] = payload.get("title")
            emit_progress(job_id=job_id, doc_id=progress_doc_id, progress=round(min(pct, progress_end), 2), status="QA_GENERATING", current_step="qa", extra=extra)
            logger.info("GenQ prog  | job=%s docs=%s qa_progress=%s question=%s", job_id, available_doc_ids, ga_progress["count"], preview)

        qa_pairs = ga_composer.generate(candidates, max_answer_words=int(settings.get("qa_answer_length", 60)), ga_format=payload.get("question_format") or settings.get("qa_format"), progress_cb=ga_progress_cb)
        if not qa_pairs and candidates:
            qa_pairs = self._build_fallback_qa_pairs(
                candidates,
                question_format=str(payload.get("question_format") or settings.get("qa_format") or "multiple_choice"),
                target_questions=int(payload.get("quantity_question") or 1),
            )
            logger.info("GenQ fallback| job=%s docs=%s pairs=%s", job_id, available_doc_ids, len(qa_pairs))
        logger.info("GenQ QA    | job=%s docs=%s pairs=%s", job_id, available_doc_ids, len(qa_pairs))

        try:
            chunk_index_lookup_by_doc: Dict[str, Dict[str, int]] = {}
            chunk_doc_lookup: Dict[str, str] = {}
            for doc_id, doc_embeddings in embeddings_by_doc.items():
                chunk_index_lookup_by_doc[doc_id] = {}
                for emb in doc_embeddings:
                    meta = emb.metadata or {}
                    cid = meta.get("chunk_id")
                    chunk_index = meta.get("chunk_index")
                    if cid is not None and chunk_index is not None:
                        chunk_index_lookup_by_doc[doc_id][str(cid)] = chunk_index
                        chunk_doc_lookup[str(cid)] = doc_id

            qa_pairs_by_doc: Dict[str, List[dict]] = {}
            chunk_question_updates_by_doc: Dict[str, Dict[str, List[str]]] = {}

            for qa in qa_pairs:
                meta = qa.get("metadata") or {}
                chunk_ids = meta.get("chunk_ids") or qa.get("chunk_ids") or []
                cid = meta.get("chunk_id") or (chunk_ids[0] if chunk_ids else None)
                qa_doc_id = meta.get("document_id") or qa.get("document_id")
                if not qa_doc_id and cid:
                    qa_doc_id = chunk_doc_lookup.get(str(cid))
                if not qa_doc_id:
                    for chunk_id in chunk_ids:
                        qa_doc_id = chunk_doc_lookup.get(str(chunk_id))
                        if qa_doc_id:
                            break
                qa_doc_id = str(qa_doc_id or primary_doc_id)

                if cid and "chunk_id" not in meta:
                    meta["chunk_id"] = cid
                if cid:
                    chunk_index_lookup = chunk_index_lookup_by_doc.get(qa_doc_id, {})
                    if str(cid) in chunk_index_lookup:
                        meta["chunk_index"] = chunk_index_lookup[str(cid)]
                        qa.setdefault("chunk_index", chunk_index_lookup[str(cid)])
                meta["document_id"] = qa_doc_id
                meta["job_id"] = job_id
                qa["document_id"] = qa_doc_id
                qa.setdefault("job_id", job_id)
                if cid:
                    qa.setdefault("chunk_id", cid)
                if "question_id" not in meta:
                    meta["question_id"] = str(uuid.uuid4())
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

                qa_pairs_by_doc.setdefault(qa_doc_id, []).append(qa)
                chunk_id_for_map = meta.get("chunk_id")
                question_id = meta.get("question_id")
                if chunk_id_for_map and question_id:
                    chunk_question_updates_by_doc.setdefault(qa_doc_id, {}).setdefault(str(chunk_id_for_map), []).append(str(question_id))

            with LocalKnowledgeStore(self.db_path) as knowledge_store:
                metadata_index = getattr(knowledge_store, "metadata_index", None)
                store = getattr(knowledge_store, "_store", None)
                for qa_doc_id, qa_items in qa_pairs_by_doc.items():
                    if metadata_index is not None:
                        metadata_index.save(qa_doc_id, qa_items, job_id=job_id)
                    elif store and hasattr(store, "store_qa_pairs"):
                        store.store_qa_pairs(qa_doc_id, qa_items, job_id=job_id)
                    else:
                        raise RuntimeError("No metadata index/store available to persist QA pairs")

                    updates = chunk_question_updates_by_doc.get(qa_doc_id) or {}
                    if updates:
                        knowledge_store.update_chunk_question_ids(qa_doc_id, updates)
        except Exception:
            logger.warning("Failed to persist generated QA for docs=%s", available_doc_ids, exc_info=True)

        try:
            summaries = self._summarize_qa_pairs(qa_pairs, doc_id=primary_doc_id, job_id=job_id) if _summary_enabled(payload) else []
            if summaries:
                save_question_summaries(self.db_path, summaries)
        except Exception:
            logger.warning("Failed to summarize QA | job=%s docs=%s", job_id, available_doc_ids, exc_info=True)

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

        emit_progress(job_id=job_id, doc_id=progress_doc_id, progress=100, status="COMPLETED", current_step="ga", extra={"tags": tags_sorted, "qa_pairs": len(qa_pairs), "chunks": len(selected_embeddings), "document_ids": available_doc_ids, "title": payload.get("title")})
        logger.info("GenQ done  | job=%s docs=%s pairs=%s", job_id, available_doc_ids, len(qa_pairs))

        return {"job_id": job_id, "document_ids": available_doc_ids, "title": payload.get("title"), "qa_pairs": qa_pairs, "count": len(qa_pairs)}

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

        emit_progress(job_id=job_id, doc_id=None, progress=20, status="ANSWERING", current_step="direct_answer")

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
        emit_progress(job_id=job_id, doc_id=None, progress=100, status="COMPLETED", current_step="direct_answer", extra={"answer": answer or ""})

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
                    status="QA_VARIANTS",
                    current_step="qa_variants",
                    extra={"parent_question_id": question_id, "quantity": quantity, "difficulty": difficulty, "question_format": question_format},
                )
                embeddings, _ = knowledge_store.load_document(doc_id)
        except Exception as exc:
            logger.warning("Variant lookup failed | job=%s question_id=%s", job_id, question_id, exc_info=True)
            emit_progress(job_id=job_id, doc_id=None, progress=100, status="FAILED", current_step="qa_variants", extra={"error": str(exc)})
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
        total_target = int(quantity or 0)
        # Variants emit an initial 10% after lookup; start from there for smooth progress.
        progress_start = 10.0
        progress_end = 100.0

        def ga_progress_cb(item: Any, *_args: Any, **_kwargs: Any) -> None:
            if not job_id:
                return
            ga_progress["count"] += 1
            question_text = (item.get("question") if isinstance(item, dict) else "") or ""
            preview = question_text.strip().replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:157] + "..."
            extra = {"count": ga_progress["count"]}
            if total_target:
                extra["total"] = total_target
            if preview:
                extra["question_preview"] = preview
            if total_target:
                effective_count = min(ga_progress["count"], total_target)
                pct = progress_start + (effective_count / total_target) * (progress_end - progress_start)
            else:
                pct = progress_end
            emit_progress(job_id=job_id, doc_id=doc_id, progress=round(min(pct, progress_end), 2), status="QA_VARIANTS", current_step="qa_variants", extra=extra)

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
            if "format" in qa and "format" not in qa_meta:
                qa_meta["format"] = qa.get("format")
            if "pages" in qa and "pages" not in qa_meta:
                qa_meta["pages"] = qa.get("pages")
            if "page" in qa and "page" not in qa_meta:
                qa_meta["page"] = qa.get("page")
            if "chunk_ids" in qa and "chunk_ids" not in qa_meta:
                qa_meta["chunk_ids"] = qa.get("chunk_ids")
            qa["metadata"] = qa_meta
            qa.setdefault("job_id", job_id)
            if target_chunk_id:
                qa.setdefault("chunk_id", target_chunk_id)
            if target_chunk_index is not None:
                qa.setdefault("chunk_index", target_chunk_index)
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
            emit_progress(job_id=job_id, doc_id=doc_id, progress=100, status="FAILED", current_step="qa_variants", extra={"error": str(exc), "parent_question_id": question_id})
            return {"error": f"persist failed: {exc}", "job_id": job_id}

        try:
            summaries = self._summarize_qa_pairs(qa_pairs, doc_id=doc_id, job_id=job_id)
            if summaries:
                save_question_summaries(self.db_path, summaries)
        except Exception:
            logger.warning("Failed to summarize QA variants | job=%s doc=%s", job_id, doc_id, exc_info=True)

        emit_progress(job_id=job_id, doc_id=doc_id, progress=100, status="COMPLETED", current_step="qa_variants", extra={"qa_pairs": len(qa_pairs), "parent_question_id": question_id})
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
    doc_id = LLMTaskService._require_doc_id(payload, settings)
    job_id = payload.get("job_id") or settings.get("job_id")
    file_path = payload.get("file_path") or payload.get("file path")
    try:
        tagged = svc.tag_chunks({"doc_id": doc_id, "job_id": job_id})
        emit_progress(job_id=job_id, doc_id=doc_id, progress=100, status="COMPLETED", current_step="done", extra={"batches": len(batch_results or [])})
        return tagged
    except Exception as exc:
        logger.warning("Finalize batch pipeline failed | job=%s doc=%s", job_id, doc_id, exc_info=True)
        emit_progress(job_id=job_id, doc_id=doc_id, progress=100, status="FAILED", current_step="done", extra={"error": str(exc)})
        return {"error": str(exc), "job_id": job_id, "doc_id": doc_id}
    finally:
        # Clean up any cached remote temp file for this document.
        try:
            from pipeline.workflow.ingestion import PdfIngestion
            if file_path:
                PdfIngestion.cleanup_cached_file(Path(file_path))
            else:
                PdfIngestion.cleanup_cached_file()
        except Exception:
            logger.debug("Cleanup of cached remote PDF failed for job=%s doc=%s", job_id, doc_id, exc_info=True)


@celery_app.task(name="pipeline.llm.tag")
def tag_chunks_task(payload: dict, settings: dict) -> dict:
    """Tag each enriched chunk using the LLM service and propagate tags to embeddings."""
    return LLMTaskService(settings).tag_chunks(payload)


@celery_app.task(name="pipeline.llm.generate_questions")
def generate_questions_task(payload: dict, settings: dict) -> dict:
    """Generate questions for an existing document using stored embeddings."""
    try:
        result = LLMTaskService(settings).generate_questions(payload)
    except Exception as exc:
        _post_battery_finalize_callback(payload, {"title": payload.get("title")}, status_value="failed", error_message=str(exc))
        raise
    _post_battery_finalize_callback(payload, result, status_value="completed")
    return result


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
