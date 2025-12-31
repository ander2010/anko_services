from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Sequence

from celery_app import celery_app  # type: ignore
from openai import OpenAI
from pipeline.knowledge_store import LocalKnowledgeStore
from workflow.llm import LLMQuestionGenerator
from pipeline.logging_config import get_logger
from workflow.progress import emit_progress
from pipeline.types import ChunkCandidate, ChunkEmbedding
from pipeline.workflow_core import QAComposer, Chunkvectorizer
from workflow.conversation import append_message, format_history
from workflow.utils.persistence import save_conversation_message
from workflow.utils.tags import collect_tags_from_payload, ensure_llm_active_warning, infer_tags_with_llm
from workflow.utils.settings import normalize_settings
from workflow.utils.persistence import save_document, save_notification, save_tags

logger = get_logger(__name__)


def deserialize_chunks(chunks: List[dict]) -> List[ChunkCandidate]:
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


def deserialize_embeddings(items: List[dict]) -> List[ChunkEmbedding]:
    return [
        ChunkEmbedding(
            text=item.get("text", ""),
            embedding=item.get("embedding", []),
            metadata=item.get("metadata", {}),
        )
        for item in items or []
    ]


def embeddings_to_candidates(embeddings: Sequence[ChunkEmbedding], theme: Optional[str] = None, difficulty: Optional[str] = None) -> List[ChunkCandidate]:
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


def _openai_client(settings: dict) -> OpenAI:
    api_key = settings.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for LLM calls")
    return OpenAI(api_key=api_key)


@celery_app.task(name="pipeline.persist.document")
def persist_document_task(payload: dict, settings: dict) -> dict:
    """Persist embeddings/QA pairs to the knowledge store without generating QA content."""
    settings = normalize_settings(settings or {})
    job_id = payload.get("job_id") or settings.get("job_id")
    doc_id = payload.get("doc_id") or payload.get("document_id") or settings.get("document_id")

    qa_pairs: List[dict] = payload.get("qa_pairs") or []

    if settings.get("persist_local"):
        embeddings = deserialize_embeddings(payload.get("embeddings", []))
        db_path = settings.get("db_path", "hope/vector_store.db")
        save_document(db_path, doc_id or settings.get("document_id", "celery-doc"), payload.get("file_path", ""), embeddings, qa_pairs, allow_overwrite=settings.get("allow_overwrite", True), job_id=job_id)
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

    emit_progress(job_id=job_id, doc_id=doc_id, progress=85, step_progress=100, status="PERSISTED", current_step="persist", extra=extra)

    return payload


@celery_app.task(name="pipeline.llm.tag")
def tag_chunks_task(payload: dict, settings: dict) -> dict:
    """Tag each enriched chunk using the LLM service and propagate tags to embeddings."""
    settings = normalize_settings(settings or {})
    job_id = payload.get("job_id") or settings.get("job_id")
    doc_id = payload.get("doc_id") or settings.get("document_id")

    chunks = payload.get("enriched_chunks") or []
    embeddings = payload.get("embeddings") or []
    total_chunks = len(chunks) or 1
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

        step_pct = round((idx / total_chunks) * 100, 2)
        progress_val = round(85 + (step_pct / 100.0) * 10, 2)  # tagging spans 85-95
        emit_progress(job_id=job_id, doc_id=doc_id, progress=progress_val, step_progress=step_pct, status="TAGGING", current_step="tagging", extra={"chunk_index": idx, "total_chunks": total_chunks})
        logger.info("Tag progress | job=%s doc=%s chunk=%s/%s tags=%s", job_id, doc_id, idx, total_chunks, tags)

    # Persist tagged chunks back to the knowledge store
    db_path = settings.get("db_path", "hope/vector_store.db")
    save_document(db_path, doc_id or settings.get("document_id", "celery-doc"), payload.get("file_path", ""), deserialize_embeddings(embeddings), payload.get("qa_pairs", []), allow_overwrite=settings.get("allow_overwrite", True), job_id=job_id)

    tags_sorted = collect_tags_from_payload(chunks, embeddings)
    # Persist tags + completion notification for durability.
    save_tags(db_path, doc_id or settings.get("document_id", "celery-doc"), tags_sorted)
    save_notification(
        db_path,
        job_id or "",
        {
            "status": "COMPLETED",
            "doc_id": doc_id,
            "tags": tags_sorted,
            "chunks": len(chunks),
            "embeddings": len(embeddings),
        },
    )
    # Signal tagging done, then mark the overall pipeline as completed.
    emit_progress(job_id=job_id, doc_id=doc_id, progress=100, step_progress=100, status="TAGGED", current_step="tagging", extra={"tags": tags_sorted,  "chunks": len(chunks), "embeddings": len(embeddings)})
    emit_progress(job_id=job_id, doc_id=doc_id, progress=100, step_progress=100, status="COMPLETED", current_step="done", extra={"tags": tags_sorted,  "chunks": len(chunks), "embeddings": len(embeddings)})
    logger.info("Tag done    | job=%s doc=%s chunks=%s tags=%s", job_id, doc_id, len(chunks), len(tags_sorted))

    payload["enriched_chunks"] = chunks
    payload["embeddings"] = embeddings
    payload["tags"] = tags_sorted

    return payload


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


@celery_app.task(name="pipeline.llm.generate_questions")
def generate_questions_task(payload: dict, settings: dict) -> dict:
    """Generate questions for an existing document using stored embeddings."""
    settings = normalize_settings(settings or {})
    doc_id = payload.get("doc_id")
    if not doc_id:
        raise ValueError("doc_id is required for generate_questions_task")

    db_path = settings.get("db_path", "hope/vector_store.db")
    job_id = payload.get("job_id") or settings.get("job_id")
    logger.info("GenQ start | job=%s doc=%s db=%s", job_id, doc_id, db_path)

    with LocalKnowledgeStore(db_path) as knowledge_store:
        embeddings, _ = knowledge_store.load_document(doc_id)

    emit_progress(job_id=job_id, doc_id=doc_id, progress=15, step_progress=0, status="GENERATING_QUESTIONS", current_step="load_embeddings", extra={"embeddings": len(embeddings), "process": "generate_question"})
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
            with LocalKnowledgeStore(db_path) as knowledge_store:
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

    selected_embeddings: List[ChunkEmbedding]
    if merged:
        selected_embeddings = [item[2] for item in merged.values()]
        logger.info("GenQ source | job=%s doc=%s source=merged", job_id, doc_id)
    else:
        fallback_embedding = _average_embedding_vectors([emb.embedding or [] for emb in embeddings])
        if fallback_embedding:
            with LocalKnowledgeStore(db_path) as knowledge_store:
                results = knowledge_store.query_similar_chunks(fallback_embedding, document_ids=[doc_id], tags=None, min_importance=payload.get("min_importance", settings.get("importance_threshold")), top_k=top_k)
            selected_embeddings = [item[2] for item in results] if results else embeddings[:top_k]
            logger.info("GenQ source | job=%s doc=%s source=fallback results=%s", job_id, doc_id, len(selected_embeddings))
        else:
            selected_embeddings = embeddings[:top_k]
            logger.info("GenQ source | job=%s doc=%s source=topk", job_id, doc_id)

    emit_progress(job_id=job_id, doc_id=doc_id, progress=40, step_progress=len(selected_embeddings), status="GENERATING_QUESTIONS", current_step="select_chunks", extra={"selected_chunks": len(selected_embeddings), "top_k": top_k, "process": "generate_question"})
    logger.info("GenQ chunks| job=%s doc=%s selected=%s top_k=%s", job_id, doc_id, len(selected_embeddings), top_k)

    candidates = embeddings_to_candidates(selected_embeddings, theme=payload.get("theme"), difficulty=payload.get("difficulty"))
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
        emit_progress(job_id=job_id, doc_id=doc_id, progress=85, step_progress=ga_progress["count"], status="QA_GENERATING", current_step="qa", extra={ "count": ga_progress["count"]})
        logger.info("GenQ prog  | job=%s doc=%s qa_progress=%s question=%s", job_id, doc_id, ga_progress["count"], (item.get("question") if isinstance(item, dict) else None))

    qa_pairs = ga_composer.generate(candidates, max_answer_words=int(settings.get("qa_answer_length", 60)), ga_format=payload.get("question_format") or settings.get("qa_format"), progress_cb=ga_progress_cb)
    logger.info("GenQ QA    | job=%s doc=%s pairs=%s", job_id, doc_id, len(qa_pairs))

    # Attach job and chunk references to QA metadata and build chunk->question map
    chunk_question_map: Dict[str, List[dict]] = {}
    for qa_idx, qa in enumerate(qa_pairs):
        qa_meta = qa.get("metadata") or {}
        chunk_ids = qa_meta.get("chunk_ids") or []
        qa_meta["job_id"] = job_id
        primary_chunk_id = chunk_ids[0] if chunk_ids else None
        qa_meta["chunk_id"] = primary_chunk_id
        qa["metadata"] = qa_meta
        for cid in chunk_ids:
            chunk_question_map.setdefault(str(cid), []).append({"job_id": job_id, "qa_index": qa_idx})

    try:
        with LocalKnowledgeStore(db_path) as knowledge_store:
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
                cid = meta.get("chunk_id")
                if cid and str(cid) in chunk_index_lookup:
                    meta["chunk_index"] = chunk_index_lookup[str(cid)]
                    qa["metadata"] = meta
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

    emit_progress(job_id=job_id, doc_id=doc_id, progress=100, step_progress=100, status="COMPLETED", current_step="ga", extra={"tags": tags_sorted, "qa_pairs": len(qa_pairs), "chunks": len(selected_embeddings)})
    logger.info("GenQ done  | job=%s doc=%s pairs=%s", job_id, doc_id, len(qa_pairs))

    return {"doc_id": doc_id, "qa_pairs": qa_pairs, "count": len(qa_pairs)}


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


@celery_app.task(name="pipeline.llm.answer_question")
def answer_question_task(payload: dict, settings: dict) -> dict:
    """Answer a question using retrieved context chunks via OpenAI."""
    settings = normalize_settings(settings or {})
    job_id = payload.get("job_id") or settings.get("job_id")
    question = (payload.get("question") or "").strip()
    chunks = payload.get("chunks") or []
    doc_ids = payload.get("document_ids") or []
    session_id = (payload.get("session_id") or "").strip()
    user_id = (payload.get("user_id") or "").strip() or None
    conversation_history = payload.get("conversation_history") or []
    if not question:
        return {"error": "question is required", "job_id": job_id}

    emit_progress(job_id=job_id, doc_id=",".join(doc_ids) if doc_ids else None, progress=10, step_progress=0, status="ANSWERING", current_step="prepare_context", extra={"chunks": len(chunks)})

    try:
        client = _openai_client(settings)
    except Exception as exc:
        logger.warning("Answering failed to build OpenAI client | job=%s", job_id, exc_info=True)
        return {"error": f"OpenAI client error: {exc}", "job_id": job_id}

    prompt = (
        "You are a careful assistant.\n\n"
        "The context snippets come directly from user-provided content. The user may provide one or multiple sources as context.\n\n"
        "Your tasks:\n"
        "1. Use ONLY the provided context snippets and recent conversation.\n"
        "2. If the user asks about the topic, purpose, summary, or overall meaning of the content, respond based on the content provided. "
        "It is okay if the theme is broad or partially scattered; summarize what is most strongly represented.\n"
        "3. Respond using only information supported by the context.\n"
        "4. Do NOT add details, facts, or assumptions not present in the context.\n"
        "5. Avoid speculation beyond minimal, justified inference.\n"
        "6. If the user input is clearly unrelated, off-topic, dangling, or cannot be answered using the context, reply EXACTLY with:\n"
        "\"How can i help you today ?\"\n"
        "7. Otherwise, respond naturally and conversationally to greetings, introductions, or casual chat.\n"
        "8. Only reply with \"Can u give more info ?\" if there is literally nothing in the context relevant to the user question."
    )


    context_block = _format_context_chunks(chunks)
    conversation_text = conversation_history if isinstance(conversation_history, str) else format_history(conversation_history)
    logger.info("Answering context | job=%s context_length=%s", job_id, len(context_block or ""))
    logger.info("Answering question| job=%s question=%s", job_id, question or "")
    logger.info("example of context chunk | job=%s chunk=%s", job_id, (chunks[0].get("text") if chunks else "")[:200] if chunks else "")
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
        logger.info("Answering messages detail | job=%s messages=%s", job_id, messages)
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
            return direct_answer_task(payload, settings)
        except Exception:
            return {"error": f"LLM answer failed: {exc}", "job_id": job_id}
    logger.info("Answering done | job=%s answer_length=%s", job_id, len(answer or ""))
    logger.info("Answer content| job=%s answer=%s", job_id, answer or "")
    if answer.lower().strip() in ["how can i help you today ?"]:
        logger.info("Answer indicates insufficient context or off-topic | job=%s answer=%s", job_id, answer or "")
        return direct_answer_task(payload, settings)
    emit_progress(
        job_id=job_id,
        doc_id=",".join(doc_ids) if doc_ids else None,
        progress=100,
        step_progress=100,
        status="COMPLETED",
        current_step="answer",
        extra={"chunks": len(chunks), "answer": answer or ""},
    )

    try:
        append_message(session_id, user_id, question, answer or "")
    except Exception:
        logger.warning("Failed to persist conversation history | job=%s session=%s", job_id, session_id, exc_info=True)
    try:
        save_conversation_message(settings.get("db_path", "hope/vector_store.db"), session_id, user_id, job_id, question, answer or "")
    except Exception:
        logger.warning("Failed to persist conversation to DB | job=%s session=%s", job_id, session_id, exc_info=True)

    return {
        "job_id": job_id,
        "question": question,
        "answer": answer or "",
        "chunks_used": len(chunks),
        "document_ids": doc_ids,
    }


@celery_app.task(name="pipeline.llm.direct_answer")
def direct_answer_task(payload: dict, settings: dict) -> dict:
    """Fallback answer when no context chunks are available."""
    settings = normalize_settings(settings or {})
    job_id = payload.get("job_id") or settings.get("job_id")
    question = (payload.get("question") or "").strip()
    chunks = payload.get("chunks") or []
    session_id = (payload.get("session_id") or "").strip()
    user_id = (payload.get("user_id") or "").strip() or None
    conversation_history = payload.get("conversation_history") or []
    if not question:
        return {"error": "question is required", "job_id": job_id}

    try:
        client = _openai_client(settings)
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
    logger.info("Answer content| job=%s answer=%s", job_id, answer or "")
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
        save_conversation_message(settings.get("db_path", "hope/vector_store.db"), session_id, user_id, job_id, question, answer or "")
    except Exception:
        logger.warning("Failed to persist conversation to DB | job=%s session=%s", job_id, session_id, exc_info=True)

    return {"job_id": job_id, "question": question, "answer": answer or "", "chunks_used": 0}
