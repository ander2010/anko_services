from __future__ import annotations

import json
from pathlib import Path
import uuid
from typing import Any, Dict, List, Optional, Sequence

from celery_app import celery_app  # type: ignore
from pipeline.knowledge_store import LocalKnowledgeStore
from workflow.llm import LLMQuestionGenerator, QAFormat
from pipeline.logging_config import get_logger
from workflow.progress import emit_progress
from pipeline.types import ChunkCandidate, ChunkEmbedding
from pipeline.workflow_core import QAComposer, Chunkvectorizer

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


@celery_app.task(name="pipeline.llm.qa")
def llm_task(payload: dict, settings: dict) -> dict:
    """Generate QA content and persist artifacts if requested."""
    logger.info("QA start   | job=%s doc=%s path=%s", payload.get("job_id"), payload.get("doc_id"), payload.get("file_path"))
    job_id = payload.get("job_id") or settings.get("job_id")
    doc_id = payload.get("doc_id") or payload.get("document_id") or settings.get("document_id")

    qa_generator = None
    if settings.get("use_llm_qa"):
        qa_generator = LLMQuestionGenerator(api_key=settings.get("openai_api_key"), model=settings.get("openai_model", "gpt-4o-mini"))

    worker_count = int(settings.get("ga_workers", settings.get("qa_workers", 4)))
    qa_composer = QAComposer(ga_generator=qa_generator, ga_workers=worker_count, theme_hint=settings.get("theme"), difficulty_hint=settings.get("difficulty"), target_questions=settings.get("quantity_question"))
    logger.info("QA config  | job=%s doc=%s workers=%s use_llm_qa=%s skip_qa=%s", job_id, doc_id, worker_count, bool(qa_generator), settings.get("skip_qa"))

    qa_pairs: List[dict] = []
    if not settings.get("skip_qa"):
        chunks = deserialize_chunks(payload.get("enriched_chunks", []))
        qa_pairs = qa_composer.generate(chunks, max_answer_words=int(settings.get("qa_answer_length", 60)), ga_format=settings.get("qa_format", QAFormat.VARIETY.value))
        logger.info("QA generated | job=%s doc=%s pairs=%s chunks=%s", job_id, doc_id, len(qa_pairs), len(chunks))
        payload["qa_pairs"] = qa_pairs

    if settings.get("persist_local"):
        embeddings = deserialize_embeddings(payload.get("embeddings", []))
        db_path = settings.get("db_path", "hope/vector_store.db")
        with LocalKnowledgeStore(db_path) as knowledge_store:
            knowledge_store.save_document(doc_id or settings.get("document_id", "celery-doc"), payload.get("file_path", ""), embeddings, qa_pairs, allow_overwrite=settings.get("allow_overwrite", True), job_id=job_id)
            logger.info("Persisted    | job=%s doc=%s db=%s embeddings=%s qa_pairs=%s", job_id, doc_id, db_path, len(embeddings), len(qa_pairs))
    else:
        logger.info("Persistence  | job=%s doc=%s skipped (persist_local=%s)", job_id, doc_id, settings.get("persist_local"))

    # Collect unique tags observed across enriched chunks/embeddings
    tag_set = set()
    for chunk in payload.get("enriched_chunks", []) or []:
        for tag in chunk.get("tags", []) or []:
            tag_set.add(str(tag))
    for embedding in payload.get("embeddings", []) or []:
        meta = embedding.get("metadata") or {}
        for tag in meta.get("tags", []) or []:
            tag_set.add(str(tag))

    tags_sorted = sorted(tag_set)

    extra = {
        "tags": tags_sorted,
        "tag_set": tags_sorted,
        "qa_pairs": len(payload.get("qa_pairs", [])),
        "chunks": len(payload.get("enriched_chunks", [])),
        "embeddings": len(payload.get("embeddings", [])),
    }

    emit_progress(job_id=job_id, doc_id=doc_id, progress=100, step_progress=100, status="COMPLETED", current_step="ga", extra=extra)

    return payload


@celery_app.task(name="pipeline.llm.tag")
def tag_chunks_task(payload: dict, settings: dict) -> dict:
    """Tag each enriched chunk using the LLM service and propagate tags to embeddings."""
    job_id = payload.get("job_id") or settings.get("job_id")
    doc_id = payload.get("doc_id") or settings.get("document_id")

    chunks = payload.get("enriched_chunks") or []
    embeddings = payload.get("embeddings") or []
    total_chunks = len(chunks) or 1
    logger.info("Tag start  | job=%s doc=%s chunks=%s embeddings=%s use_llm_qa=%s", job_id, doc_id, len(chunks), len(embeddings), settings.get("use_llm_qa"))

    llm_generator = LLMQuestionGenerator(api_key=settings.get("openai_api_key"), model=settings.get("openai_model", "gpt-4o-mini"))

    for idx, chunk in enumerate(chunks, 1):
        text = chunk.get("text") or ""
        inferred_tags: List[str] = []
    
        # Build a strict prompt instructing the LLM to return a JSON array of tags
        prompt = (
            "You are an expert document indexing assistant. "
            "Your task is to generate concise, high-value tags that represent the core topics, entities," 
            "technologies, processes, or concepts discussed in the text.\n\n"
            
            "Guidelines:\n"
            "- Produce 3-5 tags\n"
            "- Tags must be specific and concrete (use domain terms, named entities, standards, tools, or methods)\n"
            "- Avoid generic labels such as 'overview', 'introduction', 'general', 'background'\n"
            "- Use lowercase\n"
            "- Prefer noun phrases (1-3 words)\n"
            "- Do not invent topics not present in the text\n\n"
            
            "Output format:\n"
            "- Respond ONLY with a valid JSON array of strings\n"
            "- No explanations, no markdown, no extra text\n\n"
            
            f"Text: {text[:2000]}"
        )
        content = None
        # Try to call the underlying client's chat completions if present
        try:
            client = getattr(llm_generator, "_client", None)
            model = getattr(llm_generator, "model", None)
            if client and hasattr(client, "chat") and hasattr(client.chat, "completions"):
                logger.info("Tagging LLM call | job=%s doc=%s chunk=%s/%s", job_id, doc_id, idx, total_chunks)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You generate JSON tag arrays only."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=96,
                )
                # Try to extract message content from common response shapes
                try:
                    content = getattr(resp.choices[0].message, "content", None) or getattr(resp.choices[0], "text", None)
                except Exception as e :
                    logger.warning("Tagging LLM response extraction failed for chunk %s: %s", idx, e)
                    content = None
        except Exception as e:
            logger.warning("Tagging LLM call failed for chunk %s: %s", idx, e)
            content = None

      
        try:
            parsed = json.loads(content)
            logger.info("Tagging LLM response parsed (chunk %s/%s): %s", idx, total_chunks, parsed)
            if isinstance(parsed, list):
                inferred_tags = [str(tag) for tag in parsed if tag]
            else:
                inferred_tags = [str(parsed)]
        except Exception as e:
            inferred_tags = [part.strip() for part in content.split(",") if part.strip()][:5]
            logger.warning("Tagging LLM response JSON parse failed for chunk %s: %s", idx, e)
            logger.warning("Tagging LLM response parse failed; using fallback for chunk %s ", idx)

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
        progress_val = round(80 + (step_pct / 100.0) * 10, 2)  # tagging spans 80-90
        emit_progress(job_id=job_id, doc_id=doc_id, progress=progress_val, step_progress=step_pct, status="TAGGING", current_step="tagging", extra={"chunk_index": idx, "total_chunks": total_chunks})
        logger.info("Tag progress | job=%s doc=%s chunk=%s/%s tags=%s", job_id, doc_id, idx, total_chunks, tags)


    # Persist tagged chunks back to the knowledge store
    try:
        db_path = settings.get("db_path", "hope/vector_store.db")
        with LocalKnowledgeStore(db_path) as knowledge_store:
            knowledge_store.save_document(document_id=doc_id or settings.get("document_id", "celery-doc"), source_path=payload.get("file_path", ""), chunk_embeddings=deserialize_embeddings(embeddings), qa_pairs=payload.get("qa_pairs", []), allow_overwrite=settings.get("allow_overwrite", True), job_id=job_id)
            logger.info("Persisted tagged chunks for %s", doc_id)
    except Exception:
        logger.warning("Failed to persist tagged chunks for %s", doc_id, exc_info=True)

    emit_progress(job_id=job_id, doc_id=doc_id, progress=100, step_progress=100, status="TAGGED", current_step="tagging", extra={"tags": tags, "tag_set": tags, "chunks": len(chunks), "embeddings": len(embeddings)})
    logger.info("Tag done    | job=%s doc=%s chunks=%s tags=%s", job_id, doc_id, len(chunks), len(tags))

    payload["enriched_chunks"] = chunks
    payload["embeddings"] = embeddings
    payload["tags"] = tags

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
                    single_results = knowledge_store.query_similar_chunks(vector.tolist(), document_id=doc_id, tags=payload.get("tags") or None, min_importance=payload.get("min_importance", settings.get("importance_threshold")), top_k=top_k)
                    if not single_results and payload.get("tags"):
                        single_results = knowledge_store.query_similar_chunks(vector.tolist(), document_id=doc_id, tags=None, min_importance=payload.get("min_importance", settings.get("importance_threshold")), top_k=top_k)
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
                results = knowledge_store.query_similar_chunks(fallback_embedding, document_id=doc_id, tags=None, min_importance=payload.get("min_importance", settings.get("importance_threshold")), top_k=top_k)
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
    ga_generator = None
    if settings.get("use_llm_qa"):
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
        emit_progress(job_id=job_id, doc_id=doc_id, progress=85, step_progress=ga_progress["count"], status="QA_GENERATING", current_step="qa", extra={"question": item.get("question") if isinstance(item, dict) else None, "count": ga_progress["count"]})
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

    emit_progress(job_id=job_id, doc_id=doc_id, progress=100, step_progress=100, status="COMPLETED", current_step="ga", extra={"tag_set": tags_sorted, "qa_pairs": len(qa_pairs), "chunks": len(selected_embeddings)})
    logger.info("GenQ done  | job=%s doc=%s pairs=%s", job_id, doc_id, len(qa_pairs))

    return {"doc_id": doc_id, "qa_pairs": qa_pairs, "count": len(qa_pairs)}
