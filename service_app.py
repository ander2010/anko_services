from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import uuid
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from pipeline.workflow.knowledge_store import LocalKnowledgeStore
from pipeline.utils.logging_config import get_logger
from pipeline.celery_tasks.llm import answer_question_task, direct_answer_task, generate_questions_task, generate_question_variants_task
from pipeline.celery_tasks.flashcards import generate_flashcards_task
from pipeline.workflow.utils.celery_pipeline import enqueue_pipeline
from pipeline.workflow.conversation import CONVERSATION_MAX_MESSAGES, CONVERSATION_MAX_TOKENS, fetch_recent_async
from pipeline.workflow.utils.safety import SafetyValidator
from pipeline.db.flashcard_storage import insert_review
from pipeline.workflow.utils.request_models import (
    AskRequest,
    ProcessRequest,
    ProcessType,
    QuestionVariantsRequest,
    default_settings,
)
from pipeline.workflow.utils.request_utils import (
    CONTEXT_TOKEN_LIMIT,
    MAX_CONTEXT_CHUNKS,
    apply_external_options,
    derive_job_id,
    derive_variant_job_id,
    embed_question,
    merge_settings,
    trim_chunks_to_budget,
)
from pipeline.workflow.flashcards import (
    Flashcard,
    FlashcardWorkflow,
    RatingScale,
    SESSION_MAX_WAIT_SECONDS,
)
from pipeline.workflow.utils.progress import PROGRESS_DB_URL, get_progress_client, set_progress

logger = get_logger("pipeline.service")



app = FastAPI(title="Pipeline Streaming Service")
# init_flashcard_db(PROGRESS_DB_URL)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"sta-tus": "ok"})


@app.websocket("/ws/progress/{job_id}")
async def progress_ws(websocket: WebSocket, job_id: str):
    """Websocket endpoint that streams progress updates for a document."""
    await websocket.accept()
    client = await get_progress_client()
    channel = f"progress:{job_id}"
    key = f"job:{job_id}"
    pubsub = client.pubsub()
    await pubsub.subscribe(channel)
    try:
        snapshot = await client.hgetall(key)
        if snapshot:
            await websocket.send_json({"type": "snapshot", "job_id": job_id, **snapshot})
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=10.0)
            if message and message.get("data"):
                data = message["data"]
                try:
                    payload = json.loads(data)
                except (TypeError, json.JSONDecodeError):
                    payload = {"raw": data}
                payload.setdefault("job_id", job_id)
                payload.setdefault("type", "progress")
                snapshot = await client.hgetall(key)
                merged = {**snapshot, **payload}
                if "progress" not in merged and merged.get("progress_process") is not None:
                    merged["progress"] = merged["progress_process"]
                await websocket.send_json(merged)
                if str(merged.get("status", "")).upper() in {"COMPLETED", "FAILED", "ERROR"}:
                    break
            else:
                snapshot = await client.hgetall(key)
                heartbeat = {"type": "heartbeat", "job_id": job_id, **snapshot}
                if "progress" not in heartbeat and heartbeat.get("progress_process") is not None:
                    heartbeat["progress"] = heartbeat["progress_process"]
                await websocket.send_json(heartbeat)
                if str(heartbeat.get("status", "")).upper() in {"COMPLETED", "FAILED", "ERROR"}:
                    break
    except WebSocketDisconnect:
        logger.info("Websocket disconnected for job_id=%s", job_id)
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.close()


@app.websocket("/ws/chat/{session_id}")
async def chat_ws(websocket: WebSocket, session_id: str):
    """Bidirectional chat endpoint: receive questions, enqueue jobs, and stream progress on the same socket."""
    await websocket.accept()
    session_id = (session_id or "").strip() or str(uuid.uuid4())

    client = await get_progress_client()
    listeners: list[asyncio.Task] = []

    async def forward_progress(job: str) -> None:
        pubsub = client.pubsub()
        channel = f"progress:{job}"
        key = f"job:{job}"
        await pubsub.subscribe(channel)
        done_status = {"COMPLETED", "FAILED", "ERROR"}
        try:
            snapshot = await client.hgetall(key)
            if snapshot:
                status = str(snapshot.get("status", "")).upper()
                if status in done_status:
                    await websocket.send_json({"type": "final", "job_id": job, "answer": snapshot.get("answer")})
                    return
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=10.0)
                if message and message.get("data"):
                    data = message["data"]
                    try:
                        payload = json.loads(data)
                    except (TypeError, json.JSONDecodeError):
                        payload = {"raw": data}
                    snapshot = await client.hgetall(key)
                    merged = {**snapshot, **payload}
                    status = str(merged.get("status", "")).upper()
                    if status in done_status:
                        await websocket.send_json({"type": "final", "job_id": job, "answer": merged.get("answer")})
                        break
                else:
                    snapshot = await client.hgetall(key)
                    status = str(snapshot.get("status", "")).upper()
                    if status in done_status and snapshot:
                        await websocket.send_json({"type": "final", "job_id": job, "answer": snapshot.get("answer")})
                        break
        except Exception:
            logger.warning("Progress forwarding failed | job=%s", job, exc_info=True)
        finally:
            try:
                await pubsub.unsubscribe(channel)
            except Exception:
                pass
            try:
                await pubsub.close()
            except Exception:
                pass

    try:
        while True:
            job_id = str(uuid.uuid4())
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                await websocket.send_json({"error": "invalid JSON"})
                continue

            question = (data.get("question") or "").strip()
            if not question:
                await websocket.send_json({"error": "question is required"})
                continue

            doc_ids_raw = data.get("context") or []
            doc_ids = [str(doc).strip() for doc in doc_ids_raw if str(doc).strip()]
            top_k_raw = data.get("top_k")
            min_importance_raw = data.get("min_importance")
            user_id = (data.get("user_id") or "").strip() or None

            settings = default_settings(PROGRESS_DB_URL)
            api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            validator = SafetyValidator(api_key)
            try:
                validator.validate_question(question)
                validator.validate_text_list(doc_ids, field="context document ids")
            except HTTPException as exc:
                logger.info("Input validation failed | session=%s question=%s error=%s", session_id, question, exc.detail)
                await websocket.send_json({"type": "final", "job_id": job_id, "answer": "How can i help you?"})
                continue
            except Exception as exc:
                logger.info("Input validation unexpected error | session=%s question=%s error=%s", session_id, question, exc)
                await websocket.send_json({"type": "final", "job_id": job_id, "answer": "How can i help you?"})
                continue

            min_importance = min_importance_raw if min_importance_raw is not None else settings.importance_threshold
            conversation_history: list[dict] = []
            try:
                conversation_history = await fetch_recent_async(session_id, token_budget=CONVERSATION_MAX_TOKENS, max_items=CONVERSATION_MAX_MESSAGES)
            except Exception:
                logger.warning("Failed to load conversation history | session=%s", session_id, exc_info=True)

            selected_chunks: list[dict] = []
            missing_docs: list[str] = []
            if doc_ids:
                try:
                    query_vector = embed_question(question, settings.embedding_model)
                except Exception as exc:
                    await websocket.send_json({"error": f"Failed to embed question: {exc}"})
                    continue

                top_k = MAX_CONTEXT_CHUNKS
                try:
                    if top_k_raw:
                        top_k = max(1, int(top_k_raw))
                except (TypeError, ValueError):
                    top_k = MAX_CONTEXT_CHUNKS

                retrieved_chunks: list[dict] = []
                with LocalKnowledgeStore(settings.db_path) as knowledge_store:
                    for doc_id in doc_ids:
                        if not knowledge_store.document_exists(doc_id):
                            missing_docs.append(doc_id)
                            continue
                        results = knowledge_store.query_similar_chunks(query_vector, document_ids=[doc_id], min_importance=min_importance, top_k=top_k)
                        for docid, idx, chunk, similarity in results:
                            meta = dict(chunk.metadata or {})
                            meta.setdefault("chunk_index", idx)
                            meta.setdefault("document_id", docid)
                            retrieved_chunks.append(
                                {
                                    "document_id": docid,
                                    "chunk_index": idx,
                                    "text": chunk.text,
                                    "metadata": meta,
                                    "similarity": float(similarity),
                                }
                            )
                        # Fallback: if no chunks retrieved, pull a top 10 by importance for context.
                        if not retrieved_chunks:
                            fallback = knowledge_store.query_similar_chunks([0.0], document_ids=[doc_id], min_importance=None, top_k=10)
                            for docid, idx, chunk, similarity in fallback:
                                meta = dict(chunk.metadata or {})
                                meta.setdefault("chunk_index", idx)
                                meta.setdefault("document_id", docid)
                                retrieved_chunks.append(
                                    {
                                        "document_id": docid,
                                        "chunk_index": idx,
                                        "text": chunk.text,
                                        "metadata": meta,
                                        "similarity": float(similarity),
                                    }
                                )

                retrieved_chunks.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
                max_chunks = min(len(retrieved_chunks), max(top_k * max(len(doc_ids), 1), 1), MAX_CONTEXT_CHUNKS)
                selected_chunks = trim_chunks_to_budget(retrieved_chunks[:max_chunks], question, CONTEXT_TOKEN_LIMIT)

            settings_payload = merge_settings(settings.__dict__, {"importance_threshold": min_importance})
            task_payload = {
                "job_id": job_id,
                "question": question,
                "document_ids": doc_ids,
                "session_id": session_id,
                "user_id": user_id,
                "conversation_history": conversation_history,
            }

            if selected_chunks and not missing_docs:
                task_payload["chunks"] = selected_chunks
                task = answer_question_task.apply_async(args=[task_payload, settings_payload], task_id=job_id)
                await set_progress(job_id=job_id, doc_id=",".join(doc_ids), progress=0, status="QUEUED", current_step="answer_question", extra={"chunks": len(selected_chunks)})
                mode = "contextual"
            else:
                task = direct_answer_task.apply_async(args=[task_payload, settings_payload], task_id=job_id)
                await set_progress(job_id=job_id, doc_id=",".join(doc_ids), progress=0, status="QUEUED", current_step="direct_answer", extra={"missing_documents": missing_docs})
                mode = "direct"

            await websocket.send_json(
                {
                    "type": "enqueued",
                    "job_id": job_id,
                    "mode": mode,
                    "question": question,
                    "document_ids": doc_ids,
                    "chunk_count": len(selected_chunks),
                    "missing_documents": missing_docs,
                    "session_id": session_id,
                    "history_messages": len(conversation_history),
                }
            )
            listener = asyncio.create_task(forward_progress(job_id))
            listeners.append(listener)
    except WebSocketDisconnect:
        logger.info("Chat websocket disconnected | session=%s", session_id)
    finally:
        for task in listeners:
            if not task.done():
                task.cancel()

@app.post("/process-request")
async def process_request(payload: ProcessRequest = Body(...)) -> JSONResponse:
    """Accepts a JSON request with doc_id, file_path, and options."""
    settings = default_settings(PROGRESS_DB_URL)
    settings.document_id = payload.doc_id
    settings = apply_external_options(settings, payload)
    job_id = derive_job_id(payload)
    settings.job_id = job_id

    if payload.process == ProcessType.PROCESS_PDF:
        if not payload.file_path:
            return JSONResponse({"error": "file_path is required for process_pdf"}, status_code=400)
        merged_settings = merge_settings(settings.__dict__, payload.metadata or {})
        # Ensure file_path is JSON serializable before passing to Celery.
        merged_settings["file_path"] = str(payload.file_path)
        task = enqueue_pipeline(Path(payload.file_path), settings=merged_settings, persist_local=settings.persist_local)
        await set_progress(job_id=job_id, doc_id=payload.doc_id, progress=0, status="QUEUED", current_step="ingestion", extra={"process": ProcessType.PROCESS_PDF.value, "task_id": task.id})
        return JSONResponse(
            {
                "task_id": task.id,
                "job_id": job_id,
                "document_id": payload.doc_id,
                "process": payload.process.value,
                "options": merged_settings,
                "metadata": payload.metadata,
                "status": "queued",
            }
        )

    if payload.process == ProcessType.GENERATE_QUESTION:
        task_payload = {
            "job_id": job_id,
            "doc_id": payload.doc_id,
            "theme": payload.theme,
            "quantity_question": payload.quantity_question,
            "difficulty": payload.difficulty,
            "question_format": payload.question_format,
            "tags": payload.tags,
            "query_text": payload.query_text,
            "top_k": payload.top_k,
            "min_importance": payload.min_importance,
            "metadata": payload.metadata,
        }
        settings_payload = merge_settings(settings.__dict__, payload.metadata or {})
        task = generate_questions_task.apply_async(args=[task_payload, settings_payload], task_id=job_id)
        await set_progress(job_id=job_id, doc_id=payload.doc_id, progress=0, status="QUEUED", current_step="generate_question", extra={"process": ProcessType.GENERATE_QUESTION.value, "task_id": task.id})
        return JSONResponse(
            {
                "task_id": task.id,
                "job_id": job_id,
                "document_id": payload.doc_id,
                "process": payload.process.value,
                "options": settings_payload,
                "metadata": payload.metadata,
                "status": "queued",
            }
        )

    return JSONResponse({"error": f"Unsupported process {payload.process}"}, status_code=400)


@app.post("/ask")
async def ask(payload: AskRequest = Body(...)) -> JSONResponse:
    """Answer a question using stored document chunks, with a direct LLM fallback."""
    settings = default_settings(PROGRESS_DB_URL)
    question = (payload.question or "").strip()
    session_id = (payload.session_id or str(uuid.uuid4())).strip()
    user_id = (payload.user_id or "").strip() or None
    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    validator = SafetyValidator(api_key)
    validator.validate_question(question)
    validator.validate_text_list(payload.context, field="context document ids")
    min_importance = payload.min_importance if payload.min_importance is not None else settings.importance_threshold

    doc_ids = [str(doc).strip() for doc in payload.context if str(doc).strip()]  
    if not doc_ids:
        logger.info("No document IDs provided in context, falling back to direct LLM answer | question=%s", question)

    conversation_history: list[dict] = []
    try:
        conversation_history = await fetch_recent_async(session_id, token_budget=CONVERSATION_MAX_TOKENS, max_items=CONVERSATION_MAX_MESSAGES)
    except Exception:
        logger.warning("Failed to load conversation history | session=%s", session_id, exc_info=True)

    selected_chunks: list[dict] = []
    missing_docs: list[str] = []
    if doc_ids:
        try:
            query_vector = embed_question(question, settings.embedding_model)
        except Exception as exc:
            logger.warning("Failed to embed question", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to embed question") from exc

        top_k_raw = payload.top_k or 8
        try:
            top_k = max(1, int(top_k_raw))
        except (TypeError, ValueError):
            top_k = 8

        retrieved_chunks: list[dict] = []
        with LocalKnowledgeStore(settings.db_path) as knowledge_store:
            for doc_id in doc_ids:
                if not knowledge_store.document_exists(doc_id):
                    missing_docs.append(doc_id)
                    continue
                results = knowledge_store.query_similar_chunks(query_vector, document_ids=[doc_id], min_importance=min_importance, top_k=top_k)
                for docid, idx, chunk, similarity in results:
                    meta = dict(chunk.metadata or {})
                    meta.setdefault("chunk_index", idx)
                    meta.setdefault("document_id", docid)
                    retrieved_chunks.append(
                        {
                            "document_id": docid,
                            "chunk_index": idx,
                            "text": chunk.text,
                            "metadata": meta,
                            "similarity": float(similarity),
                        }
                    )

        retrieved_chunks.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
        max_chunks = min(len(retrieved_chunks), max(top_k * max(len(doc_ids), 1), 1), MAX_CONTEXT_CHUNKS)
        selected_chunks = trim_chunks_to_budget(retrieved_chunks[:max_chunks], question, CONTEXT_TOKEN_LIMIT)

    job_id = str(uuid.uuid4())
    settings_payload = merge_settings(settings.__dict__, {"importance_threshold": min_importance})
    task_payload = {
        "job_id": job_id,
        "question": question,
        "document_ids": doc_ids,
        "session_id": session_id,
        "user_id": user_id,
        "conversation_history": conversation_history,
    }

    if selected_chunks and not missing_docs:
        task_payload["chunks"] = selected_chunks
        task = answer_question_task.apply_async(args=[task_payload, settings_payload], task_id=job_id)
        await set_progress(job_id=job_id, doc_id=",".join(doc_ids), progress=0, status="QUEUED", current_step="answer_question", extra={"chunks": len(selected_chunks)})
        mode = "contextual"
    else:
        logger.info("No relevant chunks found, falling back to direct LLM answer | question=%s", question)
        task = direct_answer_task.apply_async(args=[task_payload, settings_payload], task_id=job_id)
        await set_progress(job_id=job_id, doc_id=",".join(doc_ids), progress=0, status="QUEUED", current_step="direct_answer", extra={ "missing_documents": missing_docs})
        mode = "direct"
    return JSONResponse(
        {
            "task_id": task.id,
            "job_id": job_id,
            "mode": mode,
            "question": question,
            "document_ids": doc_ids,
            "chunk_count": len(selected_chunks),
            "missing_documents": missing_docs,
            "session_id": session_id,
            "history_messages": len(conversation_history),
        }
    )


@app.post("/questions/{question_id}/variants")
async def generate_question_variants(question_id: str, payload: QuestionVariantsRequest = Body(...)) -> JSONResponse:
    """Generate variant questions for an existing question_id."""
    settings = default_settings(PROGRESS_DB_URL)
    quantity = payload.quantity or 10
    difficulty = (payload.difficulty or "medium").strip() or "medium"
    question_format = (payload.question_format or "variety").strip() or "variety"
    job_id = payload.job_id or derive_variant_job_id(question_id, quantity, difficulty, question_format)

    task_payload = {
        "job_id": job_id,
        "question_id": question_id,
        "quantity": quantity,
        "difficulty": difficulty,
        "question_format": question_format,
    }
    settings_payload = merge_settings(settings.__dict__, {})
    task = generate_question_variants_task.apply_async(args=[task_payload, settings_payload], task_id=job_id)
    await set_progress(job_id=job_id, doc_id=None, progress=0, status="QUEUED", current_step="qa_variants", extra={"parent_question_id": question_id, "quantity": quantity})
    return JSONResponse(
        {
            "task_id": task.id,
            "job_id": job_id,
            "parent_question_id": question_id,
            "quantity": quantity,
            "difficulty": difficulty,
            "question_format": question_format,
            "status": "queued",
        }
    )


@app.post("/flashcards/create")
async def flashcards_create(payload: dict = Body(...)) -> JSONResponse:
    """Create/generate a flashcard set asynchronously and return a job_id."""
    job_id = FlashcardWorkflow.derive_flashcard_job_id(payload)
    task = generate_flashcards_task.apply_async(args=[job_id, payload], task_id=job_id)
    return JSONResponse(
        {
            "job_id": job_id,
            "task_id": task.id,
            "status": "queued",
        }
    )

@app.websocket("/ws/flashcards/{job_id}")
async def flashcards_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    session = {
        "seq": 0,
        "job_id": None,
        "user_id": None,
        "in_flight_card": None,
    }

    try:
        raw = await websocket.receive_text()
        data = json.loads(raw) if raw else {}
        if data.get("message_type") != "subscribe_job":
            await websocket.send_json({"error": "first message must be subscribe_job"})
            await websocket.close()
            return

        job_id = str(job_id or data.get("job_id") or "").strip()
        user_id = str(data.get("user_id") or job_id or "").strip()
        last_seq = data.get("last_seq") or 0
        request_id = data.get("request_id") or str(uuid.uuid4())
        token = str(data.get("token") or websocket.headers.get("authorization") or "").replace("Bearer ", "").strip()
        if not job_id:
            await websocket.send_json({"error": "job_id is required"})
            await websocket.close()
            return

        async with FlashcardWorkflow.flashcard_lock:
            req = FlashcardWorkflow.flashcard_requests.get(job_id)
            if req is None:
                # Attempt to hydrate from stored flashcards when no request exists.
                existing_cards = await FlashcardWorkflow.load_flashcards(job_id, user_id)
                if not existing_cards:
                    logger.warning("WS subscribe unknown job | job_id=%s user_id=%s", job_id, user_id)
                    await websocket.send_json({"error": "unknown job_id"})
                    await websocket.close()
                    return
                req = {
                    "document_ids": [],
                    "tags": [],
                    "quantity": len(existing_cards),
                    "difficulty": None,
                    "job_id": job_id,
                    "user_id": user_id,
                }
                FlashcardWorkflow.flashcard_requests[job_id] = req
                FlashcardWorkflow.flashcard_tokens[job_id] = ""
                FlashcardWorkflow.flashcard_store[job_id] = existing_cards
            req_user = (req.get("user_id") or "").strip()
            if req_user and req_user != user_id:
                logger.warning("WS subscribe user mismatch | job_id=%s user_id=%s expected=%s", job_id, user_id, req_user)
                await websocket.send_json({"error": "job_id does not belong to user"})
                await websocket.close()
                return
            expected_token = FlashcardWorkflow.flashcard_tokens.get(job_id)
            if expected_token and token and token != expected_token:
                logger.warning("WS subscribe token invalid | job_id=%s user_id=%s", job_id, user_id)
                await websocket.send_json({"error": "invalid token"})
                await websocket.close()
                return
            req_quantity = int(req.get("quantity") or 0)
            store = FlashcardWorkflow.flashcard_store.setdefault(job_id, {})
            existing_cards = await FlashcardWorkflow.load_flashcards(job_id, user_id)
            if existing_cards:
                store.update(existing_cards)
            existing_count = len(store)
            user_provided_job_id = bool(str(req.get("job_id") or "").strip())
            to_generate = 0 if user_provided_job_id else max(0, req_quantity - existing_count)
            if to_generate:
                logger.info("WS triggering generation | job_id=%s user_id=%s existing=%s to_generate=%s", job_id, user_id, existing_count, to_generate)
                # Offload generation to Celery; idempotent on job_id + request.
                generate_flashcards_task.apply_async(args=[job_id, req])
            elif user_provided_job_id:
                logger.info("WS using provided job_id; skipping generation | job_id=%s user_id=%s existing=%s", job_id, user_id, existing_count)
            inflight = FlashcardWorkflow.flashcard_inflight.get(job_id)

        session["seq"] = int(last_seq) if last_seq else 0
        session["job_id"] = job_id
        session["user_id"] = user_id
        session["in_flight_card"] = inflight["card_id"] if inflight else None

        await websocket.send_json({"message_type": "accepted", "job_id": job_id, "request_id": request_id})
        logger.info("WS accepted | job_id=%s user_id=%s seq=%s", job_id, user_id, session["seq"])

        async def send_card(card: Flashcard) -> None:
            session["seq"] += 1
            seq = session["seq"]
            session["in_flight_card"] = card.card_id
            async with FlashcardWorkflow.flashcard_lock:
                FlashcardWorkflow.flashcard_inflight[job_id] = {"seq": seq, "card_id": card.card_id}
                await FlashcardWorkflow.save_flashcards(job_id, FlashcardWorkflow.flashcard_store.get(job_id, {}))
            await websocket.send_json(
                {
                    "message_type": "card",
                    "seq": seq,
                    "job_id": job_id,
                    "kind": card.kind,
                    "card": {
                        "id": card.card_id,
                        "front": card.front,
                        "back": card.back,
                        "source_doc_id": card.source_doc_id,
                        "tags": card.tags,
                        "difficulty": card.difficulty,
                    },
                }
            )

        async def send_done() -> None:
            async with FlashcardWorkflow.flashcard_lock:
                cards = FlashcardWorkflow.flashcard_store.get(job_id, {})
            delivered_new, delivered_review = FlashcardWorkflow.flashcard_stats(cards)
            await websocket.send_json(
                {
                    "message_type": "done",
                    "job_id": job_id,
                    "delivered_new": delivered_new,
                    "delivered_review": delivered_review,
                }
            )

        # If reconnect and a card was in-flight, resend it with same seq.
        if session["in_flight_card"]:
            async with FlashcardWorkflow.flashcard_lock:
                cards = FlashcardWorkflow.flashcard_store.get(job_id, {})
                inflight_card = cards.get(session["in_flight_card"])
                inflight_info = FlashcardWorkflow.flashcard_inflight.get(job_id, {})
            if inflight_card and inflight_info:
                await websocket.send_json(
                    {
                        "message_type": "card",
                        "seq": inflight_info.get("seq", session["seq"]),
                        "job_id": job_id,
                        "kind": inflight_card.kind,
                        "card": {
                            "id": inflight_card.card_id,
                            "front": inflight_card.front,
                            "back": inflight_card.back,
                            "source_doc_id": inflight_card.source_doc_id,
                            "tags": inflight_card.tags,
                            "difficulty": inflight_card.difficulty,
                        },
                    }
                )
        else:
            async with FlashcardWorkflow.flashcard_lock:
                if to_generate:
                    # Wait briefly for background generation to finish.
                    refreshed = await FlashcardWorkflow.wait_for_cards(job_id, user_id)
                    if refreshed:
                        FlashcardWorkflow.flashcard_store[job_id] = refreshed
                        logger.info("WS refreshed after generation | job_id=%s cards=%s", job_id, len(refreshed))
                next_card = FlashcardWorkflow.select_next_due_card(FlashcardWorkflow.flashcard_store.get(job_id, {}))
            if next_card:
                await send_card(next_card)
            else:
                await websocket.send_json({"message_type": "idle", "job_id": job_id})
                logger.info("WS idle after subscribe | job_id=%s", job_id)
                # Wait a bit more for cards to appear before completing.
                refreshed = await FlashcardWorkflow.wait_for_cards(job_id, user_id, retries=4, delay=0.5)
                if refreshed:
                    FlashcardWorkflow.flashcard_store[job_id] = refreshed
                    next_card = FlashcardWorkflow.select_next_due_card(refreshed)
                    if next_card:
                        await send_card(next_card)
                        # fall through to main loop
                    else:
                        await send_done()
                        logger.info("WS done no card after refresh | job_id=%s", job_id)
                        return
                else:
                    await send_done()
                    logger.info("WS done no cards found | job_id=%s", job_id)
                    return

        # Main loop: wait for feedback, update, then send next.
        while True:
            raw_msg = await websocket.receive_text()
            try:
                message = json.loads(raw_msg) if raw_msg else {}
            except json.JSONDecodeError:
                await websocket.send_json({"error": "invalid JSON"})
                continue

            if message.get("message_type") != "card_feedback":
                await websocket.send_json({"error": "expected card_feedback"})
                continue

            feedback_seq = message.get("seq")
            card_id = message.get("card_id")
            rating = message.get("rating")
            if session["in_flight_card"] != card_id:
                logger.warning("WS feedback card mismatch | job_id=%s expected=%s got=%s", job_id, session["in_flight_card"], card_id)
                await websocket.send_json({"error": "card mismatch"})
                continue

            try:
                rating_int = int(rating)
            except (TypeError, ValueError):
                await websocket.send_json({"error": "rating must be int 0-2"})
                continue
            if rating_int not in {RatingScale.HARD.value, RatingScale.GOOD.value, RatingScale.EASY.value}:
                await websocket.send_json({"error": "rating must be 0 (hard), 1 (good), or 2 (easy)"})
                continue

            async with FlashcardWorkflow.flashcard_lock:
                cards = FlashcardWorkflow.flashcard_store.get(job_id, {})
                card = cards.get(card_id)
                if not card:
                    logger.warning("WS feedback card not found | job_id=%s card_id=%s", job_id, card_id)
                    await websocket.send_json({"error": "card not found"})
                    continue
                time_to_answer_ms = message.get("time_to_answer_ms")
                FlashcardWorkflow.update_card_schedule(card, rating_int, time_to_answer_ms=time_to_answer_ms)
                FlashcardWorkflow.flashcard_inflight.pop(job_id, None)
                session["in_flight_card"] = None
                await FlashcardWorkflow.save_flashcards(job_id, cards)
            logger.info(
                "WS feedback applied | job_id=%s card_id=%s rating=%s status=%s step=%s interval_days=%s due_at=%s",
                job_id,
                card_id,
                rating_int,
                card.status,
                card.learning_step_index,
                card.interval_days,
                card.due_at,
            )
            try:
                insert_review(
                    PROGRESS_DB_URL,
                    {
                        "card_id": card_id,
                        "user_id": user_id,
                        "job_id": job_id,
                        "rating": rating_int,
                        "time_to_answer_ms": message.get("time_to_answer_ms"),
                        "notes": message.get("notes"),
                        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    },
                )
            except Exception:
                logger.warning("Failed to insert review | card=%s job=%s", card_id, job_id, exc_info=True)

            await websocket.send_json(
                {
                    "message_type": "ack",
                    "card_id": card_id,
                    "seq": feedback_seq,
                    "job_id": job_id,
                    "next_due": card.due_at.isoformat(),
                    "interval_days": card.interval_days,
                    "ease_factor": round(card.ease_factor, 3),
                }
            )

            async with FlashcardWorkflow.flashcard_lock:
                next_card = FlashcardWorkflow.select_next_due_card(FlashcardWorkflow.flashcard_store.get(job_id, {}))
            if next_card:
                await send_card(next_card)
            else:
                # No card due; wait briefly in case new cards become due/generated.
                refreshed = await FlashcardWorkflow.wait_for_cards(job_id, user_id)
                if refreshed:
                    async with FlashcardWorkflow.flashcard_lock:
                        FlashcardWorkflow.flashcard_store[job_id] = refreshed
                    next_card = FlashcardWorkflow.select_next_due_card(refreshed)
                    if next_card:
                        await send_card(next_card)
                        continue
                await websocket.send_json({"message_type": "idle", "job_id": job_id})
                next_due = FlashcardWorkflow.next_due_seconds(FlashcardWorkflow.flashcard_store.get(job_id, {}))
                if next_due is not None and next_due <= SESSION_MAX_WAIT_SECONDS:
                    await asyncio.sleep(next_due)
                    continue
                await send_done()
                logger.info("WS loop done | job_id=%s delivered=%s", job_id, FlashcardWorkflow.flashcard_stats(FlashcardWorkflow.flashcard_store.get(job_id, {})))
                break

    except WebSocketDisconnect:
        logger.info("Flashcard websocket disconnected | job=%s", session.get("job_id"))
    except Exception:
        logger.warning("Flashcard websocket error", exc_info=True)
        try:
            await websocket.send_json({"error": "internal error"})
        except Exception:
            pass
