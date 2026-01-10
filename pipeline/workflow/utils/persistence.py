from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
from typing import Sequence

from pipeline.workflow.knowledge_store import LocalKnowledgeStore
from pipeline.utils.logging_config import get_logger
from pipeline.utils.types import ChunkEmbedding

logger = get_logger(__name__)

_NOTIFICATION_BATCH_SIZE = 100
_notification_queue: Queue["_NotificationItem"] = Queue()
_notification_worker: Thread | None = None
_notification_stop = Event()
_FINAL_STATUSES = {"COMPLETED", "DONE", "FAILED", "ERROR", "CANCELLED", "CANCELED"}


@dataclass
class _NotificationItem:
    db_path: str
    job_id: str
    metadata: dict
    finalize: bool = False


def _start_notification_worker() -> None:
    global _notification_worker
    _notification_stop.clear()
    if _notification_worker and _notification_worker.is_alive():
        return
    _notification_worker = Thread(target=_notification_worker_loop, name="notification-batcher", daemon=True)
    _notification_worker.start()


def _flush_notification_buffers(buffers: dict[str, dict[str, dict]]) -> None:
    for db_path, pending in list(buffers.items()):
        if not pending:
            continue
        items = list(pending.items())
        try:
            with LocalKnowledgeStore(db_path) as store:
                store.save_notifications(items)
            logger.debug("Flushed %s notifications | db=%s", len(items), db_path)
        except Exception:
            logger.warning("Failed to flush notifications | db=%s", db_path, exc_info=True)
        buffers[db_path] = {}


def _notification_worker_loop() -> None:
    buffers: dict[str, dict[str, dict]] = {}
    while not _notification_stop.is_set():
        try:
            item = _notification_queue.get(timeout=1.0)
            if item.job_id == "__flush_all__":
                _flush_notification_buffers(buffers)
                buffers.clear()
                continue
            db_buffer = buffers.setdefault(item.db_path, {})
            db_buffer[item.job_id] = item.metadata
            if item.finalize:
                _flush_notification_buffers({item.db_path: dict(db_buffer)})
                db_buffer.clear()
                continue
        except Empty:
            continue

        total_pending = sum(len(buf) for buf in buffers.values())
        if total_pending >= _NOTIFICATION_BATCH_SIZE:
            _flush_notification_buffers(buffers)


def shutdown_notification_worker(flush: bool = True) -> None:
    """Signal the background worker to stop; optionally flush pending notifications."""
    if flush:
        _notification_queue.put_nowait(_NotificationItem(db_path="", job_id="__flush_all__", metadata={}, finalize=True))
    _notification_stop.set()
    if _notification_worker and _notification_worker.is_alive():
        _notification_worker.join(timeout=1.0)


def save_document(db_path, document_id: str, source_path: str, embeddings: Sequence[ChunkEmbedding], qa_pairs, *, allow_overwrite: bool = True, job_id=None):
    """Persist embeddings and QA pairs with consistent logging and error handling."""
    try:
        with LocalKnowledgeStore(db_path) as store:
            store.save_document(document_id, source_path, embeddings, qa_pairs, allow_overwrite=allow_overwrite, job_id=job_id)
            logger.info("Persisted document | job=%s doc=%s db=%s embeddings=%s qa_pairs=%s", job_id, document_id, db_path, len(embeddings), len(qa_pairs))
    except Exception:
        logger.warning("Failed to persist document | job=%s doc=%s db=%s", job_id, document_id, db_path, exc_info=True)


def update_chunk_question_ids(db_path, document_id: str, updates: dict):
    try:
        with LocalKnowledgeStore(db_path) as store:
            store.update_chunk_question_ids(document_id, updates)
    except Exception:
        logger.warning("Failed to update chunk question ids | doc=%s", document_id, exc_info=True)


def save_notification(db_path, job_id: str, metadata: dict):
    status = str((metadata or {}).get("status", "")).upper()
    if status not in _FINAL_STATUSES and "FAIL" not in status and "ERROR" not in status:
        return
    try:
        with LocalKnowledgeStore(db_path) as store:
            store.save_notification(job_id, metadata or {})
            logger.info("Saved notification | job=%s db=%s status=%s", job_id, db_path, status or "n/a")
    except Exception:
        logger.warning("Failed to save notification | job=%s db=%s", job_id, db_path, exc_info=True)


def save_notification_async(db_path, job_id: str, metadata: dict):
    """Fire-and-forget notification save only for final/error states."""
    status = str((metadata or {}).get("status", "")).upper()
    if not job_id or (status not in _FINAL_STATUSES and "FAIL" not in status and "ERROR" not in status):
        return
    try:
        _start_notification_worker()
        meta = metadata or {}
        finalize = True
        _notification_queue.put_nowait(_NotificationItem(str(db_path), job_id, meta, finalize))
    except Exception:
        logger.warning("Failed to enqueue async notification | job=%s db=%s", job_id, db_path, exc_info=True)


def save_tags(db_path, document_id: str, tags, job_id: str | None = None):
    try:
        with LocalKnowledgeStore(db_path) as store:
            store.save_tags(document_id, tags, job_id=job_id)
            logger.info("Saved tags | doc=%s job=%s tags=%s", document_id, job_id or "n/a", len(tags or []))
    except Exception:
        logger.warning("Failed to save tags | doc=%s job=%s", document_id, job_id or "n/a", exc_info=True)


def save_conversation_message(db_path, session_id: str, user_id: str | None, job_id: str | None, question: str, answer: str):
    """Persist a conversation turn to the configured knowledge store."""
    try:
        with LocalKnowledgeStore(db_path) as store:
            store.save_conversation_message(session_id, user_id, job_id, question, answer)
            logger.info("Saved conversation | session=%s user=%s job=%s", session_id, user_id or "anonymous", job_id or "n/a")
    except Exception:
        logger.warning("Failed to save conversation | session=%s user=%s job=%s", session_id, user_id or "anonymous", job_id or "n/a", exc_info=True)


def load_notification(db_path, job_id: str) -> dict | None:
    try:
        with LocalKnowledgeStore(db_path) as store:
            return store.load_notification(job_id)
    except Exception:
        logger.warning("Failed to load notification | job=%s", job_id, exc_info=True)
        return None
