from __future__ import annotations

import logging
import os
import uuid
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from celery import Celery
from redis import Redis
from redis.exceptions import ReadOnlyError, RedisError
from pipeline.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in {None, ""}:
        return default
    return int(value)


def env_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    if value in {None, "", "None"}:
        return None
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in {None, ""}:
        return default
    return float(value)


def redact_url(url: str) -> str:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return url

    username = parts.username or ""
    hostname = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""

    auth = f"{username}:***@" if username and parts.password else (f"{username}@" if username else "")
    netloc = f"{auth}{hostname}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def validate_redis_broker_write_access(url: str, timeout_seconds: float) -> None:
    parts = urlsplit(url)
    if parts.scheme not in {"redis", "rediss"}:
        return

    client = Redis.from_url(
        url,
        socket_connect_timeout=timeout_seconds,
        socket_timeout=timeout_seconds,
        health_check_interval=0,
        retry_on_timeout=True,
        decode_responses=True,
    )
    probe_key = f"celery:broker:startup-check:{uuid.uuid4()}"
    broker_label = redact_url(url)

    try:
        client.ping()
        client.set(probe_key, "1", ex=max(5, int(timeout_seconds * 2)))
        client.delete(probe_key)
    except ReadOnlyError as exc:
        raise RuntimeError(
            f"Celery broker {broker_label} is read-only. Point CELERY_BROKER_URL at the writable Redis primary."
        ) from exc
    except RedisError as exc:
        raise RuntimeError(f"Celery broker startup validation failed for {broker_label}: {exc}") from exc


def build_celery_config() -> dict[str, Any]:
    broker_max_retries = os.getenv("CELERY_BROKER_MAX_RETRIES")
    broker_healthcheck_interval = env_int("CELERY_BROKER_HEALTHCHECK_INTERVAL", 30)
    broker_channel_error_retry = env_bool("CELERY_BROKER_CHANNEL_ERROR_RETRY", True)
    task_always_eager = env_bool("CELERY_TASK_ALWAYS_EAGER", False)
    worker_cancel_on_connection_loss = env_bool(
        "CELERY_WORKER_CANCEL_LONG_RUNNING_TASKS_ON_CONNECTION_LOSS",
        False,
    )
    worker_enable_remote_control = env_bool("CELERY_WORKER_ENABLE_REMOTE_CONTROL", True)
    worker_send_task_events = env_bool("CELERY_WORKER_SEND_TASK_EVENTS", False)
    redis_visibility_timeout = env_optional_int("CELERY_REDIS_VISIBILITY_TIMEOUT")

    broker_transport_options: dict[str, Any] = {
        "health_check_interval": broker_healthcheck_interval,
        "retry_on_timeout": True,
    }
    if redis_visibility_timeout is not None:
        broker_transport_options["visibility_timeout"] = redis_visibility_timeout

    return {
        "task_serializer": "json",
        "result_serializer": "json",
        "accept_content": ["json"],
        "task_track_started": True,
        "worker_prefetch_multiplier": 1,
        "broker_connection_retry_on_startup": True,
        "broker_connection_retry": True,
        "broker_channel_error_retry": broker_channel_error_retry,
        "broker_connection_max_retries": None if broker_max_retries in {None, "", "None"} else int(broker_max_retries),
        "broker_transport_options": broker_transport_options,
        "task_time_limit": env_int("CELERY_TASK_TIME_LIMIT", 3600),
        "task_always_eager": task_always_eager,
        "task_eager_propagates": True,
        "task_default_queue": os.getenv("CELERY_DEFAULT_QUEUE", "celery"),
        "worker_cancel_long_running_tasks_on_connection_loss": worker_cancel_on_connection_loss,
        "worker_enable_remote_control": worker_enable_remote_control,
        "worker_send_task_events": worker_send_task_events,
        "task_routes": {
            "pipeline.ocr.pages": {"queue": os.getenv("CELERY_OCR_QUEUE", "ocr")},
            "pipeline.ocr.paragraphs": {"queue": os.getenv("CELERY_OCR_QUEUE", "ocr")},
        },
    }


CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", CELERY_BROKER_URL)
CELERY_CONFIG = build_celery_config()

if env_bool("CELERY_VALIDATE_BROKER_ON_STARTUP", False):
    validate_redis_broker_write_access(
        CELERY_BROKER_URL,
        timeout_seconds=env_float("CELERY_BROKER_STARTUP_CHECK_TIMEOUT", 5.0),
    )
    logger.info("Celery broker startup validation passed for %s", redact_url(CELERY_BROKER_URL))

celery_app = Celery(
    "pipeline",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        "pipeline.celery_tasks.validate",
        "pipeline.celery_tasks.ocr",
        "pipeline.celery_tasks.embedding",
        "pipeline.celery_tasks.llm",
        "pipeline.celery_tasks.flashcards",
        "pipeline.celery_tasks.prepare",
    ],
)

celery_app.conf.update(**CELERY_CONFIG)
