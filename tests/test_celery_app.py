from __future__ import annotations

import pytest
from redis.exceptions import ReadOnlyError

import celery_app


def test_build_celery_config_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CELERY_BROKER_CHANNEL_ERROR_RETRY", "false")
    monkeypatch.setenv("CELERY_BROKER_MAX_RETRIES", "7")
    monkeypatch.setenv("CELERY_BROKER_HEALTHCHECK_INTERVAL", "11")
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "true")
    monkeypatch.setenv("CELERY_TASK_TIME_LIMIT", "123")
    monkeypatch.setenv("CELERY_DEFAULT_QUEUE", "default-q")
    monkeypatch.setenv("CELERY_OCR_QUEUE", "ocr-q")
    monkeypatch.setenv("CELERY_WORKER_CANCEL_LONG_RUNNING_TASKS_ON_CONNECTION_LOSS", "true")
    monkeypatch.setenv("CELERY_WORKER_ENABLE_REMOTE_CONTROL", "false")
    monkeypatch.setenv("CELERY_WORKER_SEND_TASK_EVENTS", "true")
    monkeypatch.setenv("CELERY_REDIS_VISIBILITY_TIMEOUT", "600")

    config = celery_app.build_celery_config()

    assert config["broker_channel_error_retry"] is False
    assert config["broker_connection_max_retries"] == 7
    assert config["broker_transport_options"] == {
        "health_check_interval": 11,
        "retry_on_timeout": True,
        "visibility_timeout": 600,
    }
    assert config["task_always_eager"] is True
    assert config["task_time_limit"] == 123
    assert config["task_default_queue"] == "default-q"
    assert config["task_routes"]["pipeline.ocr.pages"]["queue"] == "ocr-q"
    assert config["worker_cancel_long_running_tasks_on_connection_loss"] is True
    assert config["worker_enable_remote_control"] is False
    assert config["worker_send_task_events"] is True


def test_validate_redis_broker_write_access_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class ReadOnlyClient:
        def ping(self) -> bool:
            return True

        def set(self, *args, **kwargs) -> None:
            raise ReadOnlyError("You can't write against a read only replica.")

        def delete(self, *args, **kwargs) -> None:
            raise AssertionError("delete should not be called after a read-only failure")

    class FakeRedis:
        @staticmethod
        def from_url(*args, **kwargs) -> ReadOnlyClient:
            return ReadOnlyClient()

    monkeypatch.setattr(celery_app, "Redis", FakeRedis)

    with pytest.raises(RuntimeError, match="read-only"):
        celery_app.validate_redis_broker_write_access("redis://hope-redis:6379/0", timeout_seconds=0.5)
