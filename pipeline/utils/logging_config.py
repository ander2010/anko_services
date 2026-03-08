from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock

_LOGGING_LOCK = Lock()
_LOGGING_CONFIGURED = False


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


def configure_logging() -> None:
    """Configure app-wide logging once with stdout + rotating file handlers."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    with _LOGGING_LOCK:
        if _LOGGING_CONFIGURED:
            return

        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, level_name, logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s"
        )

        handlers: list[logging.Handler] = []
        if _env_bool("LOG_TO_STDOUT", True):
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            handlers.append(stream_handler)

        if _env_bool("LOG_TO_FILE", True):
            log_dir = Path(os.getenv("LOG_DIR", "logs"))
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file_name = os.getenv("LOG_FILE_NAME", "pipeline.log")
            log_file_path = log_dir / log_file_name
            max_bytes = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))
            backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
            file_handler = RotatingFileHandler(
                filename=log_file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
                delay=True,
            )
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        for handler in handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(log_level)

        _LOGGING_CONFIGURED = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create or reuse a module-level logger using central logging settings."""
    configure_logging()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


__all__ = ["get_logger", "configure_logging"]
