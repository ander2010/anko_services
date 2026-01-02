from __future__ import annotations

from pathlib import Path
from typing import Tuple

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker


def _ensure_sqlite_dirs(url: str) -> None:
    if not url.startswith("sqlite:///"):
        return
    path = url.replace("sqlite:///", "", 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def build_sqlite_url(db_path: str | Path) -> str:
    path = Path(db_path)
    return f"sqlite:///{path}"


def normalize_db_url(db_url: str) -> str:
    # Normalize postgres scheme and prefer psycopg driver if available.
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    if db_url.startswith("postgresql://") and "+psycopg" not in db_url and "+psycopg2" not in db_url:
        try:
            import psycopg  # type: ignore
            db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
        except Exception:
            pass
    return db_url


def create_engine_and_session(db_url: str) -> Tuple[Engine, sessionmaker]:
    db_url = normalize_db_url(db_url)
    _ensure_sqlite_dirs(db_url)
    engine = create_engine(db_url, future=True, echo=False)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)
    return engine, SessionLocal
