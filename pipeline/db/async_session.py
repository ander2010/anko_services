from __future__ import annotations

from pathlib import Path
from typing import Tuple

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


def _ensure_sqlite_dirs(url: str) -> None:
    if not url.startswith("sqlite+aiosqlite:///"):
        return
    path = url.replace("sqlite+aiosqlite:///", "", 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def build_sqlite_async_url(db_path: str | Path) -> str:
    path = Path(db_path)
    return f"sqlite+aiosqlite:///{path}"


def create_async_engine_and_session(db_url: str) -> Tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    _ensure_sqlite_dirs(db_url)
    engine = create_async_engine(db_url, future=True, echo=False)
    SessionLocal = async_sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    return engine, SessionLocal
