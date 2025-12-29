import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import psycopg

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.append(str(EXAMPLES_DIR))
if str(EXAMPLES_DIR.parent) not in sys.path:
    sys.path.append(str(EXAMPLES_DIR.parent))

try:
    from examples.util.env import load_env
except ImportError:  # pragma: no cover
    from util.env import load_env


def build_dsn() -> str:
    """Use DB_URL if set; otherwise assemble from component env vars."""
    url = os.getenv("DB_URL")
    if url:
        return url

    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME")

    missing = [var for var, val in [("DB_USER", user), ("DB_PASSWORD", password), ("DB_NAME", name)] if not val]
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)} (or set DB_URL)")

    sslmode = os.getenv("DB_SSLMODE")
    auth = f"{user}:{password}@"
    base = f"postgresql://{auth}{host}:{port}/{name}"
    return f"{base}?sslmode={sslmode}" if sslmode else base


def parse_tables(raw: str | None) -> Sequence[str]:
    if not raw:
        return (("documents", "chunks", "qa_pairs","tags","notifications"))
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if not parts:
        return (("documents", "chunks", "qa_pairs","tags","notifications"))
    # Simple validation: allow alphanumerics and underscores only.
    for name in parts:
        if not name.replace("_", "").isalnum():
            raise ValueError(f"Invalid table name: {name!r}")
    return parts


def truncate_tables(conn, tables: Iterable[str]) -> None:
    joined = ", ".join(tables)
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE {joined} RESTART IDENTITY CASCADE;")
    conn.commit()


def main() -> int:
    load_env()
    try:
        dsn = build_dsn()
        tables = parse_tables(os.getenv("TRUNCATE_TABLES"))
    except Exception as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    try:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            truncate_tables(conn, tables)
        print(f"Truncated tables: {', '.join(tables)}")
        return 0
    except Exception as exc:
        print(f"Truncate failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
