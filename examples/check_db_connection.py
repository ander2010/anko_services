from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

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


def print_rows(cur, table: str, limit: int | None = 5) -> None:
    """Print all rows or a preview for the given table, with column names."""
    query = f"SELECT * FROM {table}" + (f" LIMIT {limit}" if limit is not None else "")
    cur.execute(query)
    rows = cur.fetchall()
    col_names = [desc[0] for desc in cur.description] if cur.description else []
    print(f"\nTable {table}: {len(rows)} row(s)" + ("" if limit is None else f" (showing up to {limit})"))
    if col_names:
        print("Columns:", ", ".join(col_names))
    for row in rows:
        if col_names and isinstance(row, tuple):
            print({col: row[idx] for idx, col in enumerate(col_names)})
        else:
            print(row)


def main() -> int:
    load_env()
    try:
        dsn = build_dsn()
    except Exception as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    try:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                row = cur.fetchone()
                print(f"Connection ok: {row}")

                cur.execute(
                    """
                    SELECT table_schema, table_name
                    FROM information_schema.tables
                    WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                    ORDER BY table_schema, table_name
                    """
                )
                tables = cur.fetchall()

                table_names: Iterable[str] = [name for _schema, name in tables]
                for name in ("documents", "chunks", "qa_pairs","tags","notifications","conversation_messages"):
                    if name in table_names:
                        print_rows(cur, name, limit=None)

        if tables:
            print("\nTables:")
            for schema, name in tables:
                print(f"- {schema}.{name}")
        else:
            print("No user tables found.")
        return 0
    except Exception as exc:
        print(f"Connection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
