from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path
import asyncio
import uuid
from typing import Optional
import sqlite3

import requests
from redis import Redis
import websockets

# Support running as a module (`python -m examples.ask_client`)
# or directly (`python examples/ask_client.py`) by handling imports both ways.
EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.append(str(EXAMPLES_DIR))
if str(EXAMPLES_DIR.parent) not in sys.path:
    sys.path.append(str(EXAMPLES_DIR.parent))

try:
    from examples.util.env import load_env
    from examples.util.net import normalize_base_url, build_ws_url
except ImportError:  # pragma: no cover - fallback for direct execution
    from util.env import load_env
    from util.net import normalize_base_url, build_ws_url


def _parse_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_float(value: str | None) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(value: str | None) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def get_local_args() -> Namespace:
    """Provide locally configured arguments instead of consuming CLI input."""
    load_env()
    base_url = os.getenv("ASK_BASE_URL", os.getenv("PROCESS_REQUEST_BASE_URL", "http://localhost:8080"))
    question = os.getenv("ASK_QUESTION")
    questions_env = os.getenv("ASK_QUESTIONS")
    context = _parse_list(os.getenv("ASK_CONTEXT", "test"))
    top_k = _parse_int(os.getenv("ASK_TOP_K"))
    min_importance = _parse_float(os.getenv("ASK_MIN_IMPORTANCE"))
    session_id = os.getenv("ASK_SESSION_ID") or str(uuid.uuid4())
    user_id = os.getenv("ASK_USER_ID")
    redis_url = os.getenv("CONVERSATION_REDIS_URL") or os.getenv("PROGRESS_REDIS_URL")
    progress_db = os.getenv("ASK_PROGRESS_DB", os.getenv("DB_URL"))
    return Namespace(
        base_url=base_url,
        question=question,
        questions_env=questions_env,
        context=context,
        top_k=top_k,
        min_importance=min_importance,
        session_id=session_id,
        user_id=user_id,
        redis_url=redis_url,
        progress_db=progress_db,
    )


def trigger_ask(args: Namespace, question: str) -> tuple[int, str | None, dict]:
    """Send the ask request and capture response details locally."""
    payload = {
        "question": question,
        "context": args.context,
        "top_k": args.top_k,
        "min_importance": args.min_importance,
        "session_id": args.session_id,
        "user_id": args.user_id,
    }

    url = f"{normalize_base_url(args.base_url)}/ask"
    response = requests.post(url, json=payload, timeout=120)

    result: dict = {
        "request_url": url,
        "status_code": response.status_code,
        "request_payload": payload,
    }

    try:
        response_json = response.json()
        result["response_json"] = response_json
    except Exception:
        response_json = None
        result["response_text"] = response.text

    job_id: str | None = None
    if isinstance(response_json, dict):
        job_id = response_json.get("job_id") or job_id
        if job_id and response.ok:
            result["ws_url"] = build_ws_url(args.base_url, job_id)
    result["job_id"] = job_id

    return (0 if response.ok else 1), job_id, result


def load_notification_from_db(db_path: str | None, job_id: str | None) -> dict | None:
    """Read the notifications table for a given job (SQLite only)."""
    if not db_path or not job_id:
        return None
    # Only handle SQLite files here.
    if str(db_path).startswith("postgres"):
        return None
    path = Path(db_path)
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(path)
        try:
            cur = conn.execute("SELECT metadata FROM notifications WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            try:
                return json.loads(row[0]) if row[0] else {}
            except Exception:
                return None
        finally:
            conn.close()
    except Exception:
        return None


async def receive_final(ws: websockets.WebSocketClientProtocol) -> dict:
    """Wait for a terminal message on an open websocket."""
    done_status = {"COMPLETED", "FAILED", "ERROR"}
    async for msg in ws:
        try:
            data = json.loads(msg)
        except Exception:
            return {"error": "invalid websocket payload", "raw": msg}
        if data.get("type") == "final" or data.get("status", "").upper() in done_status:
            return data
        if data.get("error"):
            return data
    return {"error": "websocket closed without final response"}


async def main_async() -> int:
    args = get_local_args()
    questions: list[str] = []
    if args.questions_env:
        questions = [q for q in _parse_list(args.questions_env)]
    elif args.question:
        questions = [args.question]

    print(f"Using session_id={args.session_id} user_id={args.user_id or 'anonymous'} redis={args.redis_url or 'default'}")
    history_key = f"conversation:{args.session_id}"
    client: Redis | None = None
    if args.redis_url:
        try:
            client = Redis.from_url(args.redis_url, decode_responses=True)
        except Exception:
            client = None

    chat_ws_url = f"{normalize_base_url(args.base_url).replace('http', 'ws', 1)}/ws/chat/{args.session_id}"
    ws: websockets.WebSocketClientProtocol | None = None

    async def ensure_ws() -> websockets.WebSocketClientProtocol | None:
        nonlocal ws
        if ws and not ws.close:
            return ws
        try:
            ws = await websockets.connect(chat_ws_url, ping_interval=None)
            print(f"Connected to chat websocket: {chat_ws_url}")
            return ws
        except Exception as exc:
            print(f"Chat websocket connect failed: {exc}")
            ws = None
            return None

    idx = 0
    while True:
        if not questions:
            try:
                q = input("\nEnter a question (blank to exit): ").strip()
            except EOFError:
                break
            if not q:
                break
        else:
            q = questions.pop(0)

        idx += 1
        print(f"\n--- Round {idx} ---")
        chat_payload = {
            "question": q,
            "context": args.context,
            "top_k": args.top_k,
            "min_importance": args.min_importance,
            "user_id": args.user_id,
        }

        job_id: str | None = None
        ws_conn = await ensure_ws()
        if ws_conn:
            try:
                await ws_conn.send(json.dumps(chat_payload))
                final_msg = await receive_final(ws_conn)
                print(f"job_id={final_msg.get('job_id', 'n/a')}")
                print(json.dumps(final_msg, indent=2))
                job_id = final_msg.get("job_id")
            except Exception as exc:
                print(f"Websocket send/receive failed, will fall back to HTTP /ask: {exc}")
                try:
                    await ws_conn.close()
                except Exception:
                    pass
                ws = None

        if job_id is None:
            exit_code, job_id, result = trigger_ask(args, q)
            print(f"job_id={job_id or 'n/a'}")
            print(json.dumps(result, indent=2))
            if exit_code != 0:
                if job_id:
                    db_snapshot = load_notification_from_db(args.progress_db, job_id)
                    if db_snapshot:
                        print("DB snapshot fallback:")
                        print(json.dumps({"job_id": job_id, **db_snapshot}, indent=2))
                return exit_code

        if client:
            try:
                history_len = client.llen(history_key)
                print(f"Redis history length for {history_key}: {history_len}")
                if history_len:
                    latest = client.lrange(history_key, -3, -1)
                    parsed = []
                    for item in latest:
                        try:
                            parsed.append(json.loads(item))
                        except Exception:
                            parsed.append({"raw": item})
                    print("Recent redis history entries:")
                    print(json.dumps(parsed, indent=2))
            except Exception:
                print("Could not read conversation history from Redis (check DB index/URL)")

    if ws and not  ws.close:
        try:
            await ws.close()
        except Exception:
            pass
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
