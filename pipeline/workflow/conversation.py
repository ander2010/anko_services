import json
import os
from datetime import datetime
from typing import Iterable, Sequence

from redis import Redis as SyncRedis
from redis.asyncio import Redis as AsyncRedis

# Prefer a dedicated conversation Redis URL when available (e.g., CONVERSATION_REDIS_URL for dev),
# otherwise fall back to the progress Redis endpoint.
REDIS_URL = os.getenv("CONVERSATION_REDIS_URL") or os.getenv("PROGRESS_REDIS_URL", "redis://localhost:6379/2")
CONVERSATION_TTL_SECONDS = int(os.getenv("CONVERSATION_TTL_SECONDS", "86400"))
CONVERSATION_MAX_TOKENS = int(os.getenv("CONVERSATION_MAX_TOKENS", "1800"))
CONVERSATION_MAX_MESSAGES = int(os.getenv("CONVERSATION_MAX_MESSAGES", "30"))

_async_client: AsyncRedis | None = None
_sync_client: SyncRedis | None = None


def _conversation_key(session_id: str) -> str:
    return f"conversation:{session_id}"


def _estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, len(str(text).split()))


async def get_async_client() -> AsyncRedis:
    global _async_client
    if _async_client is None:
        _async_client = AsyncRedis.from_url(REDIS_URL, decode_responses=True)
    return _async_client


def get_sync_client() -> SyncRedis:
    global _sync_client
    if _sync_client is None:
        _sync_client = SyncRedis.from_url(REDIS_URL, decode_responses=True)
    return _sync_client


def _build_entry(session_id: str, user_id: str | None, question: str, answer: str) -> dict:
    question_clean = (question or "").strip()
    answer_clean = (answer or "").strip()
    return {
        "session_id": session_id,
        "user_id": (user_id or "").strip() or None,
        "question": question_clean,
        "answer": answer_clean,
        "tokens": _estimate_tokens(question_clean) + _estimate_tokens(answer_clean),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }


def _parse_entries(raw_entries: Iterable[str]) -> list[dict]:
    parsed: list[dict] = []
    for raw in raw_entries:
        try:
            item = json.loads(raw)
            if isinstance(item, dict):
                parsed.append(item)
        except Exception:
            continue
    return parsed


def _apply_token_window(entries: Sequence[dict], token_budget: int) -> list[dict]:
    """Keep the newest entries that fit within the token budget."""
    if token_budget <= 0:
        return []

    collected: list[dict] = []
    total_tokens = 0
    for entry in reversed(entries):
        tokens = int(entry.get("tokens") or 0)
        if tokens <= 0:
            tokens = _estimate_tokens(entry.get("question")) + _estimate_tokens(entry.get("answer"))
            entry["tokens"] = tokens
        if tokens <= 0:
            continue

        if total_tokens + tokens > token_budget:
            if not collected:
                collected.append(entry)
            break

        collected.append(entry)
        total_tokens += tokens

    collected.reverse()
    return collected


async def append_message_async(session_id: str | None, user_id: str | None, question: str, answer: str) -> None:
    session_clean = (session_id or "").strip()
    if not session_clean:
        return

    client = await get_async_client()
    entry = _build_entry(session_clean, user_id, question, answer)
    key = _conversation_key(session_clean)
    await client.rpush(key, json.dumps(entry))
    await client.expire(key, CONVERSATION_TTL_SECONDS)


def append_message(session_id: str | None, user_id: str | None, question: str, answer: str) -> None:
    session_clean = (session_id or "").strip()
    if not session_clean:
        return

    client = get_sync_client()
    entry = _build_entry(session_clean, user_id, question, answer)
    key = _conversation_key(session_clean)
    client.rpush(key, json.dumps(entry))
    client.expire(key, CONVERSATION_TTL_SECONDS)


async def fetch_recent_async(session_id: str | None, token_budget: int | None = None, max_items: int | None = None) -> list[dict]:
    session_clean = (session_id or "").strip()
    if not session_clean:
        return []

    client = await get_async_client()
    key = _conversation_key(session_clean)
    history = await client.lrange(key, -(max_items or CONVERSATION_MAX_MESSAGES), -1)
    parsed = _parse_entries(history)
    windowed = _apply_token_window(parsed, token_budget or CONVERSATION_MAX_TOKENS)

    # Trim Redis list to match the window so older items fall off.
    if parsed and len(windowed) < len(parsed):
        keep = len(windowed)
        # Always keep at least one entry if history exists.
        if keep == 0:
            windowed = parsed[-1:]
            keep = 1
        try:
            await client.ltrim(key, -keep, -1)
        except Exception:
            # Non-blocking trim failure; continue returning windowed data.
            pass

    return windowed


def fetch_recent(session_id: str | None, token_budget: int | None = None, max_items: int | None = None) -> list[dict]:
    session_clean = (session_id or "").strip()
    if not session_clean:
        return []

    client = get_sync_client()
    key = _conversation_key(session_clean)
    history = client.lrange(key, -(max_items or CONVERSATION_MAX_MESSAGES), -1)
    parsed = _parse_entries(history)
    windowed = _apply_token_window(parsed, token_budget or CONVERSATION_MAX_TOKENS)

    if parsed and len(windowed) < len(parsed):
        keep = len(windowed)
        if keep == 0:
            windowed = parsed[-1:]
            keep = 1
        try:
            client.ltrim(key, -keep, -1)
        except Exception:
            pass

    return windowed


def format_history(entries: Sequence[dict]) -> str:
    blocks: list[str] = []
    for entry in entries:
        question = (entry.get("question") or "").strip()
        answer = (entry.get("answer") or "").strip()
        if question and answer:
            blocks.append(f"User: {question}\nAssistant: {answer}")
        elif question:
            blocks.append(f"User: {question}")
        elif answer:
            blocks.append(f"Assistant: {answer}")
    return "\n\n".join(blocks)
