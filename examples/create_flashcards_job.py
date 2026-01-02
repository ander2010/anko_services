import asyncio
import json
import uuid

import httpx

BASE_URL = "http://localhost:8080"


async def main():
    user_id = str(uuid.uuid4())
    payload = {
        "document_ids": ["test"],
        "tags": ["Barcelona"],
        "quantity": 10,
        "difficulty": "medium",
        "user_id": user_id,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{BASE_URL}/flashcards/create", json=payload)
        resp.raise_for_status()
        data = resp.json()
        job_id = data["job_id"]
        ws_flashcards = f"ws://{BASE_URL.split('://', 1)[-1].lstrip('/')}/ws/flashcards/{job_id}"
        ws_progress = f"ws://{BASE_URL.split('://', 1)[-1].lstrip('/')}/ws/progress/{job_id}"
        print(json.dumps({**data, "ws_flashcards": ws_flashcards, "ws_progress": ws_progress}, indent=2))
        print(f"Created flashcards for user {user_id}. Use job_id={job_id} with the learn example.")


if __name__ == "__main__":
    asyncio.run(main())
