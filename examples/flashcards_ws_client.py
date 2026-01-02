import asyncio
import json
import uuid

import websockets
from websockets import exceptions as ws_exceptions


BASE_URL = "http://localhost:8080"
WS_URL = "ws://localhost:8080/ws/flashcards"


async def main():
    # Step 1: start a job via HTTP
    import httpx
    user_id = str(uuid.uuid4())
    async with httpx.AsyncClient() as client:
        start_payload = {
            "document_ids": ["test"],
            "tags": ["Barcelona","soccer","foods"],
            "quantity": 8,
            "difficulty": "medium",
            "user_id": user_id,
        }
        resp = await client.post(f"{BASE_URL}/study/start", json=start_payload)
        resp.raise_for_status()
        start_data = resp.json()
        job_id = start_data["job_id"]
        token = start_data["token"]
        print(f"Started job {job_id}\nToken: {token}\n")

    # Step 2: open websocket and subscribe (token sent in payload)
    try:
        async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=60) as ws:
            await ws.send(
                json.dumps(
                    {
                        "message_type": "subscribe_job",
                        "job_id": job_id,
                        "user_id": user_id,
                        "request_id": str(uuid.uuid4()),
                        "last_seq": 0,
                        "token": token,
                    }
                )
            )

            async for raw in ws:
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    print(f"<< raw: {raw}")
                    continue

                print(f"<< {message}")

                if message.get("message_type") == "card":
                    card = message["card"]
                    # Let the user decide rating without blocking the event loop.
                    try:
                        rating = await asyncio.wait_for(
                            asyncio.to_thread(input, f"Rate card {card['id']} (0=hard, 1=good, 2=easy, -1=exit, default=1): "),
                            timeout=30,
                        )
                        rating_int = int(rating)
                    except asyncio.TimeoutError:
                        print("No input provided in time, defaulting rating to 1 (good).")
                        rating_int = 1
                    except Exception:
                        rating_int = 1
                    if rating_int < 0:
                        print("Client requested close.")
                        await ws.close(reason="client_requested_close")
                        break
                    await ws.send(
                        json.dumps(
                            {
                                "message_type": "card_feedback",
                                "seq": message.get("seq"),
                                "job_id": job_id,
                                "card_id": card["id"],
                                "rating": rating_int,
                                "time_to_answer_ms": 1000,
                            }
                        )
                    )

                if message.get("message_type") in {"done", "error"}:
                    print("Server signaled completion, closing.")
                    break
    except ws_exceptions.ConnectionClosed as exc:
        print(f"WebSocket closed by server: code={exc.code} reason={exc.reason}")
    except Exception as exc:
        print(f"WebSocket client error: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
