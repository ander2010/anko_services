import asyncio
import json
import os
import uuid

import websockets
from websockets import exceptions as ws_exceptions

BASE_WS = os.getenv("FLASHCARD_WS_BASE", "ws://localhost:8080/ws/flashcards")


async def main():
    job_id = os.getenv("FLASHCARD_JOB_ID") or "5c83356a-0ec5-544b-bec9-ddfc74db6e04"
    user_id = os.getenv("FLASHCARD_USER_ID") or str(uuid.uuid4())
    if not job_id or "REPLACE_WITH" in job_id:
        raise SystemExit("Set FLASHCARD_JOB_ID to a valid job id (e.g., from create_flashcards_job.py output).")

    ws_url = BASE_WS.rstrip("/") + f"/{job_id}"
    print(f"Connecting to {ws_url} as user {user_id}")

    try:
        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=100) as ws:
            await ws.send(
                json.dumps(
                    {
                        "message_type": "subscribe_job",
                        "job_id": job_id,
                        "user_id": user_id,
                        "request_id": str(uuid.uuid4()),
                        "last_seq": 0,
                        "token": "",
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
                    # Prompt user asynchronously for rating: 0 hard, 1 good, 2 easy, -1 exit.
                    try:
                        rating_str = await asyncio.wait_for(
                            asyncio.to_thread(
                                input,
                                f"Rate card {card['id']} (0=hard, 1=good, 2=easy, -1=exit, default=1): "
                            ),
                            timeout=45,
                        )
                        rating_int = int(rating_str)
                    except asyncio.TimeoutError:
                        print("No input received, defaulting to 1 (good).")
                        rating_int = 1
                    except Exception:
                        rating_int = 1

                    if rating_int == -1:
                        print("User requested exit.")
                        await ws.close(reason="client_exit")
                        break

                    await ws.send(
                        json.dumps(
                            {
                                "message_type": "card_feedback",
                                "seq": message.get("seq"),
                                "job_id": job_id,
                                "card_id": card["id"],
                                "rating": rating_int,
                                "time_to_answer_ms": 500,
                            }
                        )
                    )

                if message.get("message_type") in {"done", "error"}:
                    print("Server signaled completion, closing.")
                    break
    except ws_exceptions.ConnectionClosed as exc:
        print(f"WebSocket closed: code={exc.code} reason={exc.reason}")
    except Exception as exc:
        print(f"WebSocket error: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
