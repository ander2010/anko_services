from __future__ import annotations

import argparse
import asyncio

import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Follow live /ws/progress/<job_id> events.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="FastAPI base URL.")
    parser.add_argument("--job-id", required=True, help="Job id to listen for")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    ws_url = f"{args.base_url.rstrip('/').replace('http', 'ws')}/ws/progress/{args.job_id}"
    print(f"Connecting to {ws_url}")
    async with websockets.connect(ws_url) as ws:
        async for msg in ws:
            print(msg)


if __name__ == "__main__":
    asyncio.run(main())
