from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from typing import AsyncIterator

import websockets

from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.append(str(EXAMPLES_DIR))
if str(EXAMPLES_DIR.parent) not in sys.path:
    sys.path.append(str(EXAMPLES_DIR.parent))

try:
    from examples.util.net import build_ws_url
except ImportError:  # pragma: no cover
    from util.net import build_ws_url


@dataclass
class ProgressStreamClient:
    base_url: str
    job_id: str

    @property
    def ws_url(self) -> str:
        return build_ws_url(self.base_url, self.job_id)

    async def messages(self) -> AsyncIterator[str]:
        """Yield raw websocket messages for the configured job."""
        async with websockets.connect(self.ws_url) as ws:
            async for msg in ws:
                yield msg

    async def run(self) -> None:
        print(f"Connecting to {self.ws_url}")
        try:
            async for msg in self.messages():
                print(msg)
        except KeyboardInterrupt:
            print("Interrupted by user; closing connection.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Follow live /ws/progress/<job_id> events.")
    parser.add_argument("--base-url", default="http://localhost:8080", help="FastAPI base URL.")
    parser.add_argument("--job-id", required=True, help="Job id to listen for")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    client = ProgressStreamClient(base_url=args.base_url, job_id=args.job_id)
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())
