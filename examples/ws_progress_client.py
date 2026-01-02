from __future__ import annotations

import asyncio
import os
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


def get_config() -> ProgressStreamClient:
    job_id = (sys.argv[1] if len(sys.argv) > 1 else os.getenv("JOB_ID", "")).strip()
    if not job_id:
        sys.exit("JOB_ID is required (env or first arg).")
    base_url = os.getenv("PROGRESS_BASE_URL", os.getenv("PROCESS_REQUEST_BASE_URL", "http://localhost:8080"))
    return ProgressStreamClient(base_url=base_url, job_id=job_id)


async def main() -> None:
    client = get_config()
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())
