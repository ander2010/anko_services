from __future__ import annotations


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def build_ws_url(base_url: str, job_id: str) -> str:
    cleaned = normalize_base_url(base_url)
    scheme = "wss" if cleaned.startswith("https") else "ws"
    return f"{cleaned.replace('http', scheme, 1)}/ws/progress/{job_id}"
