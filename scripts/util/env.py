from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def load_env(env_path: Path | str = ".env", *, override: bool = False) -> None:
    """Lightweight .env loader for scripts."""
    path = Path(env_path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if override or key not in os.environ:
            os.environ[key] = value


def require_env(keys: Iterable[str]) -> None:
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
