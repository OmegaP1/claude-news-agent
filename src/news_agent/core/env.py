"""Minimal .env loader.

A dependency-free stand-in for python-dotenv — we need ~15 lines of it, and
keeping the runtime deps at `anthropic` + `pydantic` is the point of this
project. Real environment variables always win over the file, so
`ANTHROPIC_API_KEY=... python -m news_agent` behaves the way you'd expect.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_dotenv(start: Path | None = None) -> Path | None:
    """Walk up from `start` looking for a .env, so the CLI works from any
    subdirectory of the project."""
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_dotenv(path: Path | None = None) -> list[str]:
    """Load KEY=VALUE pairs into os.environ. Returns the names that were set.

    Understands `export FOO=bar`, `#` comments, blank lines, and quoted values.
    Silently ignores malformed lines rather than refusing to start.
    """
    path = path or find_dotenv()
    if path is None or not path.is_file():
        return []

    loaded: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()

        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]

        # A real env var beats the file — never clobber an explicit override.
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)

    return loaded
