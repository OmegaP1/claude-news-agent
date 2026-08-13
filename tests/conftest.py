"""Test-suite-wide setup.

MUST run before any test module imports news_agent. pytest imports conftest
first, which is exactly the window we need: `news_agent/__init__.py` loads
.env, and `observability` then decides at import time whether to initialise
Langfuse. Setting the flag here means a developer with real Langfuse keys in
.env does not spray ~20 junk traces into their dashboard on every `pytest`.

(The tests never spend money — the Anthropic client is always a stub — but
they were emitting real traces, which is its own kind of mess.)
"""

from __future__ import annotations

import os

import pytest

os.environ["NEWS_AGENT_DISABLE_TRACING"] = "1"


@pytest.fixture(autouse=True)
def isolate_article_cache(tmp_path, monkeypatch):
    """Point the rolling article cache at a throwaway file, always.

    Without this the suite reads and writes the developer's real cache: tests
    would leak articles into it, and — worse — start passing or failing based
    on what yesterday's run happened to leave behind. A test that depends on
    machine state is not a test.
    """
    monkeypatch.setenv("NEWS_AGENT_CACHE", str(tmp_path / "articles.jsonl"))
