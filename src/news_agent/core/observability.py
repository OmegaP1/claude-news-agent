"""Optional Langfuse tracing.

Langfuse is a soft dependency by design: if the package is missing, the keys
are unset, or tracing is explicitly disabled, ``observe`` becomes an identity
decorator and the reporting helpers become no-ops — so the agent runs
identically with zero configuration. Import this module instead of importing
``langfuse`` directly anywhere else.

Import order matters: the decision below happens at import time, and
``@observe`` is applied at def-time on ``run_digest``. ``.env`` is therefore
loaded in ``news_agent/__init__.py`` before this module is reached.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# Set by tests/conftest.py. A real env var always beats .env in our loader, so
# this reliably wins even on a machine with Langfuse keys configured.
DISABLE_FLAG = "NEWS_AGENT_DISABLE_TRACING"


def _identity(func: F | None = None, **_kwargs: Any) -> Any:
    """No-op stand-in for langfuse.observe.

    Must support both `@observe` and `@observe(name="...")`, so a bare call
    with no function returns the decorator itself.
    """
    if func is None:
        return _identity
    return func


def _configured() -> bool:
    if os.getenv(DISABLE_FLAG) == "1":
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


_client: Any = None
enabled = False

if _configured():
    try:
        from langfuse import get_client, observe as _langfuse_observe

        _client = get_client()
        observe = _langfuse_observe
        enabled = True
    except Exception:  # noqa: BLE001 - tracing must never break the agent
        observe = _identity
else:
    observe = _identity


def report_generation(**fields: Any) -> None:
    """Attach model / usage / cost to the current observation.

    Accepts langfuse's `update_current_generation` kwargs (model,
    usage_details, cost_details, input, output, metadata). Silently does
    nothing when tracing is off.
    """
    if _client is None:
        return
    try:
        _client.update_current_generation(**fields)
    except Exception:  # noqa: BLE001
        pass


def trace_url() -> str | None:
    """Deep link to the trace just produced, so the CLI can print it."""
    if _client is None:
        return None
    try:
        return _client.get_trace_url()
    except Exception:  # noqa: BLE001
        return None


def flush() -> None:
    """Push buffered spans before the process exits.

    Langfuse batches in a background thread; a short-lived CLI can exit before
    the flush happens, silently losing the trace.
    """
    if _client is not None:
        try:
            _client.flush()
        except Exception:  # noqa: BLE001
            pass


def status() -> str:
    if enabled:
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        return f"Langfuse tracing enabled → {host}"
    if os.getenv(DISABLE_FLAG) == "1":
        return f"Langfuse tracing disabled via {DISABLE_FLAG}."
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        return "Langfuse keys set but the SDK failed to load — tracing disabled."
    return "Langfuse tracing disabled (set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY to enable)."
