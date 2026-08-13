"""Levelled telemetry for this pipeline.

See ``docs/observability.md``.

The level is emitted as ``pipeline_level``, **not** ``level``: Langfuse already
owns ``level`` on every observation (``DEBUG``/``DEFAULT``/``WARNING``/
``ERROR``) and the trace UI shows tree depth as ``L0``/``L1``/``L2`` as well.
Three meanings of one word on a single screen is a filter that silently returns
nothing. ``status`` is namespaced to ``tool_status`` for the same reason — the
UI has its own Status filter.

Three levels, numbered sequentially for this project:

- **Level 1 ``agent_run``** — one pipeline invocation end to end.
- **Level 2 ``tool_call``** — every tool, with status, latency and whether it
  mutated external state.
- **Level 3 ``eval``** — guardrails and quality checks as *scores*, not
  metadata, because scores are queryable and chartable and metadata is not.

Per-generation events are omitted (Langfuse already emits a generation span
per model call), as are retrieval events (RSS filtering is not RAG) and
return-on-automation KPIs (no business baseline exists here).

Everything here is a no-op when tracing is off, so the pipeline behaves
identically with no Langfuse keys — same contract as ``observability.observe``.

This module lives in ``core`` and therefore may not import any agent. It takes
primitives (strings, numbers, bools) rather than domain objects; the callers do
the mapping.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack, contextmanager
from enum import Enum
from typing import Any, Iterator

from .observability import _client, enabled


class Outcome(str, Enum):
    """How a run ended. Every terminal path maps to exactly one of these."""

    SUCCESS = "success"
    GOAL_UNMET = "goal_unmet"
    GUARDRAIL_BLOCK = "guardrail_block"
    TOOL_FAILURE = "tool_failure"
    APPLICATION_EXCEPTION = "application_exception"
    CANCELLED = "cancelled"


class EvalType(str, Enum):
    """Level 3 eval types actually produced here."""

    GROUNDEDNESS = "groundedness"
    HUMAN_FEEDBACK = "human_feedback"
    CUSTOM = "custom"


class ToolStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


def args_hash(value: Any) -> str:
    """SHA-256 of canonical arguments.

    Raw args and results are never emitted. Hashing keeps the
    ability to spot "same call repeated N times" without putting payloads —
    or anything derived from them — into a telemetry store.
    """
    canonical = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# --- Level 1 -----------------------------------------------------------------


@contextmanager
def agent_run(run_id: str, *, tags: list[str] | None = None, **metadata: Any) -> Iterator[None]:
    """Bind a `run_id` to every span emitted inside the block.

    Without this the research trace, the judge trace and the vault write are
    three unrelated observations. `propagate_attributes` sets it once and it
    reaches every child.

    Uses ExitStack rather than a try/except around the ``yield``. Wrapping the
    yield means an exception raised *by the body* is caught by the telemetry
    handler, which then yields a second time — illegal in a generator-based
    context manager, and it converts a clean domain error into
    ``RuntimeError: generator didn't stop after throw()``. Only *setup* may
    fail silently here; the body's exceptions must pass through untouched.
    """
    with ExitStack() as stack:
        if enabled:
            try:
                from langfuse import propagate_attributes

                stack.enter_context(
                    propagate_attributes(
                        session_id=run_id, tags=tags or [], metadata=metadata or None
                    )
                )
            except Exception:  # noqa: BLE001 - correlation lost, run continues
                pass
        yield


def emit_run_outcome(outcome: Outcome, **fields: Any) -> None:
    """Close Level 1: the outcome enum plus the run's roll-up counters.

    Uses `update_current_span`, not a trace-level update: the SDK has no
    `update_current_trace` (verified against the installed package — calling
    it would have failed silently inside the exception guard). Called from
    inside the root span, this lands on the trace root anyway.
    """
    _update_span(metadata={"pipeline_level": 1, **fields})
    # Outcome lives only as a score: a score is filterable and chartable across
    # runs, a metadata copy is readable one trace at a time. Emitting both was
    # two places to keep in sync for no extra answer.
    _score(name="outcome", value=outcome.value, data_type="CATEGORICAL")


# --- Level 2 -----------------------------------------------------------------


@contextmanager
def tool_call(
    tool_name: str, *, side_effect: bool = False, args: Any = None
) -> Iterator[dict]:
    """Emit Level 2 for a tool call.

    Yields a dict the caller may annotate (e.g. ``result["count"] = n``).
    Records `error` status and re-raises on exception — a swallowed tool
    failure that still logs `success` is worse than no telemetry at all.

    Timing is deliberately *not* recorded here: the enclosing span already
    carries start and end, so a `latency_ms` field would be a second, less
    accurate copy of it. Each tool therefore needs its own span — see the
    `@observe` decorators on the callers.
    """
    detail: dict[str, Any] = {}
    status = ToolStatus.SUCCESS
    error_type: str | None = None
    try:
        yield detail
    except TimeoutError as exc:
        status, error_type = ToolStatus.TIMEOUT, type(exc).__name__
        raise
    except Exception as exc:  # noqa: BLE001 - classify, then re-raise
        status, error_type = ToolStatus.ERROR, type(exc).__name__
        raise
    finally:
        payload = {
            "pipeline_level": 2,
            "tool_name": tool_name,
            "tool_status": status.value,
            "side_effect": side_effect,
            **detail,
        }
        if args is not None:
            payload["args_hash"] = args_hash(args)
        if error_type is not None:
            payload["error_type"] = error_type
        _update_span(metadata=payload)


# --- Level 3 -----------------------------------------------------------------


def emit_eval(
    *,
    eval_type: EvalType,
    eval_name: str,
    passed: bool | None = None,
    score: float | None = None,
    evaluator: str | None = None,
    comment: str | None = None,
    trace_level: bool = True,
    **metadata: Any,
) -> None:
    """Emit a Level 3 eval as a Langfuse *score*.

    Scores, not metadata: a score can be charted over time, filtered on, and
    aggregated across runs. Metadata can only be read one trace at a time,
    which defeats the point of measuring quality.

    `passed` and `score` are separate on purpose — a check can pass with a
    middling score, so collapsing them would lose information.
    """
    # No `pipeline_level` here: every eval is level 3, and a field with one
    # possible value teaches you nothing — same reason there is no provider_id.
    common = {"evaluator": evaluator, "eval_type": eval_type.value, **metadata}
    if passed is not None:
        _score(
            name=f"{eval_name}_passed",
            value=passed,
            data_type="BOOLEAN",
            comment=comment,
            metadata=common,
            trace_level=trace_level,
        )
    if score is not None:
        _score(
            name=eval_name,
            value=float(score),
            data_type="NUMERIC",
            comment=comment,
            metadata=common,
            trace_level=trace_level,
        )


# --- thin Langfuse adapters --------------------------------------------------
#
# Every one of these swallows exceptions: telemetry is never allowed to be the
# reason a run fails.


def _score(
    *, name: str, value: Any, data_type: str, comment: str | None = None,
    metadata: dict | None = None, trace_level: bool = True,
) -> None:
    if _client is None:
        return
    try:
        target = _client.score_current_trace if trace_level else _client.score_current_span
        target(
            name=name, value=value, data_type=data_type,
            comment=comment, metadata=metadata,
        )
    except Exception:  # noqa: BLE001
        pass


def _update_trace(**fields: Any) -> None:
    if _client is None:
        return
    try:
        _client.update_current_trace(**fields)
    except Exception:  # noqa: BLE001
        pass


def _update_span(**fields: Any) -> None:
    if _client is None:
        return
    try:
        _client.update_current_span(**fields)
    except Exception:  # noqa: BLE001
        pass
