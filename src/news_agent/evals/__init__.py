"""Offline evaluation: replay the judge against human-labelled ground truth.

This layer may import `core`, `agents.research` and `agents.judge` — and
deliberately **not** `orchestrator` or `sinks`. That restriction is the whole
economic argument: replay re-runs the *judge only*, against a frozen digest, so
iterating on the rubric costs about a cent instead of re-paying for research
and re-writing the vault. An import of the orchestrator here would quietly
reintroduce both.
"""

from .golden import (
    GoldenFixture,
    Verdict,
    acceptance_rate,
    agreement,
    capture,
    captured_topics,
    load_all,
    verdict_for,
)

__all__ = [
    "GoldenFixture",
    "Verdict",
    "acceptance_rate",
    "agreement",
    "capture",
    "captured_topics",
    "load_all",
    "verdict_for",
]
