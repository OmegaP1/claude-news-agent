"""Levelled telemetry: levels 1, 2 and 3.

Telemetry is the one subsystem where a silent failure is the *expected* mode —
every adapter swallows exceptions so a broken sink can never fail a run. That
makes it exactly the subsystem most in need of tests, because nothing else
will ever tell you it stopped working.
"""

from __future__ import annotations

import pytest

from news_agent.core import telemetry as tel
from news_agent.core.telemetry import (
    EvalType,
    Outcome,
    args_hash,
    emit_eval,
    emit_run_outcome,
    tool_call,
)


@pytest.fixture
def sink(monkeypatch):
    """Capture what would have gone to Langfuse."""
    captured = {"scores": [], "spans": []}

    class FakeClient:
        def score_current_trace(self, **kw):
            captured["scores"].append({"scope": "trace", **kw})

        def score_current_span(self, **kw):
            captured["scores"].append({"scope": "span", **kw})

        def update_current_span(self, **kw):
            captured["spans"].append(kw)

    monkeypatch.setattr(tel, "_client", FakeClient())
    monkeypatch.setattr(tel, "enabled", True)
    return captured


# --- no-op contract ----------------------------------------------------------


def test_everything_is_a_noop_without_a_client(monkeypatch):
    """Same contract as @observe: no keys, no behaviour change."""
    monkeypatch.setattr(tel, "_client", None)
    monkeypatch.setattr(tel, "enabled", False)

    emit_run_outcome(Outcome.SUCCESS)
    emit_eval(eval_type=EvalType.GROUNDEDNESS, eval_name="g", passed=True)
    with tool_call("t"):
        pass
    with tel.agent_run("run-1"):
        pass  # must not raise


def test_a_broken_sink_never_breaks_the_run(monkeypatch):
    class Exploding:
        def __getattr__(self, _):
            def boom(**kw):
                raise RuntimeError("sink is down")
            return boom

    monkeypatch.setattr(tel, "_client", Exploding())
    monkeypatch.setattr(tel, "enabled", True)

    emit_run_outcome(Outcome.SUCCESS)
    emit_eval(eval_type=EvalType.CUSTOM, eval_name="q", score=1.0)
    with tool_call("t"):
        pass  # must not raise


# --- Level 1 -----------------------------------------------------------------


def test_run_outcome_records_the_enum_and_rollups(sink):
    emit_run_outcome(Outcome.SUCCESS, items_filed=3, selected_by="human")
    meta = sink["spans"][0]["metadata"]
    assert meta["pipeline_level"] == 1
    assert meta["items_filed"] == 3
    assert meta["selected_by"] == "human"


def test_outcome_is_also_a_categorical_score(sink):
    """Metadata can only be read one trace at a time; a score can be filtered
    and charted across every run."""
    emit_run_outcome(Outcome.GOAL_UNMET)
    score = next(s for s in sink["scores"] if s["name"] == "outcome")
    assert score["value"] == "goal_unmet"
    assert score["data_type"] == "CATEGORICAL"


# --- Level 2 -----------------------------------------------------------------


def test_tool_call_records_status_and_latency(sink):
    with tool_call("search_headlines") as detail:
        detail["articles_returned"] = 8
    meta = sink["spans"][0]["metadata"]
    assert meta["pipeline_level"] == 2
    assert meta["tool_name"] == "search_headlines"
    assert meta["tool_status"] == "success"
    assert meta["articles_returned"] == 8


def test_side_effect_flag_distinguishes_reads_from_writes(sink):
    with tool_call("search_headlines", side_effect=False):
        pass
    with tool_call("write_digest", side_effect=True):
        pass
    flags = {s["metadata"]["tool_name"]: s["metadata"]["side_effect"] for s in sink["spans"]}
    assert flags == {"search_headlines": False, "write_digest": True}


def test_failure_is_recorded_and_re_raised(sink):
    """A swallowed tool failure that still logs 'success' is worse than no
    telemetry at all."""
    with pytest.raises(ValueError):
        with tool_call("search_headlines"):
            raise ValueError("feed exploded")

    meta = sink["spans"][0]["metadata"]
    assert meta["tool_status"] == "error"
    assert meta["error_type"] == "ValueError"


def test_timeout_is_classified_separately(sink):
    with pytest.raises(TimeoutError):
        with tool_call("search_headlines"):
            raise TimeoutError
    assert sink["spans"][0]["metadata"]["tool_status"] == "timeout"


def test_args_are_hashed_never_emitted_raw(sink):
    with tool_call("search_headlines", args={"keywords": ["secret-term"]}):
        pass
    meta = sink["spans"][0]["metadata"]
    assert "secret-term" not in str(meta)
    assert len(meta["args_hash"]) == 16


def test_args_hash_is_order_independent():
    assert args_hash({"a": 1, "b": 2}) == args_hash({"b": 2, "a": 1})
    assert args_hash({"a": 1}) != args_hash({"a": 2})


# --- Level 3 -----------------------------------------------------------------


def test_eval_emits_passed_as_boolean_and_score_as_numeric(sink):
    emit_eval(
        eval_type=EvalType.GROUNDEDNESS, eval_name="grounding",
        passed=False, score=0.75, evaluator="url-set-check",
    )
    by_name = {s["name"]: s for s in sink["scores"]}
    assert by_name["grounding_passed"]["data_type"] == "BOOLEAN"
    assert by_name["grounding_passed"]["value"] is False
    assert by_name["grounding"]["data_type"] == "NUMERIC"
    assert by_name["grounding"]["value"] == 0.75
    assert by_name["grounding"]["metadata"]["evaluator"] == "url-set-check"


def test_passed_and_score_are_independent(sink):
    """A check can pass with a middling score; they are distinct fields."""
    emit_eval(eval_type=EvalType.CUSTOM, eval_name="q", score=0.5)
    assert [s["name"] for s in sink["scores"]] == ["q"]


def test_evals_land_on_the_trace_so_they_aggregate(sink):
    emit_eval(eval_type=EvalType.HUMAN_FEEDBACK, eval_name="judge_accepted", passed=True)
    assert sink["scores"][0]["scope"] == "trace"


# --- the three evals this pipeline actually produces --------------------------


def test_groundedness_eval_from_the_research_agent(sink, monkeypatch):
    from news_agent.agents.research.agent import _check_grounding
    from news_agent.agents.research.models import DigestItem, NewsDigest

    digest = NewsDigest(
        topic="t", overview="o",
        items=[DigestItem(headline="h", summary="s", why_it_matters="w",
                          sources=["https://real.test/a", "https://fake.test/b"])],
        coverage_note="c",
    )
    ungrounded = _check_grounding(digest, {"https://real.test/a"})

    assert ungrounded == ["https://fake.test/b"]
    by_name = {s["name"]: s for s in sink["scores"]}
    assert by_name["grounding_passed"]["value"] is False
    assert by_name["grounding"]["value"] == 0.5      # 1 of 2 cited urls real
    assert by_name["grounding"]["metadata"]["eval_type"] == "groundedness"


def test_judge_quality_evals_are_averaged_per_dimension(sink):
    from news_agent.agents.judge.agent import _emit_quality_evals
    from news_agent.agents.judge.models import ItemVerdict, ScoredItem
    from news_agent.agents.research.models import DigestItem

    def item(score):
        v = ItemVerdict(item_index=1, reasoning="r", significance=score,
                        novelty=score, relevance=score, evidence=score)
        return ScoredItem(
            item=DigestItem(headline="h", summary="s", why_it_matters="w", sources=[]),
            verdict=v, composite=float(score),
        )

    _emit_quality_evals([item(5), item(3)], "claude-sonnet-5")

    by_name = {s["name"]: s for s in sink["scores"]}
    assert by_name["judge_significance"]["value"] == 4.0      # mean of 5 and 3
    assert by_name["judge_composite"]["value"] == 4.0
    assert by_name["judge_relevance"]["metadata"]["evaluator"] == "llm-as-judge:claude-sonnet-5"


def test_no_judge_evals_when_nothing_was_ranked(sink):
    from news_agent.agents.judge.agent import _emit_quality_evals

    _emit_quality_evals([], "claude-sonnet-5")
    assert sink["scores"] == []


# --- levels are sequential ---------------------------------------------------


def test_levels_are_1_2_3_with_no_gaps(sink):
    """Gaps were an artefact of a wider standard we no longer follow; a level
    number that skips reads as a missing implementation."""
    emit_run_outcome(Outcome.SUCCESS)
    with tool_call("t"):
        pass
    emit_eval(eval_type=EvalType.CUSTOM, eval_name="q", score=1.0)

    assert {s["metadata"]["pipeline_level"] for s in sink["spans"]} == {1, 2}
    # Evals carry no level: it would be a constant 3 on every one of them.
    assert all(
        "pipeline_level" not in (s.get("metadata") or {}) for s in sink["scores"]
    )


# Langfuse owns these on every observation. A metadata key of the same name is
# not an error — it is worse: the UI filter reads the built-in field, finds
# nothing, and shows an empty result that looks like "this never happened".
LANGFUSE_RESERVED = {
    "level",          # observation severity: DEBUG/DEFAULT/WARNING/ERROR
    "status",         # the Status filter in the trace sidebar
    "status_message",
    "name",
    "input",
    "output",
    "model",
    "usage_details",
    "cost_details",
    "session_id",
    "user_id",
    "tags",
    "version",
    "release",
}


def test_no_metadata_key_collides_with_a_langfuse_builtin(sink):
    """Regression: `level` shadowed Langfuse's own observation level, so
    filtering on it in the UI silently matched nothing."""
    emit_run_outcome(Outcome.SUCCESS, items_filed=1, selected_by="judge")
    with tool_call("t", side_effect=True, args={"a": 1}):
        pass
    emit_eval(
        eval_type=EvalType.GROUNDEDNESS, eval_name="g",
        passed=True, score=1.0, evaluator="url-set-check",
    )

    emitted = {k for s in sink["spans"] for k in s["metadata"]}
    emitted |= {k for s in sink["scores"] for k in (s.get("metadata") or {})}
    assert not (emitted & LANGFUSE_RESERVED), (
        f"shadows a Langfuse builtin: {sorted(emitted & LANGFUSE_RESERVED)}"
    )


# --- the failure path --------------------------------------------------------


def test_agent_run_does_not_corrupt_exceptions_from_the_body():
    """Regression: wrapping the yield in try/except made the context manager
    yield twice, turning a clean domain error into
    'RuntimeError: generator didn't stop after throw()'. The telemetry safety
    net was corrupting the very errors it was meant not to touch."""
    with pytest.raises(ValueError, match="real failure"):
        with tel.agent_run("run-1"):
            raise ValueError("real failure")


def test_agent_run_survives_a_broken_propagate(monkeypatch):
    """Setup may fail silently — correlation is lost, the run continues."""
    monkeypatch.setattr(tel, "enabled", True)
    import langfuse

    def boom(**kw):
        raise RuntimeError("propagation is down")

    monkeypatch.setattr(langfuse, "propagate_attributes", boom)
    with tel.agent_run("run-1"):
        pass  # must not raise


# --- no field duplicates what the platform already stores ---------------------


def test_level_1_carries_only_what_langfuse_cannot_derive(sink):
    """Every dropped field was a second copy of something Langfuse already has:
    run_id (session_id), duration_ms (span timings), total_cost_usd (aggregated
    from cost_details), generation_count (the span tree)."""
    emit_run_outcome(Outcome.SUCCESS, items_judged=4, items_filed=3, selected_by="judge")
    meta = sink["spans"][0]["metadata"]
    assert set(meta) == {
        "pipeline_level", "items_judged", "items_filed", "selected_by",
    }


def test_tool_call_does_not_restate_its_own_duration(sink):
    """The enclosing span carries start and end; latency_ms was a less
    accurate copy of it."""
    with tool_call("t"):
        pass
    assert "latency_ms" not in sink["spans"][0]["metadata"]


def test_outcome_is_not_duplicated_into_metadata(sink):
    emit_run_outcome(Outcome.SUCCESS)
    assert "outcome" not in sink["spans"][0]["metadata"]
    assert any(s["name"] == "outcome" for s in sink["scores"])


def test_vault_write_gets_its_own_span_not_the_pipeline_root():
    """Regression: with no span of its own, the vault's Level 2 metadata was
    written onto the pipeline span and then overwritten by Level 1 — the only
    side-effecting operation in the system vanished from the trace."""
    import inspect

    from news_agent.sinks import obsidian

    src = inspect.getsource(obsidian)
    decorated = src.index('@observe(name="write_digest"')
    tool = src.index('tool_call(\n        "write_digest"')
    assert decorated < tool, "@observe must wrap tool_call, not sit inside it"
