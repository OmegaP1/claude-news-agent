"""Judge tests. No API calls — the client is a stub."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from news_agent.agents.judge import (
    MIN_COMPOSITE,
    WEIGHTS,
    JudgeError,
    composite,
    judge_digest,
)
from news_agent.agents.judge.models import ItemVerdict, JudgeVerdicts
from news_agent.agents.research.models import DigestItem, NewsDigest


def item(headline: str) -> DigestItem:
    return DigestItem(
        headline=headline, summary="s", why_it_matters="w", sources=["https://e.test/a"]
    )


def digest(*headlines: str) -> NewsDigest:
    return NewsDigest(
        topic="ai regulation",
        overview="o",
        items=[item(h) for h in headlines],
        coverage_note="c",
    )


def verdict(index: int, sig=3, nov=3, rel=3, ev=3) -> dict:
    return {
        "item_index": index,
        "reasoning": f"reason {index}",
        "significance": sig,
        "novelty": nov,
        "relevance": rel,
        "evidence": ev,
    }


def make_client(verdicts: list[dict], *, inp=1000, out=400):
    captured: dict = {}

    def create(**kwargs):
        captured.update(kwargs)
        payload = JudgeVerdicts.model_validate({"verdicts": verdicts}).model_dump_json()
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=payload)],
            usage=SimpleNamespace(
                input_tokens=inp, output_tokens=out, cache_read_input_tokens=0
            ),
        )

    return SimpleNamespace(messages=SimpleNamespace(create=create)), captured


# --- rubric mechanics --------------------------------------------------------


def test_weights_sum_to_one():
    """Otherwise the composite is not on the same 1-5 scale as the floor."""
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_composite_is_a_weighted_mean_computed_in_code():
    v = ItemVerdict(
        item_index=1, reasoning="r", significance=5, novelty=1, relevance=4, evidence=2
    )
    expected = 5 * 0.40 + 4 * 0.30 + 1 * 0.20 + 2 * 0.10
    assert composite(v) == pytest.approx(expected)


def test_all_fives_hits_the_top_of_the_scale():
    v = ItemVerdict(
        item_index=1, reasoning="r", significance=5, novelty=5, relevance=5, evidence=5
    )
    assert composite(v) == pytest.approx(5.0)


def test_scores_are_enum_bound_not_just_documented():
    """Structured outputs ignore ge/le but do enforce enum — Literal is the
    only thing that actually binds the 1-5 scale."""
    schema = JudgeVerdicts.model_json_schema()["$defs"]["ItemVerdict"]
    assert schema["properties"]["significance"]["enum"] == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        ItemVerdict(
            item_index=1, reasoning="r", significance=9, novelty=3, relevance=3, evidence=3
        )


def test_reasoning_is_generated_before_the_scores():
    """Field order is the mechanism that stops post-hoc rationalisation."""
    order = list(JudgeVerdicts.model_json_schema()["$defs"]["ItemVerdict"]["properties"])
    assert order.index("reasoning") < order.index("significance")


# --- judging -----------------------------------------------------------------


def test_ranks_best_first():
    client, _ = make_client([verdict(1, 1, 1, 1, 1), verdict(2, 5, 5, 5, 5)])
    result = judge_digest(digest("weak", "strong"), client=client)
    assert [s.item.headline for s in result.ranked] == ["strong", "weak"]


def test_uses_a_different_model_than_the_generator():
    """Self-preference bias: a model grading its own output is not a check."""
    client, captured = make_client([verdict(1)])
    result = judge_digest(digest("a"), client=client)
    assert captured["model"] == "claude-sonnet-5"
    assert result.model == "claude-sonnet-5"


def test_never_sends_temperature():
    """Sonnet 5 rejects sampling parameters with a 400 — the usual
    'temperature=0 for judges' advice would break the call outright."""
    client, captured = make_client([verdict(1)])
    judge_digest(digest("a"), client=client)
    assert "temperature" not in captured
    assert "top_p" not in captured
    assert "top_k" not in captured


def test_sends_no_tools():
    """Judging is pure evaluation; a tool loop here would only add cost."""
    client, captured = make_client([verdict(1)])
    judge_digest(digest("a"), client=client)
    assert "tools" not in captured


def test_ties_keep_original_order():
    """Determinism: identical scores must not reshuffle between runs."""
    client, _ = make_client([verdict(1), verdict(2), verdict(3)])
    result = judge_digest(digest("first", "second", "third"), client=client)
    assert [s.item.headline for s in result.ranked] == ["first", "second", "third"]


def test_top_applies_the_quality_floor():
    client, _ = make_client([verdict(1, 5, 5, 5, 5), verdict(2, 1, 1, 1, 1)])
    result = judge_digest(digest("strong", "weak"), client=client)
    assert len(result.ranked) == 2
    assert [s.item.headline for s in result.top(3)] == ["strong"]  # weak is below 2.5


def test_floor_is_configurable():
    client, _ = make_client([verdict(1, 1, 1, 1, 1)])
    result = judge_digest(digest("weak"), client=client)
    assert result.top(3, floor=1.0) != []
    assert result.top(3, floor=MIN_COMPOSITE) == []


def test_empty_digest_costs_nothing():
    """No items means no reason to call the API at all."""
    called = False

    def create(**kwargs):
        nonlocal called
        called = True

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    result = judge_digest(digest(), client=client)
    assert result.ranked == []
    assert result.usage.input_tokens == 0
    assert called is False


# --- untrusted judge output --------------------------------------------------


def test_out_of_range_index_is_dropped_not_fatal():
    """An index the judge invented must not throw away a paid run."""
    client, _ = make_client([verdict(1), verdict(99)])
    result = judge_digest(digest("a"), client=client)
    assert len(result.ranked) == 1
    assert result.unmatched_verdicts == [99]


def test_duplicate_index_keeps_the_first():
    client, _ = make_client([verdict(1, 5, 5, 5, 5), verdict(1, 1, 1, 1, 1)])
    result = judge_digest(digest("a"), client=client)
    assert len(result.ranked) == 1
    assert result.ranked[0].composite == pytest.approx(5.0)
    assert result.unmatched_verdicts == [1]


def test_skipped_item_is_reported():
    client, _ = make_client([verdict(1)])
    result = judge_digest(digest("a", "b"), client=client)
    assert result.unjudged_items == [2]
    assert len(result.ranked) == 1


def test_unparseable_output_raises_judge_error():
    def create(**kwargs):
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="not json at all")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, cache_read_input_tokens=0),
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    with pytest.raises(JudgeError, match="unparseable"):
        judge_digest(digest("a"), client=client)


def test_missing_text_block_raises_judge_error():
    def create(**kwargs):
        return SimpleNamespace(
            content=[],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, cache_read_input_tokens=0),
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    with pytest.raises(JudgeError, match="no text block"):
        judge_digest(digest("a"), client=client)


def test_cost_is_reported_to_langfuse(monkeypatch):
    sent = {}
    monkeypatch.setattr("news_agent.agents.judge.agent.report_generation", lambda **kw: sent.update(kw))
    client, _ = make_client([verdict(1)], inp=1200, out=500)
    judge_digest(digest("a"), client=client)
    assert sent["model"] == "claude-sonnet-5"
    assert sent["usage_details"]["input"] == 1200
    # Sonnet 5: $3/Mtok in, $15/Mtok out
    assert sent["cost_details"]["total"] == pytest.approx(1200 * 3 / 1e6 + 500 * 15 / 1e6)
    assert sent["metadata"]["weights"] == WEIGHTS
