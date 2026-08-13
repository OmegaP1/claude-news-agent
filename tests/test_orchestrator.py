"""Pipeline tests: research → judge → select → persist, all stubbed."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from news_agent.agents.research.agent import DigestResult
from news_agent.agents.judge import JudgeResult, composite
from news_agent.agents.judge.models import ItemVerdict, ScoredItem
from news_agent.agents.research.models import DigestItem, NewsDigest
from news_agent.core.types import TokenUsage
from news_agent.orchestrator import combined_usage, run_pipeline

DAY = date(2026, 8, 13)


def _scored(headline: str, score: int) -> ScoredItem:
    v = ItemVerdict(
        item_index=1, reasoning=f"r-{headline}",
        significance=score, novelty=score, relevance=score, evidence=score,
    )
    return ScoredItem(
        item=DigestItem(
            headline=headline, summary="s", why_it_matters="w",
            sources=["https://e.test/a"],
        ),
        verdict=v,
        composite=composite(v),
    )


@pytest.fixture
def fake_vault(tmp_path):
    (tmp_path / ".obsidian").mkdir()
    return tmp_path


@pytest.fixture
def stub_stages(monkeypatch):
    """Replace both model calls; returns a knob for the ranked items."""
    state = {"ranked": [_scored(h, s) for h, s in [("A", 5), ("B", 4), ("C", 3), ("D", 1)]]}

    digest = NewsDigest(
        topic="ai regulation", overview="Overview.",
        items=[s.item for s in state["ranked"]], coverage_note="c",
    )
    research = DigestResult(
        digest=digest,
        usage=TokenUsage(input_tokens=10_000, output_tokens=1_000, turns=4),
        model="claude-haiku-4-5",
    )
    monkeypatch.setattr("news_agent.orchestrator.run_digest", lambda *a, **k: research)
    monkeypatch.setattr(
        "news_agent.orchestrator.judge_digest",
        lambda *a, **k: JudgeResult(
            ranked=state["ranked"],
            usage=TokenUsage(input_tokens=1_200, output_tokens=500, turns=1),
            model="claude-sonnet-5",
        ),
    )
    return state


def test_only_the_top_three_are_selected(stub_stages, fake_vault):
    result = run_pipeline("ai regulation", client=object(), vault=fake_vault, day=DAY)
    assert [s.item.headline for s in result.selected] == ["A", "B", "C"]
    assert len(result.vault_result.item_notes) == 3


def test_the_fourth_item_never_reaches_the_vault(stub_stages, fake_vault):
    run_pipeline("ai regulation", client=object(), vault=fake_vault, day=DAY)
    names = [p.name for p in (fake_vault / "News" / "Items").glob("*.md")]
    assert not any("D" in n for n in names)


def test_quality_floor_can_shrink_the_selection(stub_stages, fake_vault):
    """'Top 3' is a ceiling, not a quota — three weak stories are worse than
    one good one, which is the same logic as the digest's thin-coverage note."""
    stub_stages["ranked"] = [_scored("A", 5), _scored("B", 1), _scored("C", 1)]
    result = run_pipeline("t", client=object(), vault=fake_vault, day=DAY)
    assert [s.item.headline for s in result.selected] == ["A"]
    assert [s.item.headline for s in result.below_floor] == ["B", "C"]


def test_nothing_written_when_everything_is_below_the_floor(stub_stages, fake_vault):
    stub_stages["ranked"] = [_scored("A", 1)]
    result = run_pipeline("t", client=object(), vault=fake_vault, day=DAY)
    assert result.selected == []
    assert result.vault_result is None
    assert not (fake_vault / "News").exists()


def test_dry_run_judges_but_writes_nothing(stub_stages, fake_vault):
    result = run_pipeline(
        "t", client=object(), vault=fake_vault, day=DAY, dry_run=True
    )
    assert len(result.selected) == 3          # still ranked and selected
    assert result.vault_result is None
    assert not (fake_vault / "News").exists()


def test_top_n_is_configurable(stub_stages, fake_vault):
    result = run_pipeline("t", client=object(), vault=fake_vault, day=DAY, top_n=1)
    assert len(result.selected) == 1


# --- cost across two differently-priced models -------------------------------


def test_cost_sums_both_models_at_their_own_rates(stub_stages, fake_vault):
    """A single token total would be wrong: Haiku is $1/$5, Sonnet is $3/$15."""
    result = run_pipeline("t", client=object(), vault=fake_vault, day=DAY)
    haiku = 10_000 * 1 / 1e6 + 1_000 * 5 / 1e6
    sonnet = 1_200 * 3 / 1e6 + 500 * 15 / 1e6
    assert result.total_cost_usd == pytest.approx(haiku + sonnet)


def test_cost_line_names_both_stages(stub_stages, fake_vault):
    result = run_pipeline("t", client=object(), vault=fake_vault, day=DAY)
    assert "claude-haiku-4-5" in result.cost_line
    assert "claude-sonnet-5" in result.cost_line


def test_combined_usage_adds_up(stub_stages, fake_vault):
    result = run_pipeline("t", client=object(), vault=fake_vault, day=DAY)
    usage = combined_usage(result)
    assert usage.input_tokens == 11_200
    assert usage.output_tokens == 1_500
    assert usage.turns == 5


def test_combined_usage_covers_every_bucket():
    """Regression: adding a TokenUsage field is easy to forget here, and the
    omission is invisible — the total just silently under-reports."""
    import dataclasses

    from news_agent.core.types import TokenUsage

    a = TokenUsage(
        input_tokens=1, output_tokens=2, cache_read_tokens=3,
        cache_creation_tokens=4, turns=1,
    )
    result = SimpleNamespace(
        research=SimpleNamespace(usage=a), judged=SimpleNamespace(usage=a)
    )
    merged = combined_usage(result)

    # Every numeric field must be summed, not just the ones that existed when
    # this function was written.
    for name in TokenUsage.model_fields:
        assert getattr(merged, name) == getattr(a, name) * 2, f"{name} not summed"
    assert merged.total_tokens == a.total_tokens * 2


# --- the spend ceiling -------------------------------------------------------


def test_the_ceiling_aborts_before_the_judge_runs(stub_stages, fake_vault, monkeypatch):
    """Placement is the whole point. Research is already paid for by the time
    the check runs; the judge is on a pricier model and has not started. A
    check after the judge would be a report, not a guard."""
    from news_agent.core.budget import BudgetExceeded

    judged = {"ran": False}

    def spy(*a, **k):
        judged["ran"] = True
        raise AssertionError("the judge must not run once the ceiling is hit")

    monkeypatch.setattr("news_agent.orchestrator.judge_digest", spy)

    # Stubbed research: 10k in + 1k out on Haiku = $0.015.
    with pytest.raises(BudgetExceeded) as exc:
        run_pipeline("ai regulation", client=object(), vault=fake_vault, day=DAY,
                     max_usd=0.001)

    assert judged["ran"] is False
    assert exc.value.stage == "research"


def test_a_generous_ceiling_changes_nothing(stub_stages, fake_vault):
    result = run_pipeline("ai regulation", client=object(), vault=fake_vault,
                          day=DAY, max_usd=10.0)
    assert [s.item.headline for s in result.selected] == ["A", "B", "C"]


def test_no_ceiling_is_the_default(stub_stages, fake_vault):
    result = run_pipeline("ai regulation", client=object(), vault=fake_vault, day=DAY)
    assert len(result.selected) == 3


def test_a_budget_abort_is_a_guardrail_block_not_an_exception(stub_stages, fake_vault, monkeypatch):
    """`guardrail_block` was defined but never used. A ceiling that stopped a
    run on purpose is exactly what it means — filing it as
    `application_exception` would put a deliberate stop in the same bucket as
    a crash."""
    from news_agent.core import telemetry as tel

    scores = []

    class FakeClient:
        def score_current_trace(self, **kw):
            scores.append(kw)

        def __getattr__(self, _):
            return lambda **kw: None

    monkeypatch.setattr(tel, "_client", FakeClient())
    monkeypatch.setattr(tel, "enabled", True)

    with pytest.raises(Exception):
        run_pipeline("ai regulation", client=object(), vault=fake_vault, day=DAY,
                     max_usd=0.001)

    outcome = next(s for s in scores if s["name"] == "outcome")
    assert outcome["value"] == "guardrail_block"


def test_judge_selection_is_preserved_for_capture(stub_stages, fake_vault):
    """A golden fixture needs what the judge wanted *and* what the human took;
    `selected` alone cannot answer 'did they disagree?' after the hook runs."""
    def hook(ranked, selected):
        return [ranked[0], ranked[3]], "human"

    result = run_pipeline("ai regulation", client=object(), vault=fake_vault,
                          day=DAY, select_hook=hook)

    assert [s.item.headline for s in result.judge_selected] == ["A", "B", "C"]
    assert [s.item.headline for s in result.selected] == ["A", "D"]
