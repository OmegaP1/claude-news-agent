"""The golden dataset: capture, agreement, and the decision rule.

These tests never call an API. That is the point of the whole module — replay
is the only part that spends money, and it is deliberately not exercised here.
"""

from __future__ import annotations

from datetime import date

import pytest

from news_agent.agents.judge.models import ItemVerdict, ScoredItem
from news_agent.agents.research.models import DigestItem, NewsDigest
from news_agent.evals.golden import (
    MIN_SAMPLE,
    GoldenFixture,
    acceptance_rate,
    agreement,
    capture,
    load_all,
    verdict_for,
)


def _digest(n: int) -> NewsDigest:
    return NewsDigest(
        topic="AI",
        overview="o",
        items=[
            DigestItem(headline=f"h{i}", summary="s", why_it_matters="w",
                       sources=[f"https://x.test/{i}"])
            for i in range(n)
        ],
        coverage_note="c",
    )


def _scored(digest: NewsDigest, index: int, composite: float = 4.0) -> ScoredItem:
    return ScoredItem(
        item=digest.items[index],
        verdict=ItemVerdict(item_index=index + 1, reasoning="r", significance=4,
                            novelty=4, relevance=4, evidence=4),
        composite=composite,
    )


def _fixture(judge: list[int], human: list[int], day: str = "2026-08-13") -> GoldenFixture:
    digest = _digest(5)
    return GoldenFixture(
        topic="AI", captured=day, digest=digest,
        judge_ranked=[0, 1, 2, 3, 4], judge_selected=judge, human_selected=human,
    )


# --- capture -----------------------------------------------------------------


def test_capture_records_both_selections(tmp_path):
    """`selected` alone cannot answer 'did the human disagree?' once the review
    hook has replaced it — so both are stored."""
    digest = _digest(4)
    ranked = [_scored(digest, i) for i in (0, 1, 2, 3)]
    path = capture(
        topic="AI", digest=digest, ranked=ranked,
        judge_selected=ranked[:3], human_selected=[ranked[0], ranked[1], ranked[3]],
        command="-3 +4", day=date(2026, 8, 13), directory=tmp_path,
    )
    saved = GoldenFixture.model_validate_json(path.read_text("utf-8"))
    assert saved.judge_selected == [0, 1, 2]
    assert saved.human_selected == [0, 1, 3]
    assert saved.overridden is True
    assert saved.command == "-3 +4"


def test_capture_stores_indices_not_copies(tmp_path):
    """A copy of the item could drift out of sync with the digest it came from,
    and nothing would say which one was right."""
    digest = _digest(3)
    ranked = [_scored(digest, i) for i in range(3)]
    path = capture(topic="AI", digest=digest, ranked=ranked,
                   judge_selected=ranked[:2], human_selected=ranked[:2],
                   day=date(2026, 8, 13), directory=tmp_path)
    raw = path.read_text("utf-8")
    assert '"judge_selected": [\n    0,\n    1\n  ]' in raw.replace("\r\n", "\n")


def test_round_trip_through_disk(tmp_path):
    digest = _digest(3)
    ranked = [_scored(digest, i) for i in range(3)]
    capture(topic="AI", digest=digest, ranked=ranked, judge_selected=ranked[:2],
            human_selected=ranked[:2], day=date(2026, 8, 13), directory=tmp_path)
    loaded = load_all(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].digest.items[0].headline == "h0"


def test_missing_directory_is_not_an_error(tmp_path):
    """Nothing captured yet is a normal state, not a failure."""
    assert load_all(tmp_path / "nope") == []


# --- agreement ---------------------------------------------------------------


def test_agreement_is_set_overlap_not_order():
    """Reordering within the chosen three changes nothing downstream, so a
    metric that punished it would measure something nobody acts on."""
    assert agreement([0, 1, 2], [2, 1, 0]) == 1.0


def test_partial_agreement():
    assert agreement([0, 1, 2], [0, 1, 3]) == 2 / 3


def test_total_disagreement():
    assert agreement([0, 1], [2, 3]) == 0.0


def test_empty_human_selection_means_nothing_should_have_been_filed():
    assert agreement([], []) == 1.0
    assert agreement([0], []) == 0.0


# --- the decision rule -------------------------------------------------------


def test_small_samples_refuse_to_conclude():
    """Three overrides out of four runs is not evidence the rubric is broken."""
    verdict = verdict_for([_fixture([0, 1, 2], [0, 1, 3]) for _ in range(4)])
    assert verdict.decision == "insufficient-data"
    assert str(MIN_SAMPLE) in verdict.detail


def test_high_agreement_says_drop_the_review_step():
    fixtures = [_fixture([0, 1, 2], [0, 1, 2]) for _ in range(MIN_SAMPLE)]
    verdict = verdict_for(fixtures)
    assert verdict.decision == "drop-review"
    assert verdict.rate == 1.0


def test_low_agreement_says_the_rubric_is_wrong():
    fixtures = [_fixture([0, 1, 2], [0, 1, 3]) for _ in range(MIN_SAMPLE)]
    verdict = verdict_for(fixtures)
    assert verdict.decision == "fix-rubric"
    assert "agents/judge/config.py" in verdict.detail


def test_middling_agreement_keeps_the_human_in_the_loop():
    agreed = [_fixture([0, 1, 2], [0, 1, 2]) for _ in range(15)]
    disagreed = [_fixture([0, 1, 2], [0, 1, 3]) for _ in range(5)]
    verdict = verdict_for(agreed + disagreed)
    assert verdict.decision == "keep-reviewing"
    assert verdict.rate == 0.75


def test_acceptance_rate_of_nothing_is_zero_not_a_crash():
    assert acceptance_rate([]) == 0.0


def test_replay_does_not_import_the_orchestrator():
    """The economic argument, enforced: replay re-runs the judge alone against
    a frozen digest. An orchestrator import would drag research and vault
    writes back in, and a one-cent replay would silently become a full run."""
    import inspect

    from news_agent.evals import golden

    source = inspect.getsource(golden)
    assert "orchestrator" not in source.replace(
        "An import of the orchestrator here", ""
    ).replace("orchestrator` or `sinks`", "")


# --- the regression gate -----------------------------------------------------


def test_replay_reports_both_before_and_after(monkeypatch, tmp_path):
    """The gate compares the judge *now* against the judge *as captured*, not
    against an absolute bar. A rubric is only better or worse than what it
    replaced."""
    from news_agent.agents.judge import JudgeResult
    from news_agent.core.types import TokenUsage
    from news_agent.evals import golden

    digest = _digest(4)
    fixture = GoldenFixture(
        topic="AI", captured="2026-08-13", digest=digest,
        judge_ranked=[0, 1, 2, 3], judge_selected=[0, 1, 2],
        human_selected=[0, 1, 3],          # the human swapped 2 for 3
    )

    # A "fixed" rubric that now picks what the human picked.
    ranked = [_scored(digest, i) for i in (0, 1, 3, 2)]
    monkeypatch.setattr(
        golden, "judge_digest",
        lambda *a, **k: JudgeResult(ranked=ranked, usage=TokenUsage(), model="m"),
        raising=False,
    )
    monkeypatch.setattr(
        "news_agent.agents.judge.judge_digest",
        lambda *a, **k: JudgeResult(ranked=ranked, usage=TokenUsage(), model="m"),
    )

    out = golden.replay(fixture)
    assert out["agreement_before"] == 2 / 3   # as captured, it disagreed
    assert out["agreement_with_human"] == 1.0  # now it matches


def test_summarise_names_the_direction():
    improved = [{"agreement_before": 0.5, "agreement_with_human": 0.9,
                 "captured": "2026-08-13", "topic": "AI"}]
    assert "improved" in golden_summarise(improved)

    regressed = [{"agreement_before": 0.9, "agreement_with_human": 0.5,
                  "captured": "2026-08-13", "topic": "AI"}]
    assert "regressed" in golden_summarise(regressed)


def test_summarise_with_nothing_says_how_to_get_started():
    assert "--capture-golden" in golden_summarise([])


def golden_summarise(results):
    from news_agent.evals.golden import summarise

    return summarise(results)


# --- two ways to disagree, two different fixes -------------------------------


def test_adding_a_dropped_item_is_not_a_ranking_failure():
    """Measured over the first 16 real fixtures: 4 of 5 overrides were of this
    kind, and every one landed on exactly top_n. The judge ranked correctly;
    the floor cut too much."""
    f = _fixture([0, 1], [0, 1, 2])
    assert f.overridden is True      # it IS a change
    assert f.only_added is True      # but the judge's picks all survived
    assert f.swapped is False


def test_removing_a_pick_is_a_ranking_failure():
    f = _fixture([0, 1, 2], [0, 1, 3])
    assert f.swapped is True
    assert f.only_added is False


def test_the_two_rates_separate_the_two_problems():
    """The defect this fixes: a judge ranking well behind a too-strict floor
    read as a low acceptance rate, and the verdict sent you to rewrite a
    rubric that was working."""
    fixtures = (
        [_fixture([0, 1, 2], [0, 1, 2]) for _ in range(11)]   # agreed
        + [_fixture([0, 1], [0, 1, 2]) for _ in range(4)]      # floor too strict
        + [_fixture([0, 1, 2], [3, 4, 5])]                     # genuine swap
    )
    from news_agent.evals.golden import floor_rate, ranking_rate

    assert acceptance_rate(fixtures) == pytest.approx(11 / 16)   # 69%
    assert ranking_rate(fixtures) == pytest.approx(15 / 16)      # 94%
    assert floor_rate(fixtures) == pytest.approx(4 / 16)


def test_a_good_ranking_behind_a_strict_floor_says_lower_the_floor():
    fixtures = (
        [_fixture([0, 1, 2], [0, 1, 2]) for _ in range(14)]
        + [_fixture([0, 1], [0, 1, 2]) for _ in range(6)]
    )
    verdict = verdict_for(fixtures)
    assert verdict.decision == "lower-the-floor"
    assert "MIN_COMPOSITE" in verdict.detail
    assert "rubric is fine" in verdict.detail


def test_real_ranking_failures_still_point_at_the_rubric():
    fixtures = [_fixture([0, 1, 2], [3, 4, 5]) for _ in range(MIN_SAMPLE)]
    verdict = verdict_for(fixtures)
    assert verdict.decision == "fix-rubric"
    assert "agents/judge/config.py" in verdict.detail


def test_perfect_agreement_still_says_drop_review():
    fixtures = [_fixture([0, 1, 2], [0, 1, 2]) for _ in range(MIN_SAMPLE)]
    assert verdict_for(fixtures).decision == "drop-review"
