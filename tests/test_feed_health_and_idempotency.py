"""Two silent failures: a dead feed, and a vault note erased by a re-run.

Both were invisible by construction. A feed that dies still leaves four others
returning articles, and an overwrite of a *different* note looks exactly like
the intended idempotent case. Neither would ever have produced an error.
"""

from __future__ import annotations

from datetime import date

import pytest

from news_agent.agents.judge.models import ItemVerdict, ScoredItem
from news_agent.agents.research import tools
from news_agent.agents.research.models import (
    Article,
    Category,
    DigestItem,
    HeadlineQuery,
)
from news_agent.sinks.obsidian import write_digest


# --- feed health -------------------------------------------------------------


def _article(source: str, n: int) -> Article:
    return Article(title=f"{source}-{n}", source=source,
                   url=f"https://{source}.test/{n}", published="2026-08-13", summary="x")


@pytest.fixture
def five_feeds(monkeypatch):
    monkeypatch.setitem(
        tools.FEEDS, Category.AI,
        [(f"F{i}", f"https://f{i}.test") for i in range(1, 6)],
    )


def test_a_dead_feed_is_named_while_the_query_still_succeeds(monkeypatch, five_feeds):
    """The failure this catches: one dead feed out of five still returns plenty
    of articles, so `article_count` looks healthy and nothing says otherwise."""

    def fetch(source, url):
        if source == "F3":
            return [], "failed"
        return [_article(source, 1)], "ok"

    monkeypatch.setattr(tools, "_fetch", fetch)
    result, health, _ = tools.search_with_health(HeadlineQuery(category=Category.AI))

    assert result.article_count == 4       # looks fine
    assert health.failed == ("F3",)        # but it is not
    assert health.all_ok is False


def test_an_empty_feed_is_distinguished_from_a_broken_one(monkeypatch, five_feeds):
    """Reachable-but-empty is the quieter failure: a feed that silently stops
    publishing looks like a slow news day until it has been empty for a week."""

    def fetch(source, url):
        return ([], "empty") if source == "F2" else ([_article(source, 1)], "ok")

    monkeypatch.setattr(tools, "_fetch", fetch)
    _result, health, _ = tools.search_with_health(HeadlineQuery(category=Category.AI))

    assert health.empty == ("F2",)
    assert health.failed == ()


def test_health_never_reaches_the_model(monkeypatch, five_feeds):
    """It would be re-sent on every subsequent turn — each turn resends the
    whole conversation — and the model has no use for feed diagnostics."""

    monkeypatch.setattr(tools, "_fetch", lambda s, u: ([], "failed"))
    result, health, _ = tools.search_with_health(HeadlineQuery(category=Category.AI))

    payload = result.model_dump_json()
    assert health.failed  # something to leak
    for name in health.failed:
        assert name not in payload
    assert "feeds_failed" not in payload


def test_the_tool_span_carries_the_failed_feed_names(monkeypatch, five_feeds):
    """Where the information does go: Level 2 metadata, which costs nothing."""
    from news_agent.agents.research import agent as agent_mod
    from news_agent.core import telemetry as tel

    captured = []

    class FakeClient:
        def update_current_span(self, **kw):
            captured.append(kw["metadata"])

        def __getattr__(self, _):
            return lambda **kw: None

    monkeypatch.setattr(tel, "_client", FakeClient())
    monkeypatch.setattr(tel, "enabled", True)
    monkeypatch.setattr(tools, "_fetch", lambda s, u: ([], "failed"))

    agent_mod.search_headlines.call({"category": "ai"})

    meta = captured[0]
    assert meta["feeds_failed"] == [f"F{i}" for i in range(1, 6)]
    assert meta["feeds_ok"] == 0


def test_healthy_runs_do_not_emit_empty_failure_lists(monkeypatch, five_feeds):
    """A field that is always `[]` is noise on every trace."""
    from news_agent.agents.research import agent as agent_mod
    from news_agent.core import telemetry as tel

    captured = []

    class FakeClient:
        def update_current_span(self, **kw):
            captured.append(kw["metadata"])

        def __getattr__(self, _):
            return lambda **kw: None

    monkeypatch.setattr(tel, "_client", FakeClient())
    monkeypatch.setattr(tel, "enabled", True)
    monkeypatch.setattr(tools, "_fetch", lambda s, u: ([_article(s, 1)], "ok"))

    agent_mod.search_headlines.call({"category": "ai"})

    assert "feeds_failed" not in captured[0]
    assert "feeds_empty" not in captured[0]
    assert captured[0]["feeds_ok"] == 5


# --- vault idempotency -------------------------------------------------------


def _item(headline: str, composite: float = 4.0) -> ScoredItem:
    return ScoredItem(
        item=DigestItem(headline=headline, summary="s", why_it_matters="w",
                        sources=["https://x.test/1"]),
        verdict=ItemVerdict(item_index=1, reasoning="r", significance=4,
                            novelty=4, relevance=4, evidence=4),
        composite=composite,
    )


DAY = date(2026, 8, 13)


def test_identical_rerun_is_silent(tmp_path):
    """This is the idempotent case the design wants. Warning about it would
    train you to ignore the warning."""
    items = [_item("Same story")]
    first = write_digest("AI", "o", items, vault=tmp_path, day=DAY)
    second = write_digest("AI", "o", items, vault=tmp_path, day=DAY)

    assert first.conflicts == []
    assert second.conflicts == []
    assert len(second.item_notes) == 1


def test_a_differing_rerun_refuses_to_erase_the_first(tmp_path):
    """The data loss this prevents: the judge ranks differently on a second
    run, and 'converge' quietly means yesterday's note is gone."""
    write_digest("AI", "o", [_item("Story", 4.0)], vault=tmp_path, day=DAY)
    before = (tmp_path / "News" / "Items" / "2026-08-13 Story.md").read_text("utf-8")

    result = write_digest("AI", "o", [_item("Story", 2.0)], vault=tmp_path, day=DAY)

    assert result.conflicts == ["2026-08-13 Story"]
    assert result.item_notes == []
    after = (tmp_path / "News" / "Items" / "2026-08-13 Story.md").read_text("utf-8")
    assert after == before, "the original note was modified despite the conflict"


def test_force_overwrites_deliberately(tmp_path):
    write_digest("AI", "o", [_item("Story", 4.0)], vault=tmp_path, day=DAY)
    result = write_digest(
        "AI", "o", [_item("Story", 2.0)], vault=tmp_path, day=DAY, force=True
    )

    assert result.conflicts == []
    assert len(result.item_notes) == 1
    body = (tmp_path / "News" / "Items" / "2026-08-13 Story.md").read_text("utf-8")
    assert "composite: 2.0" in body


def test_the_index_is_not_written_when_items_conflicted(tmp_path):
    """A partial index links to notes that were skipped, which is worse than
    an index one run out of date."""
    write_digest("AI", "o", [_item("Story", 4.0)], vault=tmp_path, day=DAY)
    index = tmp_path / "News" / "Digests" / "2026-08-13 AI.md"
    before = index.read_text("utf-8")

    write_digest("AI", "changed overview", [_item("Story", 2.0)],
                 vault=tmp_path, day=DAY)

    assert index.read_text("utf-8") == before


def test_notes_written_counts_what_was_actually_written(tmp_path):
    """Regression: the telemetry reported `len(items)`, so a run that wrote
    nothing still logged three notes written."""
    from news_agent.core import telemetry as tel

    captured = []

    class FakeClient:
        def update_current_span(self, **kw):
            captured.append(kw["metadata"])

        def __getattr__(self, _):
            return lambda **kw: None

    write_digest("AI", "o", [_item("Story", 4.0)], vault=tmp_path, day=DAY)

    import pytest as _pytest
    monkey = _pytest.MonkeyPatch()
    try:
        monkey.setattr(tel, "_client", FakeClient())
        monkey.setattr(tel, "enabled", True)
        write_digest("AI", "o", [_item("Story", 2.0)], vault=tmp_path, day=DAY)
    finally:
        monkey.undo()

    assert captured[0]["notes_written"] == 0
    assert captured[0]["conflicts"] == 1
