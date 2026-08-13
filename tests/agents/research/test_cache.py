"""The rolling article window.

Measured on the live AI feeds: they hold about **four days**. A narrow topic
like "multimodal models" returns nothing — not because the news does not exist
but because it scrolled off before you asked. There is no "daily" setting to
widen; the only way to see a week is to remember what the feed said earlier in
the week.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from news_agent.agents.research import cache, tools
from news_agent.agents.research.models import Article, Category, HeadlineQuery


def _article(title: str, days_ago: int = 0, source: str = "F1") -> Article:
    when = (date.today() - timedelta(days=days_ago)).isoformat()
    return Article(
        title=title, source=source,
        url=f"https://x.test/{title.replace(' ', '-')}",
        published=when, summary="",
    )


@pytest.fixture
def store(tmp_path):
    return tmp_path / "articles.jsonl"


# --- the window --------------------------------------------------------------


def test_articles_survive_between_runs(store):
    cache.save([_article("monday story", days_ago=5)], path=store)
    assert [a.title for a in cache.load(path=store)] == ["monday story"]


def test_anything_outside_the_window_is_dropped(store):
    cache.save(
        [_article("recent", days_ago=2), _article("ancient", days_ago=30)],
        path=store,
    )
    assert [a.title for a in cache.load(path=store)] == ["recent"]


def test_the_window_is_configurable(store):
    cache.save([_article("six days ago", days_ago=6)], window_days=30, path=store)
    assert cache.load(window_days=3, path=store) == []
    assert len(cache.load(window_days=7, path=store)) == 1


def test_pruning_happens_on_write_so_the_file_cannot_grow_forever(store):
    cache.save([_article("old", days_ago=40)], window_days=90, path=store)
    kept = cache.save([_article("new")], window_days=7, path=store)
    assert kept == 1
    assert store.read_text("utf-8").count("\n") == 1


def test_duplicate_urls_collapse_with_the_fresh_copy_winning(store):
    cache.save([_article("original")], path=store)
    updated = _article("original")
    updated = updated.model_copy(update={"title": "corrected headline"})
    cache.save([updated], path=store)

    loaded = cache.load(path=store)
    assert len(loaded) == 1
    assert loaded[0].title == "corrected headline"


def test_results_come_back_newest_first(store):
    cache.save(
        [_article("old", days_ago=5), _article("new"), _article("mid", days_ago=2)],
        path=store,
    )
    assert [a.title for a in cache.load(path=store)] == ["new", "mid", "old"]


def test_sources_filter_keeps_categories_isolated(store):
    """Without this a cached World headline could answer an AI query, quietly
    breaking the isolation the feed map exists to provide."""
    cache.save([_article("ai story", source="Ars"), _article("world story", source="BBC")],
               path=store)
    got = cache.load(sources={"Ars"}, path=store)
    assert [a.title for a in got] == ["ai story"]


# --- it must never break a run -----------------------------------------------


def test_a_missing_cache_is_empty_not_an_error(tmp_path):
    assert cache.load(path=tmp_path / "nothing.jsonl") == []


def test_a_corrupt_line_loses_one_article_not_the_cache(store):
    cache.save([_article("good")], path=store)
    store.write_text(store.read_text("utf-8") + "{not json\n", encoding="utf-8")
    assert [a.title for a in cache.load(path=store)] == ["good"]


def test_an_unwritable_path_degrades_quietly(tmp_path):
    """A cache is an optimisation. An optimisation that can fail a run is a
    liability."""
    blocked = tmp_path / "file.txt"
    blocked.write_text("not a directory", encoding="utf-8")
    assert cache.save([_article("x")], path=blocked / "nested.jsonl") == 0


def test_an_unparseable_date_is_kept_rather_than_silently_dropped(store):
    """Feed dates are inconsistent. Dropping everything unparseable would
    shrink the window in a way nothing would report."""
    odd = Article(title="odd", source="F1", url="https://x.test/odd",
                  published="last Tuesday", summary="")
    cache.save([odd], path=store)
    assert len(cache.load(path=store)) == 1


# --- integration with the tool -----------------------------------------------


@pytest.fixture
def one_feed(monkeypatch):
    monkeypatch.setitem(tools.FEEDS, Category.AI, [("F1", "https://f1.test")])


def test_a_narrow_query_finds_what_the_feed_has_since_forgotten(monkeypatch, one_feed, store):
    """The whole point. Monday's story is gone from the feed by Thursday, so
    without the window the topic returns nothing at all."""
    monkeypatch.setenv("NEWS_AGENT_CACHE", str(store))
    monkeypatch.setattr(tools, "_fetch",
                        lambda s, u: ([_article("multimodal breakthrough", days_ago=4)], "ok"))
    tools.search_with_health(HeadlineQuery(category=Category.AI))

    # Days later the feed has moved on entirely.
    monkeypatch.setattr(tools, "_fetch",
                        lambda s, u: ([_article("something else today")], "ok"))
    result, _health, from_cache = tools.search_with_health(
        HeadlineQuery(category=Category.AI, keywords=["multimodal"])
    )

    assert result.article_count == 1
    assert result.articles[0].title == "multimodal breakthrough"
    assert from_cache >= 1


def test_todays_news_still_comes_first_for_a_broad_query(monkeypatch, one_feed, store):
    """Cached articles are appended, never interleaved — `limit` truncates from
    the end, so a broad query must not be pushed off by last week."""
    monkeypatch.setenv("NEWS_AGENT_CACHE", str(store))
    monkeypatch.setattr(tools, "_fetch",
                        lambda s, u: ([_article(f"old {i}", days_ago=5) for i in range(5)], "ok"))
    tools.search_with_health(HeadlineQuery(category=Category.AI))

    monkeypatch.setattr(tools, "_fetch",
                        lambda s, u: ([_article("today's lead")], "ok"))
    result, _health, _ = tools.search_with_health(
        HeadlineQuery(category=Category.AI, limit=1)
    )
    assert result.articles[0].title == "today's lead"


def test_the_note_says_how_far_back_it_looked(monkeypatch, one_feed, store):
    monkeypatch.setenv("NEWS_AGENT_CACHE", str(store))
    monkeypatch.setattr(tools, "_fetch", lambda s, u: ([_article("unrelated")], "ok"))
    result, _h, _c = tools.search_with_health(
        HeadlineQuery(category=Category.AI, keywords=["nonexistent"]), window_days=7
    )
    assert "last 7 days" in result.note
