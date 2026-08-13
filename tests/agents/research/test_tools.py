"""Tool tests. All offline — feed HTTP is monkeypatched."""

from __future__ import annotations

import pytest

from news_agent.agents.research import tools
from news_agent.agents.research.models import Article, Category, HeadlineQuery

RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>Chipmaker announces new AI accelerator</title>
    <link>https://example.com/a</link>
    <pubDate>Mon, 11 Aug 2026 09:00:00 GMT</pubDate>
    <description>&lt;p&gt;The company said   the chip is   faster.&lt;/p&gt;</description>
  </item>
  <item>
    <title>Local bakery wins award</title>
    <link>https://example.com/b</link>
    <pubDate>Mon, 11 Aug 2026 08:00:00 GMT</pubDate>
    <description>Sourdough.</description>
  </item>
</channel></rss>"""

ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Regulators open AI inquiry</title>
    <link href="https://example.com/c"/>
    <published>2026-08-11T10:00:00Z</published>
    <summary>An inquiry was opened.</summary>
  </entry>
</feed>"""


def test_parses_rss_and_strips_html():
    articles = tools._parse_feed("Test", RSS)
    assert [a.title for a in articles] == [
        "Chipmaker announces new AI accelerator",
        "Local bakery wins award",
    ]
    # HTML tags stripped and whitespace collapsed.
    assert articles[0].summary == "The company said the chip is faster."
    assert articles[0].url == "https://example.com/a"


def test_parses_atom_link_href():
    """Atom puts the URL in an attribute, not the element body."""
    articles = tools._parse_feed("Test", ATOM)
    assert len(articles) == 1
    assert articles[0].url == "https://example.com/c"
    assert articles[0].published == "2026-08-11"  # normalised from ISO-8601


def test_malformed_xml_returns_empty_not_raises():
    assert tools._parse_feed("Test", "<rss><broken>") == []


def test_articles_without_title_or_url_are_dropped():
    feed = "<rss><channel><item><title>No link</title></item></channel></rss>"
    assert tools._parse_feed("Test", feed) == []


def test_clean_truncates_long_text():
    out = tools._clean("x" * 900, limit=50)
    assert len(out) == 50
    assert out.endswith("…")


def _fetched(articles):
    """Shape a stubbed fetch like the real one: (articles, health verdict).

    The stubs mirror `_fetch`'s contract deliberately — a stub that is easier
    than the real thing is a stub that stops catching the real thing's bugs.
    """
    return list(articles), ("ok" if articles else "empty")


@pytest.fixture
def stub_feeds(monkeypatch):
    """Serve canned XML per feed URL instead of hitting the network."""

    def fake_fetch(source: str, url: str):
        body = ATOM if "second" in url else RSS
        return _fetched(tools._parse_feed(source, body))

    monkeypatch.setattr(tools, "_fetch", fake_fetch)
    monkeypatch.setitem(
        tools.FEEDS,
        Category.TECHNOLOGY,
        [("First", "https://first.example/rss"), ("Second", "https://second.example/rss")],
    )


def test_keyword_filter_is_case_insensitive_and_any_match(stub_feeds):
    result = tools.search_headlines(
        HeadlineQuery(category=Category.TECHNOLOGY, keywords=["ai"])
    )
    titles = [a.title for a in result.articles]
    assert "Chipmaker announces new AI accelerator" in titles  # matches "AI"
    assert "Regulators open AI inquiry" in titles
    assert "Local bakery wins award" not in titles


def test_no_keywords_returns_everything(stub_feeds):
    result = tools.search_headlines(HeadlineQuery(category=Category.TECHNOLOGY))
    assert result.article_count == 3
    assert result.note == ""


def test_limit_is_respected(stub_feeds):
    result = tools.search_headlines(
        HeadlineQuery(category=Category.TECHNOLOGY, limit=1)
    )
    assert result.article_count == 1
    assert len(result.articles) == 1


def test_duplicate_urls_across_feeds_are_deduped(monkeypatch):
    same = tools._parse_feed("Dup", RSS)
    monkeypatch.setattr(tools, "_fetch", lambda s, u: _fetched(same))
    monkeypatch.setitem(
        tools.FEEDS, Category.WORLD, [("A", "https://a.test"), ("B", "https://b.test")]
    )
    result = tools.search_headlines(HeadlineQuery(category=Category.WORLD))
    assert result.article_count == 2  # not 4


def test_note_explains_zero_keyword_matches(stub_feeds):
    """The model needs to know the difference between 'feeds down' and
    'nothing matched' so it can decide whether to retry differently."""
    result = tools.search_headlines(
        HeadlineQuery(category=Category.TECHNOLOGY, keywords=["zzzznope"])
    )
    assert result.article_count == 0
    assert "none matched" in result.note


def test_note_explains_dead_feeds(monkeypatch):
    monkeypatch.setattr(tools, "_fetch", lambda s, u: ([], "failed"))
    result = tools.search_headlines(HeadlineQuery(category=Category.SCIENCE))
    assert result.article_count == 0
    assert "failed to load" in result.note


def test_network_error_degrades_to_empty(monkeypatch):
    """One dead feed must not take down the whole query."""

    def boom(request, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(tools.urllib.request, "urlopen", boom)
    # The reason is returned rather than discarded: a network failure and a
    # feed that simply had nothing are both "no articles" to the caller, but
    # only one of them is worth waking someone up about.
    assert tools._fetch("Dead", "https://dead.example/rss") == ([], "failed")


def test_limit_bounds_enforced_by_pydantic():
    with pytest.raises(ValueError):
        HeadlineQuery(category=Category.TOP, limit=99)


def test_unknown_category_rejected():
    with pytest.raises(ValueError):
        HeadlineQuery(category="sports")


def test_article_model_roundtrips():
    a = Article(title="t", source="s", url="u", published="p", summary="x")
    assert Article.model_validate_json(a.model_dump_json()) == a


# --- token-efficiency of the payload ----------------------------------------


def test_tracking_query_strings_are_stripped():
    """Measured: URLs are 33% of a tool result and the `?at_medium=RSS...`
    tail is ~11% of the whole payload — re-sent on every later turn."""
    assert (
        tools._clean_url(
            "https://www.bbc.co.uk/news/articles/abc?at_medium=RSS&at_campaign=rss"
        )
        == "https://www.bbc.co.uk/news/articles/abc"
    )
    assert tools._clean_url("https://e.test/a#frag") == "https://e.test/a"
    assert tools._clean_url("https://e.test/a") == "https://e.test/a"
    assert tools._clean_url("") == ""


def test_rfc822_dates_are_normalised_to_iso():
    assert tools._clean_date("Mon, 11 Aug 2026 09:00:00 GMT") == "2026-08-11"
    assert tools._clean_date("Tue, 12 Aug 2026 09:00:00 +0000") == "2026-08-12"


def test_iso_dates_are_truncated_to_the_day():
    assert tools._clean_date("2026-08-11T10:00:00Z") == "2026-08-11"


def test_unknown_date_format_is_bounded_not_guessed():
    """Never invent a date — but never let a malformed field blow up the
    payload either."""
    out = tools._clean_date("sometime last tuesday, probably, who knows honestly")
    assert len(out) <= 16


def test_empty_date_stays_empty():
    assert tools._clean_date("") == ""


def test_result_does_not_echo_the_query_back(stub_feeds):
    """The model just sent these arguments; echoing them costs tokens on every
    subsequent turn for zero information."""
    result = tools.search_headlines(
        HeadlineQuery(category=Category.TECHNOLOGY, keywords=["ai"])
    )
    dumped = result.model_dump()
    assert "keywords_used" not in dumped
    assert "feeds_queried" not in dumped
    assert dumped["category"] == "technology"  # kept: grounds which feed set ran


def test_empty_feed_list_does_not_crash(monkeypatch):
    """ThreadPoolExecutor(max_workers=0) raises ValueError."""
    monkeypatch.setitem(tools.FEEDS, Category.SCIENCE, [])
    result = tools.search_headlines(HeadlineQuery(category=Category.SCIENCE))
    assert result.article_count == 0
    assert "No feeds are configured" in result.note


# --- AI category & fair feed representation ---------------------------------


def test_ai_category_exists_and_has_dedicated_feeds():
    """General news feeds carry too little AI to filter usefully — that showed
    up as honest-but-thin digests on every AI topic."""
    assert Category.AI.value == "ai"
    assert len(tools.FEEDS[Category.AI]) >= 4


def test_feeds_are_interleaved_not_concatenated(monkeypatch):
    """Regression: `limit` truncates the result, so concatenating gave the
    first feed's whole front page and the later feeds nothing. Invisible with
    two feeds, severe with five."""

    def fake_fetch(source, url):
        return [
            Article(title=f"{source}-{n}", source=source,
                    url=f"https://{source}.test/{n}", published="2026-08-13", summary="x")
            for n in range(1, 6)
        ]

    monkeypatch.setattr(tools, "_fetch", lambda s, u: _fetched(fake_fetch(s, u)))
    monkeypatch.setitem(
        tools.FEEDS, Category.AI,
        [(f"F{i}", f"https://f{i}.test") for i in range(1, 6)],
    )
    result = tools.search_headlines(HeadlineQuery(category=Category.AI, limit=5))

    # One from each feed, not five from the first.
    assert [a.source for a in result.articles] == ["F1", "F2", "F3", "F4", "F5"]


def test_interleaving_survives_uneven_feed_lengths(monkeypatch):
    """A short feed must not truncate the others — zip_longest, not zip."""

    def fake_fetch(source, url):
        count = 1 if source == "Short" else 4
        return [
            Article(title=f"{source}-{n}", source=source,
                    url=f"https://{source}.test/{n}", published="2026-08-13", summary="x")
            for n in range(count)
        ]

    monkeypatch.setattr(tools, "_fetch", lambda s, u: _fetched(fake_fetch(s, u)))
    monkeypatch.setitem(
        tools.FEEDS, Category.AI,
        [("Short", "https://a.test"), ("Long", "https://b.test")],
    )
    result = tools.search_headlines(HeadlineQuery(category=Category.AI, limit=10))
    assert result.article_count == 5           # 1 + 4, nothing dropped
    assert sum(a.source == "Long" for a in result.articles) == 4


def test_a_dead_feed_does_not_shift_the_others(monkeypatch):
    def fake_fetch(source, url):
        if source == "Dead":
            return []
        return [Article(title=f"{source}-1", source=source,
                        url=f"https://{source}.test/1", published="2026-08-13", summary="x")]

    monkeypatch.setattr(tools, "_fetch", lambda s, u: _fetched(fake_fetch(s, u)))
    monkeypatch.setitem(
        tools.FEEDS, Category.AI,
        [("Dead", "https://x.test"), ("Alive", "https://y.test")],
    )
    result = tools.search_headlines(HeadlineQuery(category=Category.AI))
    assert [a.source for a in result.articles] == ["Alive"]
