"""The custom tool: fetch and filter live news headlines from public RSS feeds.

Deliberately stdlib-only. RSS is simple enough that ``urllib`` +
``ElementTree`` is ~60 lines, and skipping ``feedparser`` keeps the dependency
tree to just ``anthropic`` + ``pydantic``.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from itertools import zip_longest
from xml.etree import ElementTree

from . import cache
from .cache import DEFAULT_WINDOW_DAYS
from .models import Article, Category, HeadlineQuery, HeadlineSearchResult

FEEDS: dict[Category, list[tuple[str, str]]] = {
    # Dedicated AI publications. General news feeds carry too little AI to
    # filter usefully — the digests come back honest but thin. These are all
    # AI-only, so a bare category query is already on-topic before keywords.
    Category.AI: [
        ("MIT Tech Review", "https://www.technologyreview.com/topic/artificial-intelligence/feed/"),
        ("Ars Technica", "https://arstechnica.com/ai/feed/"),
        ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/"),
        ("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
        # points=100 filters to stories the HN community actually upvoted,
        # which is a far better signal than the raw firehose.
        ("Hacker News", "https://hnrss.org/newest?q=AI&points=100"),
    ],
    Category.TOP: [
        ("BBC", "https://feeds.bbci.co.uk/news/rss.xml"),
        ("NPR", "https://feeds.npr.org/1001/rss.xml"),
    ],
    Category.WORLD: [
        ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
        ("NPR World", "https://feeds.npr.org/1004/rss.xml"),
    ],
    Category.BUSINESS: [
        ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
        ("NPR Business", "https://feeds.npr.org/1006/rss.xml"),
    ],
    Category.TECHNOLOGY: [
        ("BBC Tech", "https://feeds.bbci.co.uk/news/technology/rss.xml"),
        ("Hacker News", "https://hnrss.org/frontpage"),
    ],
    Category.SCIENCE: [
        ("BBC Science", "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"),
        ("NPR Science", "https://feeds.npr.org/1007/rss.xml"),
    ],
}

_ATOM = "{http://www.w3.org/2005/Atom}"
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_TIMEOUT = 10.0
_USER_AGENT = "news-agent/0.1 (+https://github.com/example/news-agent)"


def _clean(text: str | None, limit: int = 400) -> str:
    """Strip HTML tags and collapse whitespace from a feed field."""
    if not text:
        return ""
    flat = _WS_RE.sub(" ", _TAG_RE.sub(" ", text)).strip()
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _clean_url(url: str) -> str:
    """Drop tracking query strings.

    Measured: URLs are the single largest part of a tool result (33%), and
    feed URLs carry `?at_medium=RSS&at_campaign=rss`-style junk that is ~11%
    of the whole payload — re-sent on every subsequent turn, since each turn
    resends the full conversation. The article still resolves without it.
    """
    return url.split("?", 1)[0].split("#", 1)[0] if url else url


def _clean_date(raw: str) -> str:
    """Normalise a feed date to ISO `YYYY-MM-DD`.

    RFC-822 (`Mon, 11 Aug 2026 09:00:00 GMT`) tokenises badly — 147 tokens
    across 8 articles. The model only needs recency and ordering, not the
    second the story published.
    """
    if not raw:
        return ""
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # ISO-8601 (Atom) — the date is already the leading 10 characters.
    if len(raw) >= 10 and raw[4] == "-" and raw[7] == "-":
        return raw[:10]
    return raw[:16]  # unknown format: keep it bounded rather than guessing


def _text(node, *names: str) -> str:
    """First non-empty child matching any of ``names`` (RSS or Atom form)."""
    for name in names:
        for tag in (name, f"{_ATOM}{name}"):
            found = node.find(tag)
            if found is not None:
                # Atom <link> carries the URL in an attribute, not the body.
                value = found.text or found.get("href") or ""
                if value.strip():
                    return value.strip()
    return ""


def _parse_feed(source: str, xml: str) -> list[Article]:
    """Parse an RSS 2.0 or Atom document into Articles."""
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError:
        return []

    entries = root.iter("item")
    articles = [
        Article(
            title=_clean(_text(item, "title"), 300),
            source=source,
            url=_clean_url(_text(item, "link")),
            published=_clean_date(_text(item, "pubDate", "published", "updated")),
            summary=_clean(_text(item, "description", "summary", "content")),
        )
        for item in entries
    ]
    if not articles:  # Atom feeds use <entry> rather than <item>.
        articles = [
            Article(
                title=_clean(_text(entry, "title"), 300),
                source=source,
                url=_clean_url(_text(entry, "link")),
                published=_clean_date(_text(entry, "published", "updated")),
                summary=_clean(_text(entry, "summary", "content")),
            )
            for entry in root.iter(f"{_ATOM}entry")
        ]
    return [a for a in articles if a.title and a.url]


@dataclass(frozen=True)
class FeedHealth:
    """Which feeds worked, per query.

    Kept **out** of `HeadlineSearchResult` on purpose. That model is serialised
    into the conversation and re-sent on every subsequent turn, so anything
    added to it is paid for repeatedly — and the model has no use for feed
    diagnostics. This goes to telemetry only.
    """

    #: Fetched and parsed to at least one article.
    ok: tuple[str, ...] = ()
    #: Network error, timeout, or unparseable XML.
    failed: tuple[str, ...] = ()
    #: Reachable and well-formed, but zero articles — the quieter failure.
    #: A feed that silently empties looks identical to a slow news day until
    #: you notice it has been empty for a week.
    empty: tuple[str, ...] = ()

    @property
    def all_ok(self) -> bool:
        return not self.failed and not self.empty


def _fetch(source: str, url: str) -> tuple[list[Article], str]:
    """Fetch one feed, returning its articles and a health verdict.

    Failures still degrade to an empty list and never raise — one dead feed
    must not take down a multi-feed query. But the *reason* is now returned
    instead of discarded: previously a dead feed was indistinguishable from a
    quiet one, because the results were merged before anyone counted them.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            body = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return [], "failed"
    articles = _parse_feed(source, body)
    # Unparseable XML also lands here as zero articles; both mean "this feed
    # gave us nothing", which is the fact worth alerting on.
    return articles, "ok" if articles else "empty"


def _matches(article: Article, keywords: list[str]) -> bool:
    if not keywords:
        return True
    haystack = f"{article.title} {article.summary}".lower()
    return any(k.lower() in haystack for k in keywords)


def search_headlines(query: HeadlineQuery) -> HeadlineSearchResult:
    """Core tool logic, decoupled from the Anthropic SDK so it stays testable."""
    return search_with_health(query)[0]


def search_with_health(
    query: HeadlineQuery, *, window_days: int = DEFAULT_WINDOW_DAYS
) -> tuple[HeadlineSearchResult, FeedHealth, int]:
    """As `search_headlines`, but also reports feed health and cache usage.

    Separate returns rather than one enriched model, because the audiences are
    different: the articles go to Claude and cost tokens on every turn, the
    diagnostics go to Langfuse and cost nothing.
    """
    feeds = FEEDS.get(query.category, [])
    if not feeds:
        return (
            HeadlineSearchResult(
                category=query.category.value,
                article_count=0,
                articles=[],
                note="No feeds are configured for this category.",
            ),
            FeedHealth(),
            0,
        )

    with ThreadPoolExecutor(max_workers=len(feeds)) as pool:
        outcomes = list(pool.map(lambda f: _fetch(*f), feeds))

    fetched = [articles for articles, _ in outcomes]
    by_status: dict[str, list[str]] = {"ok": [], "failed": [], "empty": []}
    for (name, _url), (_articles, status) in zip(feeds, outcomes):
        by_status[status].append(name)
    health = FeedHealth(
        ok=tuple(by_status["ok"]),
        failed=tuple(by_status["failed"]),
        empty=tuple(by_status["empty"]),
    )

    # Interleave round-robin rather than concatenating. `limit` truncates the
    # result, so concatenation gives the first feed's whole front page and the
    # later feeds nothing — invisible with two feeds, severe with five.
    articles: list[Article] = []
    seen: set[str] = set()
    for row in zip_longest(*fetched):
        for article in row:
            if article is not None and article.url not in seen:
                seen.add(article.url)
                articles.append(article)

    # Everything fresh goes into the rolling window before anything is read
    # back out, so today's fetch is available to next week's narrow query.
    cache.save(articles, window_days=window_days)

    # Cached articles are appended, never interleaved: a broad query still
    # gets today's news first because fresh entries come earlier in the list
    # and `limit` truncates from the end. The window only shows through when
    # the keyword filter would otherwise return nothing — which is exactly the
    # case it exists for.
    from_cache = 0
    for article in cache.load(
        window_days=window_days, sources={name for name, _ in feeds}
    ):
        if article.url not in seen:
            seen.add(article.url)
            articles.append(article)
            from_cache += 1

    total_fetched = len(articles)
    matched = [a for a in articles if _matches(a, query.keywords)]

    note = ""
    if total_fetched == 0:
        note = "All feeds for this category failed to load or returned nothing."
    elif not matched and query.keywords:
        note = (
            f"Searched {total_fetched} articles from the last {window_days} days "
            f"but none matched {query.keywords}. Retry with broader keywords or "
            f"none at all."
        )

    return (
        HeadlineSearchResult(
            category=query.category.value,
            article_count=len(matched[: query.limit]),
            articles=matched[: query.limit],
            note=note,
        ),
        health,
        from_cache,
    )
