"""A rolling window of articles seen across runs.

The problem: an RSS feed holds whatever it holds. Measured on the AI feeds,
that is about **four days** — so a narrow query like "multimodal models" finds
nothing, not because there is no such news but because it scrolled off the feed
before you asked.

There is no "daily" setting to widen; the window is a property of the feed. The
only way to see a week is to remember what the feed said earlier in the week.

Each run merges the live fetch with everything cached inside the window,
de-duplicated by URL. Fresh articles keep their position at the front, so a
broad query still gets today's news first; the cache only surfaces when the
keyword filter would otherwise return nothing.

**This starts empty.** It cannot retroactively recover last week — the benefit
accrues from the next run onward.

Every operation degrades to a no-op on error. A cache is an optimisation, and
an optimisation that can fail a run is a liability.
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

from .models import Article

#: A week. Long enough to catch a topic that surfaced on Monday, short enough
#: that "current news" still means current.
DEFAULT_WINDOW_DAYS = 7


def cache_path() -> Path:
    """Where the rolling window lives. Overridable with NEWS_AGENT_CACHE."""
    override = os.getenv("NEWS_AGENT_CACHE")
    if override:
        return Path(override)
    base = os.getenv("LOCALAPPDATA") or os.path.expanduser("~")
    return Path(base) / "news-agent" / "articles.jsonl"


def _within(article: Article, cutoff: date) -> bool:
    """Is this article inside the window?

    Falls back to *keeping* an article whose date we cannot parse. Feed dates
    are inconsistent, and dropping everything unparseable would silently shrink
    the window in a way nothing would report.
    """
    raw = (article.published or "")[:10]
    try:
        return date.fromisoformat(raw) >= cutoff
    except ValueError:
        return True


def load(*, window_days: int = DEFAULT_WINDOW_DAYS, sources: set[str] | None = None,
         path: Path | None = None) -> list[Article]:
    """Cached articles inside the window, newest first.

    `sources` filters to the feeds belonging to the queried category. Without
    it a cached World headline could answer an AI query, which would quietly
    break the category isolation the feed map exists to provide.
    """
    target = path or cache_path()
    if not target.exists():
        return []
    cutoff = date.today() - timedelta(days=window_days)
    articles: list[Article] = []
    try:
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                article = Article.model_validate_json(line)
            except ValueError:
                continue  # a corrupt line loses one article, not the cache
            if sources is not None and article.source not in sources:
                continue
            if _within(article, cutoff):
                articles.append(article)
    except OSError:
        return []
    articles.sort(key=lambda a: a.published or "", reverse=True)
    return articles


def save(articles: list[Article], *, window_days: int = DEFAULT_WINDOW_DAYS,
         path: Path | None = None) -> int:
    """Merge `articles` into the cache and prune anything outside the window.

    Returns the number of entries kept. Pruning on write means the file cannot
    grow without bound, and no separate cleanup step can be forgotten.
    """
    target = path or cache_path()
    cutoff = date.today() - timedelta(days=window_days)

    merged: dict[str, Article] = {}
    for article in load(window_days=window_days, path=target):
        merged[article.url] = article
    for article in articles:
        if article.url and _within(article, cutoff):
            # Fresh wins: a re-fetched article may have a corrected title.
            merged[article.url] = article

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "\n".join(a.model_dump_json() for a in merged.values()) + "\n",
            encoding="utf-8",
        )
    except OSError:
        return 0
    return len(merged)
