"""Research agent: searches live RSS feeds and writes a structured digest."""

from .agent import DigestError, DigestResult, run_digest, search_headlines
from .models import Article, Category, DigestItem, HeadlineQuery, NewsDigest

__all__ = [
    "Article",
    "Category",
    "DigestError",
    "DigestItem",
    "DigestResult",
    "HeadlineQuery",
    "NewsDigest",
    "run_digest",
    "search_headlines",
]
