"""Langfuse ingestion contract.

Verified against Langfuse's own docs: usage_details keys must be **mutually
exclusive buckets** — every token counted exactly once — and cost_details keys
must mirror them. Violating that silently corrupts the cost display rather
than erroring, which is why it needs a test.
"""

from __future__ import annotations

import pytest

from news_agent.core.pricing import PRICING
from news_agent.core.types import TokenUsage

HAIKU = PRICING["claude-haiku-4-5"]  # $1/Mtok in, $5/Mtok out


def test_buckets_are_mutually_exclusive_and_sum_to_total():
    """The Anthropic API reports input_tokens as the *uncached remainder*, so
    the four fields never overlap. total must equal their sum."""
    u = TokenUsage(
        input_tokens=1000, output_tokens=200,
        cache_read_tokens=500, cache_creation_tokens=300,
    )
    details = u.usage_details()
    assert details["total"] == 2000 == u.total_tokens
    assert sum(v for k, v in details.items() if k != "total") == details["total"]


def test_cost_keys_mirror_usage_keys():
    """Regression: cache-read cost used to be folded into `input`, so the cost
    buckets did not line up with the token buckets."""
    u = TokenUsage(
        input_tokens=1000, output_tokens=200,
        cache_read_tokens=500, cache_creation_tokens=300,
    )
    assert set(HAIKU.breakdown_usd(u)) == set(u.usage_details())


def test_cost_total_is_the_sum_of_its_buckets():
    u = TokenUsage(input_tokens=1000, output_tokens=200, cache_read_tokens=500)
    costs = HAIKU.breakdown_usd(u)
    assert costs["total"] == pytest.approx(
        sum(v for k, v in costs.items() if k != "total")
    )


def test_cache_reads_bill_at_one_tenth_of_input():
    u = TokenUsage(cache_read_tokens=1_000_000)
    assert HAIKU.breakdown_usd(u)["cache_read_input_tokens"] == pytest.approx(0.10)


def test_cache_writes_bill_above_input():
    """Cache creation is ~1.25x the input rate, not 1x — treating it as plain
    input would under-report."""
    u = TokenUsage(cache_creation_tokens=1_000_000)
    assert HAIKU.breakdown_usd(u)["cache_creation_input_tokens"] == pytest.approx(1.25)


def test_zero_buckets_are_omitted_not_sent_as_zero():
    """Caching never engages on this prefix, so shipping two permanently-zero
    buckets on every trace is noise."""
    u = TokenUsage(input_tokens=100, output_tokens=50)
    details = u.usage_details()
    assert "cache_read_input_tokens" not in details
    assert "cache_creation_input_tokens" not in details
    assert set(HAIKU.breakdown_usd(u)) == {"input", "output", "total"}


def test_cache_creation_is_captured_from_the_api(monkeypatch):
    """Latent gap: the field was never read, so if caching ever engaged those
    tokens would vanish from both the total and the cost."""
    from types import SimpleNamespace

    from news_agent.agents.research.agent import _accumulate

    usage = TokenUsage()
    _accumulate(usage, SimpleNamespace(usage=SimpleNamespace(
        input_tokens=10, output_tokens=5,
        cache_read_input_tokens=3, cache_creation_input_tokens=7,
    )))
    assert usage.cache_creation_tokens == 7
    assert usage.total_tokens == 25
