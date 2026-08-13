"""Agent tests. No network and no API key required — the Anthropic client is
injected as a stub, so these assert our wiring rather than Claude's behaviour."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from news_agent import DigestError, run_digest
from news_agent.agents.research.agent import search_headlines
from news_agent.agents.research.models import DigestItem, NewsDigest
from news_agent.core.types import TokenUsage
from news_agent.core.pricing import PRICING, format_cost
from news_agent.__main__ import render

DIGEST = NewsDigest(
    topic="ai regulation",
    overview="Regulators moved on AI this week.",
    items=[
        DigestItem(
            headline="EU opens inquiry",
            summary="An inquiry was opened into model providers.",
            why_it_matters="It sets precedent for enforcement.",
            sources=["https://example.com/c"],
        )
    ],
    coverage_note="Searched technology and world; coverage was moderate.",
)


def turn(*, digest=None, stop_reason="tool_use", inp=0, out=0, cached=0, text=None):
    """One message as the runner would yield it.

    `digest` is serialised into a text content block, mirroring what
    output_config.format actually produces on the wire.
    """
    if text is None and digest is not None:
        text = digest.model_dump_json()
    content = [SimpleNamespace(type="text", text=text)] if text is not None else []
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=inp, output_tokens=out, cache_read_input_tokens=cached
        ),
    )


def make_client(*messages):
    """A stand-in for anthropic.Anthropic that records the kwargs it was called
    with, so we can assert on how we configure the runner."""
    captured: dict = {}

    def tool_runner(**kwargs):
        captured.update(kwargs)
        return iter(messages)

    client = SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(tool_runner=tool_runner))
    )
    return client, captured


def test_returns_parsed_digest():
    client, _ = make_client(turn(digest=DIGEST, stop_reason="end_turn"))
    result = run_digest("ai regulation", client=client)
    assert result.digest == DIGEST
    assert result.digest.items[0].headline == "EU opens inquiry"


def test_defaults_to_the_cheapest_capable_model():
    """Cost regression guard: this agent does shallow reasoning over a strict
    schema, so it must not silently default to a frontier model."""
    client, captured = make_client(turn(digest=DIGEST, stop_reason="end_turn"))
    result = run_digest("ai regulation", client=client)
    assert captured["model"] == "claude-haiku-4-5"
    assert result.model == "claude-haiku-4-5"


def test_model_can_be_overridden():
    client, captured = make_client(turn(digest=DIGEST, stop_reason="end_turn"))
    result = run_digest("x", client=client, model="claude-sonnet-5")
    assert captured["model"] == "claude-sonnet-5"
    assert result.model == "claude-sonnet-5"


def test_runner_is_configured_for_structured_output():
    """output_config.format is what constrains the final message to the schema —
    regressing it would silently turn the digest back into unparsed prose."""
    client, captured = make_client(turn(digest=DIGEST, stop_reason="end_turn"))
    run_digest("ai regulation", client=client)

    schema = captured["output_config"]["format"]
    assert schema["type"] == "json_schema"
    assert set(schema["schema"]["properties"]) == {
        "topic", "overview", "items", "coverage_note"
    }
    assert captured["tools"] == [search_headlines]
    assert captured["messages"][0]["role"] == "user"
    assert "ai regulation" in captured["messages"][0]["content"]


def test_max_iterations_is_forwarded():
    client, captured = make_client(turn(digest=DIGEST, stop_reason="end_turn"))
    run_digest("x", client=client, max_iterations=3)
    assert captured["max_iterations"] == 3


def test_missing_parsed_output_raises_with_stop_reason_and_spend():
    client, _ = make_client(turn(stop_reason="max_tokens", inp=500))
    with pytest.raises(DigestError, match="max_tokens"):
        run_digest("ai regulation", client=client)


def test_no_messages_at_all_raises():
    client, _ = make_client()
    with pytest.raises(DigestError, match="no messages"):
        run_digest("x", client=client)


# --- cost accounting ---------------------------------------------------------


def test_usage_accumulates_across_every_turn():
    """The final message only carries its own usage — a single-turn read would
    under-report a 3-turn research loop."""
    client, _ = make_client(
        turn(inp=1000, out=100),
        turn(inp=2000, out=150, cached=500),
        turn(digest=DIGEST, stop_reason="end_turn", inp=3000, out=400),
    )
    result = run_digest("x", client=client)

    assert result.usage.turns == 3
    assert result.usage.input_tokens == 6000
    assert result.usage.output_tokens == 650
    assert result.usage.cache_read_tokens == 500
    assert result.usage.total_tokens == 7150


def test_missing_usage_fields_do_not_crash():
    """Usage fields vary by model/beta; a missing counter must not kill a run
    that otherwise succeeded."""
    bare = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=DIGEST.model_dump_json())],
        stop_reason="end_turn",
        usage=None,
    )
    client, _ = make_client(bare)
    result = run_digest("x", client=client)
    assert result.usage.turns == 0
    assert result.digest == DIGEST


def test_cost_estimate_matches_price_list():
    client, _ = make_client(
        turn(digest=DIGEST, stop_reason="end_turn", inp=1_000_000, out=1_000_000)
    )
    result = run_digest("x", client=client)
    # Haiku 4.5: $1/Mtok in + $5/Mtok out
    assert PRICING["claude-haiku-4-5"].estimate_usd(result.usage) == pytest.approx(6.0)
    assert "claude-haiku-4-5" in result.cost_line
    assert "$6.0000" in result.cost_line


def test_cached_tokens_bill_at_one_tenth():
    usage = TokenUsage(input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000)
    assert PRICING["claude-haiku-4-5"].estimate_usd(usage) == pytest.approx(0.10)


def test_haiku_is_five_times_cheaper_than_opus():
    usage = TokenUsage(input_tokens=100_000, output_tokens=20_000)
    haiku = PRICING["claude-haiku-4-5"].estimate_usd(usage)
    opus = PRICING["claude-opus-5"].estimate_usd(usage)
    assert opus == pytest.approx(haiku * 5)


def test_unknown_model_reports_tokens_without_inventing_a_price():
    line = format_cost("some-future-model", TokenUsage(input_tokens=10, turns=1))
    assert "10 in" in line
    assert "no price on file" in line
    assert "$" not in line


# --- the tool as Claude sees it ---------------------------------------------


def test_tool_schema_exposes_category_enum():
    """Claude must not be able to invent a category name."""
    schema = search_headlines.to_dict()["input_schema"]
    assert schema["$defs"]["Category"]["enum"] == [
        "ai",
        "top",
        "world",
        "business",
        "technology",
        "science",
    ]
    assert "category" in schema["required"]


def test_the_schema_tells_the_model_when_to_prefer_ai_over_technology():
    """Both categories could plausibly hold an AI story, so the description has
    to disambiguate — otherwise the model defaults to 'technology' and gets
    general tech news that merely mentions AI."""
    schema = search_headlines.to_dict()["input_schema"]
    description = schema["properties"]["category"]["description"].lower()
    assert "'ai'" in description
    assert "technology" in description


def test_tool_call_returns_json_string(monkeypatch):
    """The runner serialises tool results into the conversation, so the tool
    must return a string, not a model."""
    from news_agent.agents.research import tools as tools_mod

    monkeypatch.setattr(tools_mod, "_fetch", lambda s, u: ([], "failed"))
    out = search_headlines.call({"category": "science", "keywords": [], "limit": 3})
    parsed = json.loads(out)
    assert parsed["category"] == "science"
    assert parsed["article_count"] == 0


def test_tool_rejects_bad_category():
    with pytest.raises(Exception):
        search_headlines.call({"category": "sports"})


# --- structured-output schema constraints -----------------------------------


def test_digest_schema_is_valid_for_structured_outputs():
    """The API requires additionalProperties:false and every property in
    `required`. Adding a defaulted field to NewsDigest would break this."""

    def check(schema: dict) -> None:
        assert schema.get("additionalProperties") is False
        assert set(schema["properties"]) == set(schema["required"])

    schema = NewsDigest.model_json_schema()
    check(schema)
    check(schema["$defs"]["DigestItem"])


def test_digest_forbids_extra_fields():
    with pytest.raises(ValueError):
        NewsDigest(
            topic="t", overview="o", items=[], coverage_note="c", hallucinated="x"
        )


# --- rendering ---------------------------------------------------------------


def test_render_includes_content_and_sources():
    out = render(DIGEST)
    assert "EU opens inquiry" in out
    assert "https://example.com/c" in out
    assert "Why it matters" in out
    assert "Searched technology" in out


def test_render_handles_empty_digest():
    empty = NewsDigest(
        topic="obscure", overview="Nothing found.", items=[], coverage_note="Thin."
    )
    out = render(empty)
    assert "obscure" in out.lower()
    assert "Thin." in out


# --- what gets reported to Langfuse -----------------------------------------


def test_cost_and_usage_are_reported_to_langfuse(monkeypatch):
    sent = {}
    monkeypatch.setattr(
        "news_agent.agents.research.agent.report_generation", lambda **kw: sent.update(kw)
    )
    client, _ = make_client(
        turn(inp=1000, out=100),
        turn(digest=DIGEST, stop_reason="end_turn", inp=3000, out=400, cached=500),
    )
    run_digest("ai regulation", client=client)

    assert sent["model"] == "claude-haiku-4-5"
    assert sent["input"] == "ai regulation"
    assert sent["usage_details"]["input"] == 4000
    assert sent["usage_details"]["output"] == 500
    assert sent["usage_details"]["cache_read_input_tokens"] == 500

    # Cost buckets mirror the token buckets one-for-one — cache-read cost is
    # its own bucket, not folded into `input`, or Langfuse's per-type cost
    # display disagrees with the token counts beside it.
    cost = sent["cost_details"]
    assert set(cost) == set(sent["usage_details"])
    assert cost["input"] == pytest.approx(4000 / 1e6)              # $1/Mtok
    assert cost["output"] == pytest.approx(500 * 5 / 1e6)          # $5/Mtok
    assert cost["cache_read_input_tokens"] == pytest.approx(500 * 0.1 / 1e6)
    assert cost["total"] == pytest.approx(
        sum(v for k, v in cost.items() if k != "total")
    )

    assert sent["metadata"]["turns"] == 2
    assert sent["metadata"]["succeeded"] is True
    assert sent["output"]["topic"] == "ai regulation"


def test_spend_is_reported_even_when_the_digest_fails(monkeypatch):
    """A run that burned tokens without producing anything must still show its
    cost, rather than vanishing from the dashboard."""
    sent = {}
    monkeypatch.setattr(
        "news_agent.agents.research.agent.report_generation", lambda **kw: sent.update(kw)
    )
    client, _ = make_client(turn(stop_reason="max_tokens", inp=9000, out=200))
    with pytest.raises(DigestError):
        run_digest("x", client=client)

    assert sent["usage_details"]["input"] == 9000
    assert sent["cost_details"]["total"] > 0
    assert sent["metadata"]["succeeded"] is False
    assert sent["output"] is None


def test_unknown_model_reports_usage_but_no_cost(monkeypatch):
    sent = {}
    monkeypatch.setattr(
        "news_agent.agents.research.agent.report_generation", lambda **kw: sent.update(kw)
    )
    client, _ = make_client(turn(digest=DIGEST, stop_reason="end_turn", inp=10))
    run_digest("x", client=client, model="some-future-model")
    assert sent["usage_details"]["input"] == 10
    assert sent["cost_details"] is None


def test_tool_is_traced_as_a_nested_span():
    """So the fetched articles are visible in Langfuse, not just the digest."""
    import inspect

    from news_agent.agents.research import agent

    src = inspect.getsource(agent)
    assert 'as_type="tool"' in src
    # @observe must sit *under* @beta_tool, or the schema Claude sees changes.
    assert src.index("@beta_tool(input_schema=HeadlineQuery)") < src.index(
        '@observe(name="search_headlines"'
    )


# --- iteration budget --------------------------------------------------------


def test_exhausted_budget_error_names_a_concrete_remedy():
    """Regression: --max-iterations 2 burned tokens and returned nothing, and
    the old message didn't say what to do about it."""
    client, _ = make_client(turn(stop_reason="tool_use", inp=3899, out=351))
    with pytest.raises(DigestError, match=r"--max-iterations 6"):
        run_digest("x", client=client, max_iterations=2)


def test_other_stop_reasons_get_a_different_hint():
    client, _ = make_client(turn(stop_reason="max_tokens", inp=10))
    with pytest.raises(DigestError, match="before emitting a final answer"):
        run_digest("x", client=client)


def test_cli_rejects_too_few_iterations_before_spending(capsys):
    """argparse must fail up front — the whole point is that a bad value costs
    nothing rather than billing for an impossible run."""
    from news_agent.__main__ import main

    with pytest.raises(SystemExit) as exc:
        main(["topic", "--max-iterations", "2"])
    assert exc.value.code == 2
    assert "at least 3" in capsys.readouterr().err


def test_cli_accepts_the_minimum(capsys):
    from news_agent.__main__ import _iterations

    assert _iterations("3") == 3


def test_trace_url_is_read_while_the_span_is_open(monkeypatch):
    """Regression: reading it after run_digest returned logged
    'No active span in current context' and yielded nothing."""
    monkeypatch.setattr("news_agent.agents.research.agent.trace_url", lambda: "https://lf.test/t/1")
    client, _ = make_client(turn(digest=DIGEST, stop_reason="end_turn"))
    result = run_digest("x", client=client)
    assert result.trace_url == "https://lf.test/t/1"


# --- grounding: did the model actually cite what the tool returned? ----------


def _digest_citing(*urls):
    return NewsDigest(
        topic="t",
        overview="o",
        items=[
            DigestItem(headline="h", summary="s", why_it_matters="w", sources=list(urls))
        ],
        coverage_note="c",
    )


def test_grounded_when_every_source_came_from_the_tool(monkeypatch):
    from news_agent.agents.research import agent, tools as tools_mod
    from news_agent.agents.research.models import Article, HeadlineSearchResult

    real = HeadlineSearchResult(
        category="technology",
        article_count=1,
        articles=[
            Article(
                title="t", source="s", url="https://real.test/a", published="2026-08-13",
                summary="x",
            )
        ],
    )
    monkeypatch.setattr(
        tools_mod, "search_with_health",
        lambda q, **kw: (real, tools_mod.FeedHealth(ok=("s",)), 0)
    )

    def tool_runner(**kwargs):
        agent.search_headlines.call({"category": "technology"})
        return iter([turn(digest=_digest_citing("https://real.test/a"),
                          stop_reason="end_turn")])

    client = SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(tool_runner=tool_runner))
    )
    result = run_digest("t", client=client)
    assert result.ungrounded_sources == []
    assert result.is_grounded is True


def test_fabricated_source_url_is_caught(monkeypatch):
    """The system prompt says 'never invent a URL'. This is what checks it —
    a confident citation pointing nowhere is the worst failure mode for a news
    summariser, and it is invisible without this."""
    from news_agent.agents.research import agent, tools as tools_mod
    from news_agent.agents.research.models import Article, HeadlineSearchResult

    real = HeadlineSearchResult(
        category="technology",
        article_count=1,
        articles=[
            Article(
                title="t", source="s", url="https://real.test/a", published="2026-08-13",
                summary="x",
            )
        ],
    )
    monkeypatch.setattr(
        tools_mod, "search_with_health",
        lambda q, **kw: (real, tools_mod.FeedHealth(ok=("s",)), 0)
    )

    def tool_runner(**kwargs):
        agent.search_headlines.call({"category": "technology"})
        return iter([
            turn(
                digest=_digest_citing("https://real.test/a", "https://invented.test/b"),
                stop_reason="end_turn",
            )
        ])

    client = SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(tool_runner=tool_runner))
    )
    result = run_digest("t", client=client)
    assert result.ungrounded_sources == ["https://invented.test/b"]
    assert result.is_grounded is False


def test_grounding_state_does_not_leak_between_runs():
    """A ContextVar, not a global: URLs from an earlier run must not launder a
    later run's fabricated citation."""
    from news_agent.agents.research.agent import _seen_urls

    client, _ = make_client(turn(digest=_digest_citing("https://ghost.test/x"),
                                 stop_reason="end_turn"))
    result = run_digest("t", client=client)
    assert result.ungrounded_sources == ["https://ghost.test/x"]
    assert _seen_urls.get() is None  # reset after the run


def test_grounding_reported_to_langfuse(monkeypatch):
    sent = {}
    monkeypatch.setattr(
        "news_agent.agents.research.agent.report_generation", lambda **kw: sent.update(kw)
    )
    client, _ = make_client(turn(digest=_digest_citing("https://ghost.test/x"),
                                 stop_reason="end_turn"))
    run_digest("t", client=client)
    assert sent["metadata"]["ungrounded_sources"] == 1
    # No `grounded` bool: it restated the count next to it, and the Level 3
    # `grounding_passed` score already answers it in a chartable form.
    assert "grounded" not in sent["metadata"]


# --- claims must trace to the article, not just the URL ----------------------


def _run_with_article(monkeypatch, article_title, article_summary, item):
    """Run the agent with one canned article and one canned digest item."""
    from news_agent.agents.research import agent, tools as tools_mod
    from news_agent.agents.research.models import Article, HeadlineSearchResult

    result = HeadlineSearchResult(
        category="ai", article_count=1,
        articles=[Article(title=article_title, source="s",
                          url="https://real.test/a", published="2026-08-13",
                          summary=article_summary)],
    )
    monkeypatch.setattr(
        tools_mod, "search_with_health",
        lambda q, **kw: (result, tools_mod.FeedHealth(ok=("s",)), 0),
    )
    digest = NewsDigest(topic="t", overview="o", items=[item], coverage_note="c")

    def tool_runner(**kwargs):
        agent.search_headlines.call({"category": "ai"})
        return iter([turn(digest=digest, stop_reason="end_turn")])

    client = SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(tool_runner=tool_runner))
    )
    return run_digest("t", client=client)


def test_a_real_url_with_an_invented_figure_is_caught(monkeypatch):
    """The gap the URL check could never see: the citation is genuine, the
    number is not. This used to score a perfect 1.0."""
    out = _run_with_article(
        monkeypatch,
        "Anthropic raises funding round",
        "Anthropic raised $200 million in its latest round.",
        DigestItem(
            headline="Anthropic raises $2 billion",
            summary="Anthropic raised $2 billion in its latest funding round.",
            why_it_matters="w",
            sources=["https://real.test/a"],
        ),
    )
    assert out.ungrounded_sources == []        # the URL is real
    assert out.unsupported_claims             # but the claim is not
    assert "2billion" in out.unsupported_claims[0]
    assert out.is_grounded is False


def test_a_faithful_item_passes_both_checks(monkeypatch):
    out = _run_with_article(
        monkeypatch,
        "Anthropic raises $200 million",
        "Anthropic raised $200 million in its latest funding round.",
        DigestItem(
            headline="Anthropic raises $200 million",
            summary="Anthropic raised $200 million in its latest round.",
            why_it_matters="Investors continue backing frontier labs.",
            sources=["https://real.test/a"],
        ),
    )
    assert out.unsupported_claims == []
    assert out.is_grounded is True


def test_analysis_in_why_it_matters_is_not_penalised(monkeypatch):
    """`why_it_matters` is the model's own reasoning and is *supposed* to use
    words no source used. Checking it would punish what we asked for."""
    out = _run_with_article(
        monkeypatch,
        "Anthropic raises $200 million",
        "Anthropic raised $200 million in its latest funding round.",
        DigestItem(
            headline="Anthropic raises $200 million",
            summary="Anthropic raised $200 million in its latest round.",
            why_it_matters=(
                "Sovereign wealth appetite for frontier laboratories signals "
                "a structural realignment of capital allocation worldwide, "
                "with 97% implications for procurement strategy."
            ),
            sources=["https://real.test/a"],
        ),
    )
    assert out.unsupported_claims == []
