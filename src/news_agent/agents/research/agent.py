"""The agent: wires the RSS tool to Claude and returns a validated NewsDigest.

The agent loop itself is one call. ``tool_runner`` owns the request →
tool_use → tool_result → repeat cycle, and ``output_config.format`` makes the
API constrain the final message to the ``NewsDigest`` schema, so the last
message is guaranteed-valid JSON rather than prose we parse defensively.

We iterate the runner rather than calling ``until_done()`` so we can accumulate
token usage across every turn — the final message only carries its own.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field

import anthropic
from anthropic import beta_tool

from ...core.observability import observe, report_generation, trace_url
from ...core.pricing import format_cost, pricing_for
from ...core.grounding import MIN_OVERLAP, overlap, unsupported_figures
from ...core.provenance import provenance
from ...core.telemetry import EvalType, emit_eval, tool_call
from ...core.types import TokenUsage
from . import tools
from .cache import DEFAULT_WINDOW_DAYS
from .config import MAX_ITERATIONS, MAX_TOKENS, MIN_ITERATIONS, MODEL
from .instructions import SYSTEM_PROMPT
from .models import Category, HeadlineQuery, NewsDigest


# Every URL the tool actually returned during the current run. The system
# prompt tells the model never to invent a source, but nothing enforced it —
# this is what lets us *check* rather than hope. A ContextVar (not a global)
# so concurrent runs in the same process don't contaminate each other.
_seen_urls: ContextVar[set[str] | None] = ContextVar("seen_urls", default=None)

# How many days of accumulated articles the tool may draw on. A ContextVar for
# the same reason as above, and because the tool signature is what Claude
# reads — putting the window there would let the model choose it, and cost
# tokens on every call to describe a setting the operator owns.
_window_days: ContextVar[int] = ContextVar("window_days", default=DEFAULT_WINDOW_DAYS)

# url -> the article's own words, for checking that the digest reflects them.
_seen_text: ContextVar[dict[str, str] | None] = ContextVar("seen_text", default=None)


@beta_tool(input_schema=HeadlineQuery)
@observe(name="search_headlines", as_type="tool")
def search_headlines(category: str, keywords: list[str] | None = None, limit: int = 8) -> str:
    """Fetch current news headlines from live RSS feeds.

    Returns JSON with the matching articles and a `note` field explaining any
    problem (feeds down, no keyword matches).
    """
    # @observe sits *under* @beta_tool so the schema Claude sees is still
    # generated from the real signature, while each call shows up in Langfuse
    # as a nested span carrying the articles it fetched.
    query = HeadlineQuery(
        category=Category(category),
        keywords=keywords or [],
        limit=limit,
    )
    with tool_call(
        "search_headlines",
        side_effect=False,          # read-only: fetches feeds, mutates nothing
        args={"category": category, "keywords": keywords or [], "limit": limit},
    ) as telemetry:
        result, health, from_cache = tools.search_with_health(
            query, window_days=_window_days.get()
        )
        telemetry["articles_returned"] = result.article_count
        # How much of the answer came from the rolling window rather than
        # today's fetch. A narrow topic living entirely on cache is the signal
        # that the live feeds do not cover it.
        if from_cache:
            telemetry["from_cache"] = from_cache
        # The silent failure this catches: one dead feed out of five still
        # leaves `articles_returned` looking healthy, because the results are
        # merged before anyone counts them. Emitted only when something is
        # actually wrong — a field that is always `[]` is noise on every trace.
        if health.failed:
            telemetry["feeds_failed"] = list(health.failed)
        if health.empty:
            telemetry["feeds_empty"] = list(health.empty)
        telemetry["feeds_ok"] = len(health.ok)

    seen = _seen_urls.get()
    if seen is not None:
        seen.update(article.url for article in result.articles)

    # The article *text*, not just the URL. Verifying that a citation exists is
    # a different question from verifying that the digest says what the article
    # said, and the second one needs the words.
    corpus = _seen_text.get()
    if corpus is not None:
        for article in result.articles:
            corpus[article.url] = f"{article.title} {article.summary}"

    return result.model_dump_json()


class DigestError(RuntimeError):
    """The agent ran but did not produce a usable digest."""



@dataclass
class DigestResult:
    """The digest plus what it cost to produce."""

    digest: NewsDigest
    usage: TokenUsage
    model: str
    trace_url: str | None = None
    #: Source URLs in the digest that the tool never actually returned. Empty
    #: is the expected case; anything here means the model fabricated a
    #: citation and the digest should not be trusted verbatim.
    ungrounded_sources: list[str] = field(default_factory=list)
    #: Items whose figures or wording do not trace back to the article they
    #: cite. A real URL attached to an invented claim passes the check above
    #: with a perfect score, which is exactly the gap this closes.
    unsupported_claims: list[str] = field(default_factory=list)

    @property
    def cost_line(self) -> str:
        return format_cost(self.model, self.usage)

    @property
    def is_grounded(self) -> bool:
        """True only if citations exist *and* say what the digest claims."""
        return not self.ungrounded_sources and not self.unsupported_claims


def _extract_digest(message) -> NewsDigest | None:
    """Pull the validated digest out of the final message.

    `output_config.format` guarantees the text block is JSON matching the
    schema, so this is a parse, not a rescue. Returns None if the model never
    got as far as emitting a final text block.
    """
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) == "text":
            try:
                return NewsDigest.model_validate_json(block.text)
            except ValueError:
                return None
    return None


def _accumulate(usage: TokenUsage, message) -> None:
    """Fold one turn's usage into the running total.

    Defensive with getattr: usage fields vary by model and beta, and a missing
    counter must not crash a run that otherwise succeeded.
    """
    turn = getattr(message, "usage", None)
    if turn is None:
        return
    usage.turns += 1
    usage.input_tokens += getattr(turn, "input_tokens", 0) or 0
    usage.output_tokens += getattr(turn, "output_tokens", 0) or 0
    usage.cache_read_tokens += getattr(turn, "cache_read_input_tokens", 0) or 0
    usage.cache_creation_tokens += getattr(turn, "cache_creation_input_tokens", 0) or 0


# capture_input=False: the default would serialise every argument, including
# the whole Anthropic client object (and, in tests, the stub). We set a clean
# input explicitly below instead.
@observe(name="news-digest", as_type="generation", capture_input=False)
def run_digest(
    topic: str,
    *,
    client: anthropic.Anthropic | None = None,
    model: str | None = None,
    max_iterations: int = MAX_ITERATIONS,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> DigestResult:
    """Research `topic` against live news feeds and return a structured digest.

    Args:
        topic: What the user wants to read about.
        client: Injected for tests; defaults to a client built from the
            environment (ANTHROPIC_API_KEY, or an `ant auth login` profile).
        model: Override the model. Defaults to the cheapest capable one.
        max_iterations: Ceiling on tool-calling rounds. This is the main
            guard on cost — each round resends the whole conversation.
        window_days: How many days of accumulated articles the tool may draw
            on. Feeds hold roughly four days, so a narrow topic can miss news
            that existed but scrolled off before you asked.

    Raises:
        DigestError: if the model ends its turn without producing a digest.
    """
    client = client or anthropic.Anthropic()
    model = model or MODEL

    seen_urls: set[str] = set()
    seen_text: dict[str, str] = {}
    token = _seen_urls.set(seen_urls)
    text_token = _seen_text.set(seen_text)
    window = _window_days.set(window_days)
    try:
        return _run(topic, client, model, max_iterations, seen_urls, seen_text)
    finally:
        _seen_urls.reset(token)
        _seen_text.reset(text_token)
        _window_days.reset(window)


def _run(
    topic: str,
    client: anthropic.Anthropic,
    model: str,
    max_iterations: int,
    seen_urls: set[str],
    seen_text: dict[str, str] | None = None,
) -> DigestResult:
    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=[search_headlines],
        messages=[{"role": "user", "content": f"Give me a news digest on: {topic}"}],
        # The canonical parameter. The older `output_format=NewsDigest` is
        # ergonomic (it populates `.parsed_output`) but is deprecated, so we
        # pass the schema and validate ourselves.
        output_config={
            "format": {"type": "json_schema", "schema": NewsDigest.model_json_schema()}
        },
        max_iterations=max_iterations,
    )

    usage = TokenUsage()
    final = None
    for message in runner:
        _accumulate(usage, message)
        final = message

    if final is None:
        raise DigestError("Agent produced no messages at all.")

    digest = _extract_digest(final)
    ungrounded = _check_grounding(digest, seen_urls)
    unsupported = _check_support(digest, seen_text or {})
    # `final.model` is what the API says it ran — an alias like `claude-haiku-4-5`
    # is a pointer, and catching the day it moves is the entire point.
    _report_to_langfuse(
        topic, model, usage, digest, ungrounded,
        model_resolved=getattr(final, "model", None),
    )

    # Read the trace URL while the @observe span is still open. Calling this
    # after run_digest returns logs "No active span in current context".
    url = trace_url()

    if digest is None:
        hint = (
            f"The model used all {max_iterations} turns searching and never got "
            f"to write the digest — retry with --max-iterations {max_iterations + 4}."
            if final.stop_reason == "tool_use"
            else "The model stopped before emitting a final answer."
        )
        raise DigestError(
            f"Agent finished with stop_reason={final.stop_reason!r} but produced "
            f"no digest. {hint} Spent: {format_cost(model, usage)}"
            + (f"\nTrace: {url}" if url else "")
        )

    return DigestResult(
        digest=digest,
        usage=usage,
        model=model,
        trace_url=url,
        ungrounded_sources=ungrounded,
        unsupported_claims=unsupported,
    )


def _check_grounding(digest: NewsDigest | None, seen_urls: set[str]) -> list[str]:
    """Return any source URL the digest cites that the tool never returned.

    The system prompt says "never invent a URL". This is what verifies it.
    Cheap (a set lookup) and catches the failure mode that matters most in a
    news summariser: a confident citation pointing nowhere.
    """
    if digest is None:
        return []
    cited = {url for item in digest.items for url in item.sources}
    ungrounded = sorted(cited - seen_urls)

    # Level 3: this is a guardrail, so it belongs in the eval stream where it
    # can be charted over time — not only in the run's metadata blob.
    emit_eval(
        eval_type=EvalType.GROUNDEDNESS,
        eval_name="grounding",
        passed=not ungrounded,
        score=(len(cited) - len(ungrounded)) / len(cited) if cited else 1.0,
        evaluator="url-set-check",
        comment=f"{len(ungrounded)} of {len(cited)} cited sources were fabricated"
        if ungrounded else None,
        cited=len(cited),
        ungrounded=len(ungrounded),
    )
    return ungrounded


def _check_support(digest: NewsDigest | None, seen_text: dict[str, str]) -> list[str]:
    """Does each item say what its cited articles said?

    `_check_grounding` proves the citation exists. This asks the harder
    question — a model can cite a real article and describe something absent
    from it, and that passes the URL check with a perfect score.

    Two signals, both computed from text already in memory:

    - **figures** that appear in the item but in none of its sources. A
      paraphrase changes words; it must not change quantities.
    - **lexical overlap** below `MIN_OVERLAP`, which flags an item spun from
      nothing rather than one that is merely well written.

    `why_it_matters` is excluded from both. It is the model's own analysis and
    is *supposed* to contain words no source used — checking it would punish
    the thing we asked for.
    """
    if digest is None or not seen_text:
        return []

    problems: list[str] = []
    scores: list[float] = []
    for item in digest.items:
        source = " ".join(seen_text.get(url, "") for url in item.sources)
        if not source.strip():
            continue  # citation not in our corpus: that is _check_grounding's job
        claim = f"{item.headline} {item.summary}"

        missing = unsupported_figures(claim, source)
        if missing:
            problems.append(f"{item.headline!r}: figures not in any source: {missing}")

        score = overlap(claim, source)
        scores.append(score)
        if score < MIN_OVERLAP:
            problems.append(
                f"{item.headline!r}: only {score:.0%} of its wording appears in "
                f"its sources"
            )

    emit_eval(
        eval_type=EvalType.GROUNDEDNESS,
        eval_name="support",
        passed=not problems,
        score=sum(scores) / len(scores) if scores else 1.0,
        evaluator="figure-and-overlap-check",
        comment="; ".join(problems[:3]) if problems else None,
        items_checked=len(scores),
        problems=len(problems),
    )
    return problems


def _report_to_langfuse(
    topic: str,
    model: str,
    usage: TokenUsage,
    digest: NewsDigest | None,
    ungrounded: list[str],
    model_resolved: str | None = None,
) -> None:
    """Attach model, token usage and dollar cost to the trace.

    Reported even when the digest failed, so a run that burned tokens without
    producing anything still shows its spend rather than vanishing.
    """
    price = pricing_for(model)
    report_generation(
        model=model,
        input=topic,
        output=digest.model_dump() if digest is not None else None,
        usage_details=usage.usage_details(),
        cost_details=price.breakdown_usd(usage) if price else None,
        # No `grounded` flag: it was `ungrounded_sources == 0` restated as a
        # bool, sitting next to the number it was derived from — and the
        # Level 3 `grounding_passed` score already answers it, chartably.
        metadata={
            "turns": usage.turns,
            "items": len(digest.items) if digest is not None else 0,
            "succeeded": digest is not None,
            "ungrounded_sources": len(ungrounded),
            **provenance(
                prompt_parts=(SYSTEM_PROMPT,), model_resolved=model_resolved
            ),
        },
    )
