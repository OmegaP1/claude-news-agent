"""LLM-as-a-judge: score the digest's items and rank them.

Design decisions, and why — most "LLM judge" code gets these wrong:

1. **A different model from the generator.** Models exhibit self-preference
   bias: they rate their own output more highly. A Haiku judge grading a Haiku
   digest is marking its own homework, and you never see it fail. Default here
   is Sonnet 5 against a Haiku generator.

2. **No `temperature=0`.** The standard advice does not apply — Sonnet 5
   *rejects* sampling parameters with a 400. Consistency has to come from a
   tight rubric and a coarse scale instead of from decoding settings.

3. **Reasoning before scores.** Enforced structurally by field order in
   `ItemVerdict`, not by asking politely. Scores first would just be
   rationalised after the fact.

4. **Coarse 1-5 enum scales.** Finer scales add noise, not signal. `Literal`
   rather than `int` because structured outputs enforce enums but silently
   ignore numeric min/max.

5. **The composite is computed in code.** The model supplies judgement on
   named dimensions; Python does the arithmetic. That keeps the weighting
   auditable, tunable, and deterministic — and stops the model from quietly
   contradicting its own dimension scores with a vibe-based overall number.

6. **Each item scored on its own merits**, not ranked against each other in
   one pass, which reduces (though does not eliminate) position bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anthropic

from ...core.observability import observe, report_generation
from ...core.pricing import format_cost, pricing_for
from ...core.provenance import provenance
from ...core.telemetry import EvalType, emit_eval
from ...core.untrusted import fence
from ...core.types import TokenUsage
from ..research.models import DigestItem, NewsDigest
from .config import EFFORT, MAX_TOKENS, MIN_COMPOSITE, MODEL, WEIGHTS
from .instructions import JUDGE_SYSTEM
from .models import ItemVerdict, JudgeVerdicts, ScoredItem

@dataclass
class JudgeResult:
    """Ranked items plus what the judging cost."""

    ranked: list[ScoredItem]
    usage: TokenUsage
    model: str
    #: Items the judge returned a verdict for that we could not match back to a
    #: digest item, and digest items it silently skipped.
    unmatched_verdicts: list[int] = field(default_factory=list)
    unjudged_items: list[int] = field(default_factory=list)

    @property
    def cost_line(self) -> str:
        return format_cost(self.model, self.usage)

    def top(self, n: int = 3, floor: float = MIN_COMPOSITE) -> list[ScoredItem]:
        """The best `n` items that clear the quality floor."""
        return [s for s in self.ranked[:n] if s.composite >= floor]


def composite(verdict: ItemVerdict) -> float:
    """Weighted mean of the dimension scores. Deterministic, in code."""
    return round(
        verdict.significance * WEIGHTS["significance"]
        + verdict.relevance * WEIGHTS["relevance"]
        + verdict.novelty * WEIGHTS["novelty"]
        + verdict.evidence * WEIGHTS["evidence"],
        3,
    )


def _render_items(topic: str, items: list[DigestItem]) -> str:
    """Present the items numbered, so verdicts can be bound back by index.

    Item text is fenced as untrusted. It descends from RSS content an attacker
    may control, and a judge is a uniquely attractive target: a headline that
    talks its own way into a 5/5 buys attention cheaply. The item *numbers* and
    field labels stay outside the fence so the structure the model navigates by
    is ours, not the attacker's.
    """
    lines = [f"Topic the reader asked about: {topic}", "", "Items to score:", ""]
    for index, item in enumerate(items, 1):
        lines.append(f"[{index}] {fence(item.headline)}")
        lines.append(f"    {fence(item.summary)}")
        lines.append(f"    Stated significance: {fence(item.why_it_matters)}")
        lines.append(f"    Sources: {len(item.sources)}")
        lines.append("")
    return "\n".join(lines)


@observe(name="judge", as_type="generation", capture_input=False)
def judge_digest(
    digest: NewsDigest,
    *,
    client: anthropic.Anthropic | None = None,
    model: str = MODEL,
) -> JudgeResult:
    """Score every item in `digest` and return them ranked best-first.

    Ranking is stable: ties keep the digest's original ordering, so a rerun
    with identical scores produces an identical ranking.
    """
    if not digest.items:
        return JudgeResult(ranked=[], usage=TokenUsage(), model=model)

    client = client or anthropic.Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": _render_items(digest.topic, digest.items)}],
        output_config={
            "effort": EFFORT,
            "format": {
                "type": "json_schema",
                "schema": JudgeVerdicts.model_json_schema(),
            },
        },
    )

    usage = TokenUsage(
        input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
        output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        turns=1,
    )

    verdicts = _parse(response)
    ranked, unmatched, unjudged = _rank(digest.items, verdicts)
    _emit_quality_evals(ranked, model)
    _report(
        digest, model, usage, ranked, unmatched, unjudged,
        model_resolved=getattr(response, "model", None),
    )

    return JudgeResult(
        ranked=ranked,
        usage=usage,
        model=model,
        unmatched_verdicts=unmatched,
        unjudged_items=unjudged,
    )


def _parse(response) -> list[ItemVerdict]:
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "text":
            try:
                return JudgeVerdicts.model_validate_json(block.text).verdicts
            except ValueError as exc:
                raise JudgeError(f"Judge returned unparseable output: {exc}") from exc
    raise JudgeError("Judge returned no text block.")


def _rank(
    items: list[DigestItem], verdicts: list[ItemVerdict]
) -> tuple[list[ScoredItem], list[int], list[int]]:
    """Bind verdicts to items by index and sort.

    The judge is told to echo `item_index`, but a returned index is still
    untrusted input — an out-of-range or duplicate one must not raise, or one
    bad field would throw away a whole paid run.
    """
    by_index: dict[int, ItemVerdict] = {}
    unmatched: list[int] = []
    for verdict in verdicts:
        if 1 <= verdict.item_index <= len(items) and verdict.item_index not in by_index:
            by_index[verdict.item_index] = verdict
        else:
            unmatched.append(verdict.item_index)

    scored = [
        ScoredItem(item=item, verdict=by_index[i], composite=composite(by_index[i]))
        for i, item in enumerate(items, 1)
        if i in by_index
    ]
    unjudged = [i for i in range(1, len(items) + 1) if i not in by_index]

    # Stable: equal composites keep the digest's original order.
    scored.sort(key=lambda s: -s.composite)
    return scored, unmatched, unjudged


def _emit_quality_evals(ranked, model: str) -> None:
    """Level 3: the judge's scores are evals, so publish them as scores.

    One aggregate per dimension rather than one per item — per-item scores
    would swamp the eval stream and are already visible in the trace body.
    """
    if not ranked:
        return
    evaluator = f"llm-as-judge:{model}"
    dimensions = ranked[0].scores
    for dimension in dimensions:
        mean = sum(s.scores[dimension] for s in ranked) / len(ranked)
        emit_eval(
            eval_type=EvalType.CUSTOM,
            eval_name=f"judge_{dimension}",
            score=round(mean, 3),
            evaluator=evaluator,
            items=len(ranked),
        )
    emit_eval(
        eval_type=EvalType.CUSTOM,
        eval_name="judge_composite",
        score=round(sum(s.composite for s in ranked) / len(ranked), 3),
        evaluator=evaluator,
        items=len(ranked),
    )


def _judge_prompt_parts() -> tuple[str, ...]:
    """Everything that shapes the judge's behaviour, for the provenance hash.

    The weights belong here even though they never reach the model: the
    composite is computed in code, so re-weighting changes the ranking without
    changing a single prompt token. Hashing the system prompt alone would
    report "same prompt" across a change that altered every score.
    """
    return (JUDGE_SYSTEM, repr(sorted(WEIGHTS.items())), str(MIN_COMPOSITE))


def _report(digest, model, usage, ranked, unmatched, unjudged, model_resolved=None) -> None:
    price = pricing_for(model)
    report_generation(
        model=model,
        input={"topic": digest.topic, "items": len(digest.items)},
        output=[
            {"headline": s.item.headline, "composite": s.composite, **s.scores}
            for s in ranked
        ],
        usage_details=usage.usage_details(),
        cost_details=price.breakdown_usd(usage) if price else None,
        metadata={
            "judged": len(ranked),
            "unmatched_verdicts": len(unmatched),
            "unjudged_items": len(unjudged),
            "weights": WEIGHTS,
            **provenance(
                prompt_parts=_judge_prompt_parts(), model_resolved=model_resolved
            ),
        },
    )


class JudgeError(RuntimeError):
    """The judge ran but returned something unusable."""
