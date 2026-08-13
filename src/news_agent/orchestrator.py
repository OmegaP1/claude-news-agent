"""The orchestrator: research → judge → select → persist.

Deliberately a **plain pipeline, not a second agent**. Every step here is
deterministic and the order is fixed, so there is nothing for a model to
decide. Wrapping it in another agent loop would add latency, cost and failure
modes to buy nothing. The agentic part is confined to stage 1, where the model
genuinely chooses what to search for.

    research   (Haiku 4.5, tools)   → NewsDigest
    judge      (Sonnet 5, no tools) → ranked ScoredItems
    select     (code)               → top N above the quality floor
    persist    (code)               → Obsidian notes

Selection and persistence are code, not tool calls, because "only the top 3
enter the vault" is a rule, not a judgement call.
"""

from __future__ import annotations

from collections.abc import Callable
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import anthropic

from .agents.judge import MIN_COMPOSITE, JudgeResult, ScoredItem, judge_digest
from .agents.research import DigestResult, run_digest
from .core.budget import BudgetExceeded, check_budget
from .core.observability import observe
from .core.telemetry import EvalType, Outcome, agent_run, emit_eval, emit_run_outcome
from .core.pricing import pricing_for
from .core.types import TokenUsage
from .sinks.obsidian import VaultWriteResult, write_digest

TOP_N = 3


@dataclass
class PipelineResult:
    research: DigestResult
    judged: JudgeResult
    selected: list[ScoredItem]
    vault_result: VaultWriteResult | None
    #: Items that made the top N but were dropped for scoring below the floor.
    below_floor: list[ScoredItem]
    #: 'judge' unless a human changed the selection during review.
    selected_by: str = "judge"
    #: What the judge picked before any human input — the comparison a golden
    #: fixture needs. `selected` alone cannot answer "did the human disagree?"
    #: once the hook has replaced it.
    judge_selected: list[ScoredItem] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        """Combined spend across both models — they price differently, so this
        cannot be a single token count."""
        total = 0.0
        for model, usage in (
            (self.research.model, self.research.usage),
            (self.judged.model, self.judged.usage),
        ):
            price = pricing_for(model)
            if price:
                total += price.estimate_usd(usage)
        return total

    @property
    def cost_line(self) -> str:
        research = self.research.usage
        judge = self.judged.usage
        return (
            f"{self.research.model} {research.input_tokens:,}in/"
            f"{research.output_tokens:,}out + "
            f"{self.judged.model} {judge.input_tokens:,}in/"
            f"{judge.output_tokens:,}out "
            f"≈ ${self.total_cost_usd:.4f}"
        )


@observe(name="news-pipeline", capture_input=False)
def run_pipeline(
    topic: str,
    *,
    client: anthropic.Anthropic | None = None,
    model: str | None = None,
    judge_model: str | None = None,
    max_iterations: int = 6,
    top_n: int = TOP_N,
    floor: float = MIN_COMPOSITE,
    vault: Path | None = None,
    day: date | None = None,
    dry_run: bool = False,
    force: bool = False,
    max_usd: float | None = None,
    window_days: int = 7,
    select_hook: Callable[[list[ScoredItem], list[ScoredItem]], tuple[list[ScoredItem], str]]
    | None = None,
) -> PipelineResult:
    """Research, judge, and file the best `top_n` stories.

    Args:
        dry_run: Do everything except write to the vault. Useful for seeing
            what *would* be filed before touching a real vault.
        select_hook: Optional human-in-the-loop. Receives (ranked, selected)
            and returns (final_selection, who_chose). Injected rather than
            called directly so the pipeline has no terminal I/O in it and
            stays testable.

    Raises:
        DigestError: research produced no digest.
        JudgeError: the judge returned something unusable.
    """
    client = client or anthropic.Anthropic()

    # Level 1: one id shared by research, judge and the vault write. Without
    # it those are three unrelated observations in the UI.
    run_id = uuid.uuid4().hex

    with agent_run(run_id, tags=["news-pipeline"], topic=topic):
        try:
            return _run_stages(
                topic, client, model, judge_model, max_iterations,
                top_n, floor, vault, day, dry_run, select_hook, run_id,
                force, max_usd, window_days,
            )
        except BaseException as exc:
            # Without this, only successful runs carry a Level 1 outcome and
            # the dashboard shows survivorship bias: every failure is simply
            # absent rather than visibly failed.
            emit_run_outcome(_outcome_for(exc), error_type=type(exc).__name__)
            raise


def _outcome_for(exc: BaseException) -> Outcome:
    """Map a terminal exception to the Level 1 outcome enum."""
    from .agents.judge import JudgeError
    from .agents.research import DigestError
    from .review import ReviewAborted

    if isinstance(exc, ReviewAborted):
        return Outcome.CANCELLED          # a person declined; not a failure
    if isinstance(exc, BudgetExceeded):
        # The one true guardrail in the system: a limit that stopped the run
        # on purpose. Not an exception in the "something broke" sense.
        return Outcome.GUARDRAIL_BLOCK
    if isinstance(exc, DigestError):
        return Outcome.GOAL_UNMET         # ran, produced nothing usable
    if isinstance(exc, JudgeError):
        return Outcome.APPLICATION_EXCEPTION
    if isinstance(exc, KeyboardInterrupt):
        return Outcome.CANCELLED
    return Outcome.APPLICATION_EXCEPTION


def _run_stages(
    topic, client, model, judge_model, max_iterations,
    top_n, floor, vault, day, dry_run, select_hook, run_id,
    force=False, max_usd=None, window_days=7,
) -> PipelineResult:
    research = run_digest(
        topic, client=client, model=model, max_iterations=max_iterations,
        window_days=window_days,
    )

    # The only checkpoint where stopping still saves money. Research is already
    # paid for by now; the judge runs on a pricier model and has not started.
    # Checking after the judge would be a report, not a guard.
    research_price = pricing_for(research.model)
    if research_price is not None:
        check_budget(research_price.estimate_usd(research.usage), max_usd, "research")

    judged = judge_digest(
        research.digest,
        client=client,
        **({"model": judge_model} if judge_model else {}),
    )

    ranked_top = judged.ranked[:top_n]
    selected = [s for s in ranked_top if s.composite >= floor]
    below_floor = [s for s in ranked_top if s.composite < floor]

    # Kept separately from `selected`, which the hook may replace. Capturing a
    # golden fixture needs both: what the judge wanted and what the human took.
    judge_selected = list(selected)

    selected_by = "judge"
    human_interventions = 0
    if select_hook is not None:
        # The human sees the whole ranking, not just the top N — the point of
        # review is being able to promote something the cut-off excluded.
        before = [id(s) for s in selected]
        selected, selected_by = select_hook(judged.ranked, selected)
        if [id(s) for s in selected] != before:
            human_interventions = 1
            # Level 3: an override is a human verdict on the judge's output —
            # the most valuable eval signal there is, and free to collect.
            emit_eval(
                eval_type=EvalType.HUMAN_FEEDBACK,
                eval_name="judge_accepted",
                passed=False,
                evaluator="human",
                comment="human changed the judge's selection",
                kept=sum(1 for s in selected if id(s) in before),
                added=sum(1 for s in selected if id(s) not in before),
            )
        else:
            emit_eval(
                eval_type=EvalType.HUMAN_FEEDBACK,
                eval_name="judge_accepted",
                passed=True,
                evaluator="human",
            )

    vault_result = None
    if selected and not dry_run:
        vault_result = write_digest(
            topic,
            research.digest.overview,
            selected,
            vault=vault,
            day=day,
            models={"generator": research.model, "judge": judged.model},
            selected_by=selected_by,
            force=force,
        )

    # Dropped from this payload as duplicates of what Langfuse already holds:
    #   run_id          -> already the trace's session_id
    #   duration_ms     -> the span carries start and end
    #   total_cost_usd  -> aggregated from the cost_details on each generation
    #   generation_count-> visible in the span tree, and correlates with cost
    #   human_intervention_count -> `selected_by` answers it, and better
    emit_run_outcome(
        Outcome.SUCCESS if selected else Outcome.GOAL_UNMET,
        items_judged=len(judged.ranked),
        items_filed=len(selected),
        selected_by=selected_by,
    )

    return PipelineResult(
        research=research,
        judged=judged,
        selected=selected,
        vault_result=vault_result,
        below_floor=below_floor,
        selected_by=selected_by,
        judge_selected=judge_selected,
    )


def combined_usage(result: PipelineResult) -> TokenUsage:
    """Token totals across both stages.

    Useful for a single 'how much did this run use' number — but note the two
    stages bill at different rates, so prefer `total_cost_usd` for money.
    """
    research, judged = result.research.usage, result.judged.usage
    return TokenUsage(
        input_tokens=research.input_tokens + judged.input_tokens,
        output_tokens=research.output_tokens + judged.output_tokens,
        cache_read_tokens=research.cache_read_tokens + judged.cache_read_tokens,
        cache_creation_tokens=(
            research.cache_creation_tokens + judged.cache_creation_tokens
        ),
        turns=research.turns + judged.turns,
    )
