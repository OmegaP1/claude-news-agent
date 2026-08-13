"""CLI: python -m news_agent "AI regulation" """

from __future__ import annotations

import argparse
import os
import pathlib
import sys

from . import DOTENV_LOADED  # importing the package loads .env first
from .agents.judge import MIN_COMPOSITE
from .agents.research import DigestError, NewsDigest, run_digest
from .agents.research.config import MIN_ITERATIONS, MODEL
from .core.doctor import check
from .core.observability import flush, status
from .core.pricing import PRICING
from .orchestrator import run_pipeline


def _iterations(value: str) -> int:
    """Reject a budget too small to finish — argparse fails before any API call,
    so a bad value costs nothing instead of billing for a run that cannot
    possibly produce a digest."""
    number = int(value)
    if number < MIN_ITERATIONS:
        raise argparse.ArgumentTypeError(
            f"must be at least {MIN_ITERATIONS}: the model needs one round per "
            f"search plus one to write the digest, so {number} would burn tokens "
            f"and return nothing."
        )
    return number


def _force_utf8() -> None:
    """Make stdout/stderr survive non-ASCII output.

    Windows consoles default to cp1252, which cannot encode the glyphs we use
    (— → ≈ …) or anything non-Latin in a headline. Without this the CLI dies
    with UnicodeEncodeError at print time, after the API call has already been
    paid for. `errors="replace"` means a terminal that still cannot render a
    character degrades to '?' instead of crashing.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def render(digest: NewsDigest) -> str:
    """Format a digest for a terminal."""
    lines = [
        "",
        f"  {digest.topic.upper()}",
        f"  {'─' * max(len(digest.topic), 10)}",
        "",
        f"  {digest.overview}",
        "",
    ]
    for i, item in enumerate(digest.items, 1):
        lines.append(f"  {i}. {item.headline}")
        lines.append(f"     {item.summary}")
        lines.append(f"     Why it matters: {item.why_it_matters}")
        for url in item.sources:
            lines.append(f"     → {url}")
        lines.append("")
    lines.append(f"  Coverage: {digest.coverage_note}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    # Before parse_args: --help prints and exits from inside it.
    _force_utf8()

    parser = argparse.ArgumentParser(
        prog="news-agent",
        description="Research a topic against live news feeds using Claude.",
    )
    parser.add_argument(
        "topic",
        nargs="?",
        help='Topic to research, e.g. "AI regulation"',
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify your API key and billing without running a digest (~$0.00001).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit raw JSON instead of formatted text."
    )
    parser.add_argument(
        "--wiki",
        action="store_true",
        help=(
            "Run the full pipeline: research, judge the items with a second "
            "model, and file the top 3 into your Obsidian vault."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="How many items may enter the vault with --wiki (default: 3).",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=MIN_COMPOSITE,
        help=(
            f"Quality floor 1-5; items below it are dropped even if they make "
            f"the top N (default: {MIN_COMPOSITE})."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        choices=sorted(PRICING),
        help="Model to judge with (default: claude-sonnet-5, i.e. not the generator).",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help=(
            "With --wiki: show the ranking and let you adjust the selection "
            "before anything is written (e.g. '-2 +4')."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --wiki: judge and rank, but do not write to the vault.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite vault notes that already exist with different content. "
            "Without this, a conflicting note is reported and left alone."
        ),
    )
    parser.add_argument(
        "--max-usd",
        type=float,
        default=None,
        help=(
            "Hard spend ceiling for this run. Checked after research and before "
            "the judge — the one point where stopping still saves money."
        ),
    )
    parser.add_argument(
        "--capture-golden",
        action="store_true",
        help=(
            "With --review: save this run as a golden fixture (digest + your "
            "picks) so judge changes can be replayed against it offline."
        ),
    )
    parser.add_argument(
        "--feedback",
        action="store_true",
        help="Report whether the judge is earning its cost, from captured fixtures.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=7,
        help=(
            "How many days of accumulated articles to search (default: 7). "
            "Feeds only hold about four days, so the extra comes from a local "
            "cache that fills up as you run."
        ),
    )
    parser.add_argument(
        "--topics-file",
        default=None,
        metavar="PATH",
        help=(
            "Build the golden dataset from a list of topics, one per line "
            "(# comments allowed). Runs each as --wiki --review "
            "--capture-golden, and skips topics already captured, so an "
            "interrupted session resumes where it stopped."
        ),
    )
    parser.add_argument(
        "--push-dataset",
        action="store_true",
        help=(
            "Mirror the captured fixtures into Langfuse Datasets for the UI. "
            "One-way: the files stay authoritative."
        ),
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help=(
            "Re-judge every captured fixture and compare against your picks. "
            "The regression gate for rubric changes. Costs ~1c per fixture "
            "(judge only — research is not re-run)."
        ),
    )
    parser.add_argument(
        "--replay-limit",
        type=int,
        default=None,
        metavar="N",
        help="With --replay: only the N most recent fixtures, to bound the spend.",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        choices=sorted(PRICING),
        help=f"Model to use (default: {MODEL}).",
    )
    parser.add_argument(
        "--max-iterations",
        type=_iterations,
        default=6,
        help=(
            f"Ceiling on tool-calling rounds — the main cost guard "
            f"(default: 6, minimum {MIN_ITERATIONS}). Each search costs one "
            f"round and the digest itself costs one more."
        ),
    )
    args = parser.parse_args(argv)

    if DOTENV_LOADED:
        print(f"[loaded {', '.join(DOTENV_LOADED)} from .env]", file=sys.stderr)

    if args.check:
        result = check(model=args.model)
        print(result.report())
        return 0 if result.ok else 1

    # Reads local fixtures only — no API key, no network, no spend.
    if args.feedback:
        return _feedback()

    if args.push_dataset:
        return _push_dataset()

    if args.replay:
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("--replay calls the judge and needs ANTHROPIC_API_KEY.", file=sys.stderr)
            return 2
        return _replay(args)

    if args.topics_file:
        if args.topic:
            parser.error("--topics-file replaces the topic argument, not adds to it")
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("--topics-file runs real digests and needs ANTHROPIC_API_KEY.",
                  file=sys.stderr)
            return 2
        return _batch(args, parser)

    if not args.topic:
        parser.error(
            "a topic is required (or pass --check, --feedback, --replay or "
            "--topics-file)"
        )

    if args.capture_golden and not (args.wiki and args.review):
        parser.error("--capture-golden needs --wiki --review: the fixture is your verdict")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set. Put it in a .env file, export it, "
            "or run `ant auth login`.\n"
            "Note: this bills your API account per token — a Claude Code or "
            "Claude.ai subscription does not cover it.\n"
            "Run `python -m news_agent --check` to test your setup.",
            file=sys.stderr,
        )
        return 2

    print(f"[{status()}]", file=sys.stderr)
    print(f"Researching {args.topic!r} with {args.model}…", file=sys.stderr)

    if args.wiki:
        return _wiki(args)

    try:
        result = run_digest(
            args.topic, model=args.model, max_iterations=args.max_iterations,
            window_days=args.window_days,
        )
    except DigestError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        flush()

    print(result.digest.model_dump_json(indent=2) if args.json else render(result.digest))
    print(f"[{result.cost_line}]", file=sys.stderr)
    if result.trace_url:
        print(f"[trace: {result.trace_url}]", file=sys.stderr)

    if not result.is_grounded:
        # Exit 3, not 0: a digest that cites a source the tool never returned —
        # or states a figure that source never gave — is a correctness failure,
        # and a pipeline consuming this should notice.
        _print_grounding_problems(result)
        return 3
    return 0


def _print_grounding_problems(result) -> None:
    if result.ungrounded_sources:
        print(
            f"\nWARNING: {len(result.ungrounded_sources)} cited source(s) were "
            "never returned by the tool — the model fabricated them:",
            file=sys.stderr,
        )
        for url in result.ungrounded_sources:
            print(f"  - {url}", file=sys.stderr)
    if result.unsupported_claims:
        print(
            f"\nWARNING: {len(result.unsupported_claims)} item(s) make claims "
            "their cited article does not support:",
            file=sys.stderr,
        )
        for problem in result.unsupported_claims:
            print(f"  - {problem}", file=sys.stderr)



def render_pipeline(result) -> str:
    """Show what was judged, what was selected, and where it landed."""
    lines = ["", "  RANKED BY THE JUDGE", "  " + "─" * 20, ""]
    for scored in result.judged.ranked:
        marks = " ".join(f"{k[:3]}={v}" for k, v in scored.scores.items())
        chosen = "✓" if scored in result.selected else " "
        lines.append(f"  {chosen} {scored.composite:>5.2f}  {scored.item.headline}")
        lines.append(f"          {marks}")
        lines.append(f"          {scored.verdict.reasoning}")
        lines.append("")

    if result.below_floor:
        lines.append(
            f"  Dropped for scoring below the floor: {len(result.below_floor)} "
            "(a thin story is worse than no story)."
        )
        lines.append("")

    if result.vault_result:
        vr = result.vault_result
        chooser = (
            "you (judge overridden)" if result.selected_by == "human" else "the judge"
        )
        lines.append(f"  Filed {len(vr.item_notes)} notes into {vr.vault}")
        lines.append(f"    chosen by: {chooser}")
        lines.append(f"    index: {vr.index_note.name}")
        for path in vr.item_notes:
            lines.append(f"    item : {path.name}")
        for problem in vr.skipped:
            lines.append(f"    SKIPPED: {problem}")
    else:
        lines.append("  Nothing written to the vault.")
    lines.append("")
    return "\n".join(lines)


def _feedback() -> int:
    """Is the judge earning its cost? Answered from captured fixtures."""
    from .evals.golden import load_all, verdict_for

    fixtures = load_all()
    verdict = verdict_for(fixtures)
    swaps = sum(1 for f in fixtures if f.swapped)
    adds = sum(1 for f in fixtures if f.only_added)

    print(f"\n  JUDGE ACCEPTANCE — {verdict.sample} captured run(s)")
    print("  " + "─" * 46)
    print(f"\n  exact match      : {verdict.rate:>4.0%}   you changed nothing")
    print(f"  ranking held     : {verdict.ranking:>4.0%}   you kept every pick "
          f"({swaps} swap{'s' if swaps != 1 else ''})")
    print(f"  floor too strict : {verdict.floor:>4.0%}   you added back a "
          f"dropped item ({adds})")
    print(f"\n  decision : {verdict.decision}")
    print(f"\n  {verdict.detail}\n")
    return 0


def read_topics(text: str) -> list[str]:
    """One topic per line; `#` comments and blank lines ignored.

    Duplicates are dropped, preserving order — a list you have edited a few
    times accumulates them, and paying twice for the same topic to get a
    second fixture that measures the same thing is pure waste.
    """
    seen: set[str] = set()
    topics: list[str] = []
    for line in text.splitlines():
        topic = line.split("#", 1)[0].strip()
        if topic and topic.casefold() not in seen:
            seen.add(topic.casefold())
            topics.append(topic)
    return topics


def _batch(args, parser) -> int:
    """Work through a topics file, capturing a fixture per topic.

    Refuses without a terminal. The review would fall through, nothing would
    be captured, and you would have paid for every run in the file to produce
    an empty dataset — the single most expensive way to get nothing.
    """
    from .evals.golden import captured_topics

    path = pathlib.Path(args.topics_file)
    try:
        topics = read_topics(path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"Cannot read {path}: {exc}", file=sys.stderr)
        return 1
    if not topics:
        print(f"{path} has no topics in it.", file=sys.stderr)
        return 1

    if not sys.stdin.isatty():
        print(
            "--topics-file needs a terminal: every run would fall through the "
            "review unreviewed, capture nothing, and still be billed.",
            file=sys.stderr,
        )
        return 2

    done = captured_topics()
    pending = [t for t in topics if t.casefold() not in done]
    skipped = len(topics) - len(pending)

    print(f"\n  {len(topics)} topic(s) in {path.name}", file=sys.stderr)
    if skipped:
        print(f"  {skipped} already captured — resuming with {len(pending)}",
              file=sys.stderr)
    if not pending:
        print("  Nothing left to capture.\n", file=sys.stderr)
        return 0
    print(f"  Roughly ${0.03 * len(pending):.2f}. Ctrl-C between runs to stop.\n",
          file=sys.stderr)

    captured = failed = 0
    for number, topic in enumerate(pending, 1):
        print(f"\n{'═' * 60}\n  [{number}/{len(pending)}]  {topic}\n{'═' * 60}",
              file=sys.stderr)
        # Each topic is a normal single run, flags and all. Reusing `_wiki`
        # rather than reimplementing means the batch cannot drift away from
        # what a single run does.
        run_args = argparse.Namespace(**vars(args))
        run_args.topic = topic
        run_args.wiki = run_args.review = run_args.capture_golden = True
        try:
            code = _wiki(run_args)
        except KeyboardInterrupt:
            print(f"\n  Stopped. {captured} captured this session.", file=sys.stderr)
            return 0
        # A cancelled review (`q`) is a judgement, not a failure: it means the
        # topic was too thin to be worth filing. Move on.
        if code in (0, 3, 5):
            captured += 1
        else:
            failed += 1
            print(f"  [{topic}: exit {code}]", file=sys.stderr)

    print(f"\n  Done. {captured} captured, {failed} failed.", file=sys.stderr)
    print("  Check progress with: python -m news_agent --feedback\n", file=sys.stderr)
    return 0


def _push_dataset() -> int:
    """Mirror fixtures into Langfuse Datasets. No Anthropic spend."""
    from .evals.golden import load_all, push_to_langfuse

    fixtures = load_all()
    if not fixtures:
        print("No fixtures to push.", file=sys.stderr)
        return 1
    pushed = push_to_langfuse(fixtures)
    flush()
    if not pushed:
        print(
            "Langfuse is not configured — set LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY. The fixtures on disk are unaffected.",
            file=sys.stderr,
        )
        return 1
    print(f"Pushed {pushed} fixture(s) to the 'news-golden' dataset.")
    return 0


def _replay(args) -> int:
    """The regression gate: re-judge frozen digests, compare against the human.

    Exit 7 when agreement drops. That is the whole point — a gate that only
    reports is a dashboard, and you already have one of those.
    """
    from .evals.golden import load_all, replay, summarise

    fixtures = load_all()
    if not fixtures:
        print(
            "No fixtures yet. Capture some first:\n"
            '  python -m news_agent "AI" --wiki --review --capture-golden',
            file=sys.stderr,
        )
        return 1

    if args.replay_limit:
        fixtures = fixtures[-args.replay_limit:]

    judge = args.judge_model or "the default judge"
    estimate = 0.01 * len(fixtures)
    print(
        f"Replaying {len(fixtures)} fixture(s) against {judge} — roughly "
        f"${estimate:.2f}. Research is not re-run.",
        file=sys.stderr,
    )

    results = []
    for fixture in fixtures:
        try:
            results.append(replay(fixture, model=args.judge_model))
        except Exception as exc:  # noqa: BLE001 - one bad fixture must not
            # abandon the fixtures already paid for.
            print(f"  {fixture.slug}: {exc}", file=sys.stderr)
    flush()

    print(summarise(results))
    if not results:
        return 1

    before = sum(r["agreement_before"] for r in results) / len(results)
    after = sum(r["agreement_with_human"] for r in results) / len(results)
    if after < before:
        print(
            f"REGRESSION: agreement fell from {before:.0%} to {after:.0%}. "
            "The rubric change made the judge match you less often.",
            file=sys.stderr,
        )
        return 7
    return 0


def _wiki(args) -> int:
    from .agents.judge import JudgeError
    from .agents.research import DigestError
    from .core.budget import BudgetExceeded
    from .review import ReviewAborted, interactive_review

    hook = None
    # The command lives in the CLI's closure rather than on PipelineResult:
    # the orchestrator has no business knowing about review syntax.
    typed = {"command": "", "reviewed": False}
    if args.review:
        def hook(ranked, selected):
            outcome = interactive_review(ranked, selected, floor=args.floor)
            typed["command"] = outcome.command
            typed["reviewed"] = outcome.reviewed
            return outcome.selected, outcome.selected_by

    try:
        result = run_pipeline(
            args.topic,
            model=args.model,
            judge_model=args.judge_model,
            max_iterations=args.max_iterations,
            top_n=args.top,
            floor=args.floor,
            dry_run=args.dry_run,
            window_days=args.window_days,
            force=args.force,
            max_usd=args.max_usd,
            select_hook=hook,
        )
    except ReviewAborted as exc:
        print(str(exc), file=sys.stderr)
        return 0
    except BudgetExceeded as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 6
    except (DigestError, JudgeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 4
    finally:
        flush()

    print(render(result.research.digest))
    print(render_pipeline(result))
    print(f"[{result.cost_line}]", file=sys.stderr)

    if args.capture_golden and not typed["reviewed"]:
        # The failure this prevents: with no terminal the review falls through
        # and `selected` is still the judge's pick — so a captured fixture
        # would record "the human agreed" when no human was present. Twenty of
        # those and --feedback reports 100% acceptance and tells you to drop
        # the review step, on the strength of reviews that never happened.
        print(
            "[not captured: nothing was reviewed, so there is no human verdict "
            "to record]",
            file=sys.stderr,
        )
    elif args.capture_golden:
        from .evals.golden import capture

        path = capture(
            topic=args.topic,
            digest=result.research.digest,
            ranked=result.judged.ranked,
            judge_selected=result.judge_selected,
            human_selected=result.selected,
            command=typed["command"],
        )
        print(f"[golden fixture: {path.name}]", file=sys.stderr)

    if result.vault_result and result.vault_result.conflicts:
        # Exit 5, not 0: nothing was written and the user asked for a write.
        print(
            f"\n{len(result.vault_result.conflicts)} note(s) already exist with "
            "different content and were left alone:",
            file=sys.stderr,
        )
        for name in result.vault_result.conflicts:
            print(f"  - {name}", file=sys.stderr)
        print("Pass --force to overwrite them.", file=sys.stderr)
        return 5

    if not result.research.is_grounded:
        _print_grounding_problems(result.research)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
