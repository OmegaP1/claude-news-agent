"""Human-in-the-loop: let a person override the judge before anything is filed.

Sits between judging and persistence. The judge ranks; the human decides. That
ordering matters — reviewing a *ranked, scored, justified* list is a much
cheaper cognitive task than reading raw items, so the LLM does the sifting and
the person does the deciding.

The orchestrator never imports this module's interactive parts directly: it
takes a `select_hook` callable instead, so the pipeline stays pure and
testable and the terminal I/O lives here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TextIO

from .agents.judge import ScoredItem


class ReviewAborted(RuntimeError):
    """The human declined to file anything."""


@dataclass
class ReviewOutcome:
    selected: list[ScoredItem]
    #: True when the human changed the judge's pick — recorded in the note's
    #: frontmatter so the vault says who chose, not just what was chosen.
    overridden: bool
    command: str = ""
    #: Whether a person actually looked. False when there was no terminal, so
    #: the selection fell through untouched.
    #:
    #: This is **not** the same as `overridden is False`. "A human read the
    #: ranking and agreed" and "nobody was there" produce an identical
    #: selection and mean opposite things — and treating the second as the
    #: first fabricates ground truth. See `--capture-golden`.
    reviewed: bool = True

    @property
    def selected_by(self) -> str:
        return "human" if self.overridden else "judge"


def parse_command(
    command: str, ranked: list[ScoredItem], current: list[ScoredItem]
) -> list[ScoredItem]:
    """Turn a review command into a selection.

    Accepts either form, because both are natural:

    - ``2 4 5``   — an explicit set, replacing the current pick
    - ``-2 +4``   — deltas against the current pick
    - ``""``      — accept as-is

    Raises ValueError on an out-of-range or unparseable index rather than
    guessing, since guessing here silently files the wrong story.
    """
    tokens = command.replace(",", " ").split()
    if not tokens:
        return list(current)

    # Work in 0-based indices throughout: ScoredItem is a Pydantic model and
    # therefore unhashable, so sets and dict.fromkeys are not available for
    # de-duplication — and identity is what we actually mean anyway.
    def resolve(raw: str) -> int:
        if not raw.lstrip("+-").isdigit():
            raise ValueError(f"{raw!r} is not an item number")
        number = int(raw.lstrip("+-"))
        if not 1 <= number <= len(ranked):
            raise ValueError(f"item {number} does not exist (1-{len(ranked)})")
        return number - 1

    current_indices = [i for i, item in enumerate(ranked) if any(item is c for c in current)]

    if any(t[0] in "+-" for t in tokens):
        if not all(t[0] in "+-" for t in tokens):
            raise ValueError("mix of deltas and plain numbers — use one or the other")
        chosen = list(current_indices)
        for token in tokens:
            index = resolve(token)
            if token[0] == "-":
                chosen = [i for i in chosen if i != index]
            elif index not in chosen:
                chosen.append(index)
    else:
        chosen = []
        for token in tokens:
            index = resolve(token)
            if index not in chosen:
                chosen.append(index)

    return [ranked[i] for i in chosen]


def render_for_review(
    ranked: list[ScoredItem], selected: list[ScoredItem], floor: float
) -> str:
    lines = ["", "  REVIEW — the judge picked the ticked items", "  " + "─" * 42, ""]
    for number, scored in enumerate(ranked, 1):
        mark = "✓" if scored in selected else " "
        below = "  (below floor)" if scored.composite < floor else ""
        lines.append(f"  [{number}] {mark} {scored.composite:>5.2f}{below}  {scored.item.headline}")
        lines.append(f"          {scored.verdict.reasoning}")
        lines.append("")
    return "\n".join(lines)


def interactive_review(
    ranked: list[ScoredItem],
    selected: list[ScoredItem],
    *,
    floor: float,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> ReviewOutcome:
    """Prompt for edits. Returns the final selection.

    Falls through untouched when there is no terminal — a cron job or a piped
    run must not block forever waiting for input nobody will type.
    """
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stderr

    if not ranked:
        return ReviewOutcome(selected=[], overridden=False, reviewed=False)

    if not getattr(stdin, "isatty", lambda: False)():
        print(
            "[no terminal — keeping the judge's selection unreviewed]", file=stdout
        )
        return ReviewOutcome(selected=list(selected), overridden=False, reviewed=False)

    print(render_for_review(ranked, selected, floor), file=stdout)
    print(
        "  Enter to accept · '2 4' to pick exactly those · '-2 +4' to adjust · 'q' to cancel",
        file=stdout,
    )

    original = list(selected)
    while True:
        print("  > ", end="", file=stdout, flush=True)
        raw = stdin.readline()
        if not raw:  # EOF — treat as accept rather than looping forever
            return ReviewOutcome(selected=original, overridden=False, reviewed=False)

        command = raw.strip()
        if command.lower() in {"q", "quit", "n", "no"}:
            raise ReviewAborted("Review cancelled — nothing was written.")

        try:
            chosen = parse_command(command, ranked, original)
        except ValueError as exc:
            print(f"  {exc}. Try again.", file=stdout)
            continue

        return ReviewOutcome(
            selected=chosen,
            overridden=[id(c) for c in chosen] != [id(c) for c in original],
            command=command,
        )
