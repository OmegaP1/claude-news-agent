"""A hard spend ceiling that aborts, rather than a dashboard that reports.

`--max-iterations` already bounds the research loop, but it bounds *rounds*,
not *dollars* — and it does nothing about the judge, which runs on a pricier
model after research has already been paid for.

The check sits between stages because that is where an abort still saves money:
once a request is in flight the tokens are billed regardless. Aborting after
research and before the judge is the one decision point that can actually
prevent spend.

This is the general form of the `MIN_ITERATIONS` lesson: that bug burned money
and returned nothing, and a ceiling is what turns "I hope this is cheap" into
a guarantee.
"""

from __future__ import annotations


class BudgetExceeded(RuntimeError):
    """Spend hit the ceiling; the run stopped deliberately.

    Carries the numbers so the CLI can print something actionable rather than
    just "budget exceeded".
    """

    def __init__(self, spent_usd: float, ceiling_usd: float, stage: str) -> None:
        self.spent_usd = spent_usd
        self.ceiling_usd = ceiling_usd
        self.stage = stage
        super().__init__(
            f"Stopped after {stage}: spent ${spent_usd:.4f} of a "
            f"${ceiling_usd:.4f} ceiling. Raise it with --max-usd, or lower "
            f"--max-iterations to make research cheaper."
        )


def check_budget(spent_usd: float, ceiling_usd: float | None, stage: str) -> None:
    """Raise if `spent_usd` has reached the ceiling.

    `None` means no ceiling — the default, so behaviour is unchanged unless
    someone opts in. A ceiling of 0 is honoured as "spend nothing", not treated
    as absent, because `0 or None` conflating those is exactly the kind of
    falsy-value bug that makes a guard silently stop guarding.
    """
    if ceiling_usd is None:
        return
    if spent_usd >= ceiling_usd:
        raise BudgetExceeded(spent_usd, ceiling_usd, stage)
