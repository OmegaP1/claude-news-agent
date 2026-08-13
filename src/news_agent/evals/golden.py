"""Golden fixtures: what a human actually picked, frozen for replay.

The problem this solves: 200-odd tests cover the *code* and nothing covers the
*prompts*. Editing the judge rubric currently ships blind — you would discover
a regression by vaguely noticing the digests felt worse, weeks later.

Every `--wiki --review` run already produces ground truth as a by-product: a
digest, the judge's ranking, and a human's verdict on it. That was being thrown
away. `capture()` keeps it.

Two things then become possible:

1. **Regression testing.** Change the rubric, replay against the fixtures,
   compare agreement with the human before shipping.
2. **Answering "is the judge worth its cost?"** — `acceptance_rate` over the
   fixtures, which is the question `judge_accepted` was collecting data for
   without anyone ever reading it.

**Why local JSON and not Langfuse Datasets.** The Langfuse dataset API is real
and works. But the test suite runs offline in under a second and `conftest.py`
deliberately disables tracing, so ground truth living only behind a network
call could not be replayed in pytest — which is the one place it needs to run.
Files are the source of truth; pushing them to Langfuse for the UI is a
separate, optional step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field

from ..agents.judge.models import ScoredItem
from ..agents.research.models import NewsDigest

#: Ground truth lives with the tests, because it is test data.
DEFAULT_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "golden"

#: Thresholds for "is the judge worth its cost?". Written down so the metric
#: has a decision attached — an unread metric is one you pay to collect and
#: never act on.
KEEP_REVIEWING_BELOW = 0.9
RUBRIC_BROKEN_BELOW = 0.6
#: Below this many fixtures, any rate is noise. Three overrides out of four
#: runs is not evidence the rubric is broken.
MIN_SAMPLE = 20


class GoldenFixture(BaseModel):
    """One captured run: the digest, what the judge picked, what the human did.

    Stores **indices** into `digest.items`, not copies of the items. A copy
    could drift out of sync with the digest it came from and there would be no
    way to tell which was right.
    """

    topic: str
    captured: str
    digest: NewsDigest
    #: 0-based indices into `digest.items`, in the judge's ranked order.
    judge_ranked: list[int]
    #: 0-based indices the judge would have filed, before any human input.
    judge_selected: list[int]
    #: What was actually filed. Equal to `judge_selected` when the human agreed.
    human_selected: list[int]
    #: The review command typed, e.g. "-2 +4". Kept because it says *how* the
    #: human disagreed, which the indices alone do not.
    command: str = ""

    @property
    def overridden(self) -> bool:
        return self.judge_selected != self.human_selected

    @property
    def only_added(self) -> bool:
        """The human kept every judge pick and added more.

        This is **not** a ranking failure. It means the quality floor cut
        something worth filing — a different disagreement with a different fix
        (one constant in `agents/judge/config.py`, not a rewritten rubric).

        Measured over the first 16 fixtures: 4 of the 5 overrides were of this
        kind, and every one of them took the selection to exactly `top_n`. The
        judge's *ranking* was accepted 94% of the time while the headline
        acceptance rate read 69%.
        """
        judge, human = set(self.judge_selected), set(self.human_selected)
        return bool(judge < human)

    @property
    def swapped(self) -> bool:
        """The human removed something the judge picked — a real disagreement
        about ranking, and the only kind a rubric change can fix."""
        return bool(set(self.judge_selected) - set(self.human_selected))

    @property
    def slug(self) -> str:
        safe = "".join(c if c.isalnum() else "-" for c in self.topic.lower())
        return f"{self.captured}-{safe.strip('-')[:40]}"


@dataclass(frozen=True)
class Verdict:
    """The decision the rates imply.

    Two rates, because there are two ways to disagree with the judge and they
    are fixed in different files.
    """

    rate: float
    sample: int
    decision: str
    detail: str
    #: Fraction where the human kept every judge pick — ranking quality alone.
    ranking: float = 0.0
    #: Fraction where the human added back an item the floor had dropped.
    floor: float = 0.0


def _indices(items, selection: list[ScoredItem]) -> list[int]:
    """Map ScoredItems back to positions in the original digest.

    By identity, not equality: two items could in principle carry the same
    headline, and `ScoredItem` is a Pydantic model whose `==` is structural.
    """
    positions = []
    for scored in selection:
        for index, item in enumerate(items):
            if item is scored.item:
                positions.append(index)
                break
    return positions


def capture(
    *,
    topic: str,
    digest: NewsDigest,
    ranked: list[ScoredItem],
    judge_selected: list[ScoredItem],
    human_selected: list[ScoredItem],
    command: str = "",
    day: date | None = None,
    directory: Path | None = None,
) -> Path:
    """Freeze one reviewed run as a fixture. Returns the path written."""
    fixture = GoldenFixture(
        topic=topic,
        captured=(day or date.today()).isoformat(),
        digest=digest,
        judge_ranked=_indices(digest.items, ranked),
        judge_selected=_indices(digest.items, judge_selected),
        human_selected=_indices(digest.items, human_selected),
        command=command,
    )
    target = (directory or DEFAULT_DIR) / f"{fixture.slug}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(fixture.model_dump_json(indent=2), encoding="utf-8")
    return target


def load_all(directory: Path | None = None) -> list[GoldenFixture]:
    """Every fixture on disk, oldest first. Missing directory is not an error —
    it just means nothing has been captured yet."""
    root = directory or DEFAULT_DIR
    if not root.exists():
        return []
    fixtures = []
    for path in sorted(root.glob("*.json")):
        fixtures.append(GoldenFixture.model_validate_json(path.read_text("utf-8")))
    return fixtures


def captured_topics(directory: Path | None = None) -> set[str]:
    """Topics that already have a fixture, case-folded.

    This is what makes a batch resumable. Keyed on the *topic* rather than the
    filename, because the filename carries a date — matching on it would
    re-run everything the next morning, which is the opposite of resuming.
    """
    return {f.topic.strip().casefold() for f in load_all(directory)}


def agreement(judge_selected: list[int], human_selected: list[int]) -> float:
    """Overlap between the two selections, 0.0-1.0.

    Set overlap rather than rank correlation, because the decision this system
    makes is "which items get filed" — a reordering within the chosen three
    changes nothing downstream, so a metric that punished it would be measuring
    something nobody acts on.
    """
    if not human_selected:
        return 1.0 if not judge_selected else 0.0
    hits = len(set(judge_selected) & set(human_selected))
    return hits / len(human_selected)


def acceptance_rate(fixtures: list[GoldenFixture]) -> float:
    """Fraction of reviewed runs where the human accepted the judge unchanged."""
    if not fixtures:
        return 0.0
    return sum(1 for f in fixtures if not f.overridden) / len(fixtures)


def ranking_rate(fixtures: list[GoldenFixture]) -> float:
    """Fraction where the human kept every item the judge picked.

    The distinction from `acceptance_rate` is the whole point: a run where you
    added a fourth story is one where the judge ranked correctly and the
    *floor* was wrong. Counting that as a judge failure sends you to rewrite a
    rubric that is working, which is worse than having no metric.
    """
    if not fixtures:
        return 0.0
    return sum(1 for f in fixtures if not f.swapped) / len(fixtures)


def floor_rate(fixtures: list[GoldenFixture]) -> float:
    """Fraction where the human had to add back something the floor dropped."""
    if not fixtures:
        return 0.0
    return sum(1 for f in fixtures if f.only_added) / len(fixtures)


def verdict_for(fixtures: list[GoldenFixture]) -> Verdict:
    """Turn the acceptance rate into the decision it implies.

    The thresholds are written down rather than left to judgement so the answer
    does not depend on the mood of whoever reads the dashboard.
    """
    rate = acceptance_rate(fixtures)
    ranking = ranking_rate(fixtures)
    floor = floor_rate(fixtures)
    sample = len(fixtures)

    def verdict(decision: str, detail: str) -> Verdict:
        return Verdict(rate, sample, decision, detail, ranking=ranking, floor=floor)

    if sample < MIN_SAMPLE:
        return verdict(
            "insufficient-data",
            f"{sample} of {MIN_SAMPLE} runs needed before the rates mean "
            f"anything. Keep using --review --capture-golden.",
        )

    # Checked before the rubric verdicts, and this ordering is the fix for a
    # real defect: a judge that ranks well while the floor cuts too much reads
    # as a low acceptance rate, and the old code sent you to rewrite a rubric
    # that was working.
    if ranking >= KEEP_REVIEWING_BELOW and floor > 1 - KEEP_REVIEWING_BELOW:
        return verdict(
            "lower-the-floor",
            f"The judge's ranking held {ranking:.0%} of the time, but you added "
            f"back a dropped item in {floor:.0%} of runs. The rubric is fine — "
            f"MIN_COMPOSITE is too high. Lower it, or raise --floor per run.",
        )
    if rate >= KEEP_REVIEWING_BELOW:
        return verdict(
            "drop-review",
            f"The judge matched you {rate:.0%} of the time over {sample} runs. "
            f"Review is now ceremony — drop --review and save the interruption.",
        )
    if ranking < RUBRIC_BROKEN_BELOW:
        return verdict(
            "fix-rubric",
            f"You removed one of the judge's picks in {1 - ranking:.0%} of "
            f"{sample} runs. That is a ranking failure, not a threshold one — "
            f"the rubric in agents/judge/config.py does not match what you "
            f"want. Edit it, then --replay.",
        )
    return verdict(
        "keep-reviewing",
        f"Ranking held {ranking:.0%}, exact match {rate:.0%}, over {sample} "
        f"runs. Useful but not trustworthy alone — keep --review.",
    )


def replay(fixture: GoldenFixture, *, client=None, model: str | None = None) -> dict:
    """Re-judge a frozen digest and score the result against the human.

    **This spends money** — one judge call per fixture, roughly a cent. It is
    not called by the test suite; it is what you run deliberately after editing
    the rubric. Research is not re-run, which is the entire point: the frozen
    digest removes the expensive, non-deterministic half.
    """
    from ..agents.judge import judge_digest

    result = judge_digest(
        fixture.digest, client=client, **({"model": model} if model else {})
    )
    top_n = len(fixture.judge_selected) or 3
    now_selected = _indices(fixture.digest.items, result.ranked[:top_n])
    return {
        "topic": fixture.topic,
        "captured": fixture.captured,
        "agreement_with_human": agreement(now_selected, fixture.human_selected),
        "agreement_before": agreement(fixture.judge_selected, fixture.human_selected),
        "selected_now": now_selected,
        "human_selected": fixture.human_selected,
        "usage": result.usage.model_dump(),
    }


def summarise(results: list[dict]) -> str:
    """Human-readable replay report."""
    if not results:
        return "No fixtures to replay. Capture some with --wiki --review --capture-golden."
    before = sum(r["agreement_before"] for r in results) / len(results)
    after = sum(r["agreement_with_human"] for r in results) / len(results)
    arrow = "improved" if after > before else "regressed" if after < before else "unchanged"
    lines = [
        "",
        f"  REPLAY — {len(results)} fixture(s)",
        "  " + "─" * 30,
        "",
        f"  agreement with you, as captured : {before:.0%}",
        f"  agreement with you, now         : {after:.0%}   ({arrow})",
        "",
    ]
    for r in results:
        lines.append(
            f"  {r['agreement_with_human']:>4.0%}  {r['captured']}  {r['topic']}"
        )
    lines.append("")
    return "\n".join(lines)


def push_to_langfuse(fixtures: list[GoldenFixture], dataset_name: str = "news-golden") -> int:
    """Optional mirror into Langfuse Datasets for the UI. Returns items pushed.

    Deliberately one-way and opt-in. The files stay authoritative — this exists
    so the dataset is *visible*, not so it becomes another source of truth that
    can disagree with the first one.
    """
    from ..core.observability import _client

    if _client is None:
        return 0
    _client.create_dataset(name=dataset_name)
    for fixture in fixtures:
        _client.create_dataset_item(
            dataset_name=dataset_name,
            input=json.loads(fixture.digest.model_dump_json()),
            expected_output={"selected": fixture.human_selected},
            metadata={"topic": fixture.topic, "captured": fixture.captured},
            id=fixture.slug,
        )
    return len(fixtures)
