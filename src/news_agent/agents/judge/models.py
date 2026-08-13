"""Models owned by the judge.

``ScoredItem`` lives here rather than in ``core`` on purpose: it composes a
research ``DigestItem`` with a judge ``ItemVerdict``, so putting it in core
would make the shared layer depend on two agents — an inverted dependency.
Here it reads as what it is, the judge's output contract, which downstream
sinks may import.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..research.models import DigestItem


class ItemVerdict(BaseModel):
    """One judged item.

    Field order is load-bearing. The JSON Schema preserves declaration order,
    and the model generates fields in that order — so `reasoning` comes *before*
    the scores, forcing it to deliberate and then commit. Scores first would
    produce post-hoc rationalisation of a number it already guessed.

    Scores are `Literal[1..5]`, not `int` with ge/le: structured outputs ignore
    numeric constraints (min/max are unsupported) but *do* enforce enums, so
    this is the only way to actually bind the scale. A coarse 1-5 is also
    deliberate — finer scales add noise, not signal.
    """

    model_config = ConfigDict(extra="forbid")

    item_index: int = Field(description="The number of the item being judged, as shown.")
    reasoning: str = Field(
        description="One or two sentences justifying the scores below. Write this first."
    )
    significance: Literal[1, 2, 3, 4, 5] = Field(
        description=(
            "How much does this actually matter? 5 = affects many people or sets "
            "precedent. 3 = notable within its field. 1 = trivia or routine "
            "corporate news."
        )
    )
    novelty: Literal[1, 2, 3, 4, 5] = Field(
        description=(
            "Is this a new development? 5 = genuinely new information. 3 = an "
            "update to a running story. 1 = recycled or already widely known."
        )
    )
    relevance: Literal[1, 2, 3, 4, 5] = Field(
        description=(
            "How directly does this address the requested topic? 5 = squarely on "
            "topic. 3 = adjacent. 1 = only loosely connected."
        )
    )
    evidence: Literal[1, 2, 3, 4, 5] = Field(
        description=(
            "How concrete is the reporting? 5 = specific facts, figures, named "
            "actors. 3 = general but sourced. 1 = vague or speculative."
        )
    )


class JudgeVerdicts(BaseModel):
    """The judge's full response — one verdict per item, scored independently."""

    model_config = ConfigDict(extra="forbid")

    verdicts: list[ItemVerdict] = Field(
        description="One entry per item presented, in any order. Score each on its own merits."
    )


class ScoredItem(BaseModel):
    """A digest item plus its verdict and the composite computed in code."""

    item: DigestItem
    verdict: ItemVerdict
    composite: float

    @property
    def scores(self) -> dict[str, int]:
        return {
            "significance": self.verdict.significance,
            "novelty": self.verdict.novelty,
            "relevance": self.verdict.relevance,
            "evidence": self.verdict.evidence,
        }
