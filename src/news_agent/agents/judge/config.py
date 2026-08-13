"""Tuning knobs for the judge.

The weights are a *product* decision, not a model decision — which is exactly
why they live in code you can read and change, rather than being asked of the
model as an overall score.
"""

from __future__ import annotations

#: A different model from the generator, deliberately. Models show
#: self-preference bias: a Haiku judge grading a Haiku digest is marking its
#: own homework and you never see it fail.
MODEL = "claude-sonnet-5"

#: Note: NOT temperature. Sonnet 5 rejects sampling parameters with a 400, so
#: the usual "temperature=0 for judges" advice would break the call outright.
#: Consistency comes from the rubric and the coarse scale instead.
EFFORT = "medium"

MAX_TOKENS = 4_000

#: How the dimensions combine. Must sum to 1.0 or the composite leaves the 1-5
#: scale that MIN_COMPOSITE is expressed in.
WEIGHTS: dict[str, float] = {
    "significance": 0.40,
    "relevance": 0.30,
    "novelty": 0.20,
    "evidence": 0.10,
}

#: Items scoring below this (1-5 scale) never reach the vault, even if they
#: land in the top 3. "Top 3" is a ceiling, not a quota — publishing three
#: items when only one is any good is the padding this project refuses
#: everywhere else.
MIN_COMPOSITE = 2.5
