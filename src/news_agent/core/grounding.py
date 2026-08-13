"""Does the digest actually say what its sources said?

The original grounding check verified that every cited URL was one the tool
returned. That catches an invented link, and nothing else. A model can cite a
perfectly real article and describe something that is not in it — and that
scored 1.0.

Two cheap checks close most of the gap, using text already in hand:

**Figures.** Fabricated numbers are the classic summarisation failure: a
"$2 billion round" that was $200 million, a "40% increase" nobody reported.
A number in the digest that appears nowhere in the cited articles is a strong,
low-false-positive signal — far stronger than prose similarity, because a
paraphrase changes words but must not change quantities.

**Lexical overlap.** How much of the digest's distinctive vocabulary appears
in its sources. Deliberately lenient: the model is *supposed* to paraphrase,
and `why_it_matters` is analysis that will not appear in any source. This is a
smoke detector for an item spun from nothing, not a plagiarism check.

Neither is proof. Both are free, and free beats absent.
"""

from __future__ import annotations

import re

#: A number, with the magnitude word that gives it meaning. `500 billion` and
#: `500` are different claims, so the unit is part of the token.
_FIGURE = re.compile(
    r"(\d[\d,.]*)\s*(billion|million|trillion|bn|m\b|k\b|%|percent)?",
    re.IGNORECASE,
)

_MAGNITUDE = {
    "bn": "billion", "m": "million", "k": "thousand", "percent": "%",
}

_WORD = re.compile(r"[a-z][a-z'-]{3,}")

#: Words too common to indicate anything. Kept short on purpose — a long
#: stopword list starts encoding assumptions about subject matter.
_STOP = {
    "this", "that", "with", "from", "have", "been", "will", "they", "their",
    "which", "would", "could", "should", "about", "after", "into", "more",
    "than", "when", "what", "were", "also", "over", "such", "some", "other",
    "there", "these", "those", "said", "says", "according", "while", "being",
}


def _normalise_figure(number: str, unit: str | None) -> str:
    digits = number.rstrip(".,").replace(",", "")
    suffix = (unit or "").lower()
    return f"{digits}{_MAGNITUDE.get(suffix, suffix)}"


def figures(text: str) -> set[str]:
    """Every quantity stated in `text`, normalised.

    Bare one-digit numbers are skipped: "3 companies" is not the kind of claim
    worth flagging, and it would fire constantly on ordinary prose.
    """
    found = set()
    for number, unit in _FIGURE.findall(text or ""):
        digits = number.rstrip(".,").replace(",", "")
        if not digits:
            continue
        if len(digits.replace(".", "")) < 2 and not unit:
            continue
        found.add(_normalise_figure(number, unit))
    return found


def unsupported_figures(claim: str, source: str) -> list[str]:
    """Figures asserted in `claim` that do not appear in `source`.

    Matching is on the normalised form *and* on the bare digits, because a
    source may write "500 billion" where the digest writes "$500bn" — same
    claim, different rendering, and flagging it would be a false alarm.
    """
    source_figures = figures(source)
    source_digits = {f.rstrip("abcdefghijklmnopqrstuvwxyz%") for f in source_figures}
    missing = []
    for figure in sorted(figures(claim)):
        digits = figure.rstrip("abcdefghijklmnopqrstuvwxyz%")
        if figure not in source_figures and digits not in source_digits:
            missing.append(figure)
    return missing


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP}


def overlap(claim: str, source: str) -> float:
    """Fraction of the claim's distinctive words that appear in the source.

    1.0 when the claim has nothing distinctive to check — an empty claim is
    not evidence of fabrication, and returning 0.0 would make every degenerate
    case look like an attack.
    """
    claim_words = _content_words(claim)
    if not claim_words:
        return 1.0
    return len(claim_words & _content_words(source)) / len(claim_words)


#: Below this, an item's wording has almost nothing in common with the articles
#: it cites. Set low deliberately: the model paraphrases by design, so this
#: should fire on invention, not on good writing.
MIN_OVERLAP = 0.30
