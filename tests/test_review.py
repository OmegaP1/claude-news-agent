"""Human-in-the-loop review. All offline — stdin/stdout are injected."""

from __future__ import annotations

import io

import pytest

from news_agent.agents.judge import composite
from news_agent.agents.judge.models import ItemVerdict, ScoredItem
from news_agent.agents.research.models import DigestItem
from news_agent.review import (
    ReviewAborted,
    interactive_review,
    parse_command,
)


def scored(headline: str, score: int = 4) -> ScoredItem:
    v = ItemVerdict(
        item_index=1, reasoning=f"r-{headline}",
        significance=score, novelty=score, relevance=score, evidence=score,
    )
    return ScoredItem(
        item=DigestItem(
            headline=headline, summary="s", why_it_matters="w",
            sources=["https://e.test/a"],
        ),
        verdict=v,
        composite=composite(v),
    )


RANKED = [scored(h) for h in ("A", "B", "C", "D", "E")]
SELECTED = RANKED[:3]          # judge picked A, B, C


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


def names(items):
    return [i.item.headline for i in items]


# --- command parsing ---------------------------------------------------------


def test_empty_command_accepts_the_judge():
    assert parse_command("", RANKED, SELECTED) == SELECTED


def test_drop_one_and_add_another():
    """The exact ask: 'tira a notícia 2 e mete a 4'."""
    result = parse_command("-2 +4", RANKED, SELECTED)
    assert names(result) == ["A", "C", "D"]


def test_explicit_set_replaces_the_selection():
    assert names(parse_command("2 4 5", RANKED, SELECTED)) == ["B", "D", "E"]


def test_commas_are_accepted():
    assert names(parse_command("1, 4", RANKED, SELECTED)) == ["A", "D"]


def test_adding_something_already_selected_is_a_no_op():
    assert names(parse_command("+1", RANKED, SELECTED)) == ["A", "B", "C"]


def test_dropping_something_not_selected_is_a_no_op():
    assert names(parse_command("-5", RANKED, SELECTED)) == ["A", "B", "C"]


def test_duplicates_in_an_explicit_set_are_collapsed():
    assert names(parse_command("1 1 2", RANKED, SELECTED)) == ["A", "B"]


def test_can_promote_an_item_the_cutoff_excluded():
    """The whole point of review: the judge only offers the top N, but the
    human sees the full ranking and can pull something up."""
    assert names(parse_command("+5", RANKED, SELECTED)) == ["A", "B", "C", "E"]


def test_can_select_nothing():
    assert parse_command("-1 -2 -3", RANKED, SELECTED) == []


# --- refusing to guess -------------------------------------------------------


def test_out_of_range_index_raises():
    """Guessing here silently files the wrong story."""
    with pytest.raises(ValueError, match="does not exist"):
        parse_command("9", RANKED, SELECTED)


def test_zero_is_rejected():
    with pytest.raises(ValueError, match="does not exist"):
        parse_command("0", RANKED, SELECTED)


def test_non_numeric_raises():
    with pytest.raises(ValueError, match="not an item number"):
        parse_command("banana", RANKED, SELECTED)


def test_mixing_deltas_and_plain_numbers_raises():
    """'-2 4' is ambiguous: replace with 4, or drop 2 and add 4?"""
    with pytest.raises(ValueError, match="mix of deltas"):
        parse_command("-2 4", RANKED, SELECTED)


# --- the interactive loop ----------------------------------------------------


def test_enter_accepts_and_records_the_judge_as_chooser():
    out = interactive_review(
        RANKED, SELECTED, floor=2.5, stdin=FakeTTY("\n"), stdout=io.StringIO()
    )
    assert out.selected == SELECTED
    assert out.overridden is False
    assert out.selected_by == "judge"


def test_override_is_recorded_as_human():
    out = interactive_review(
        RANKED, SELECTED, floor=2.5, stdin=FakeTTY("-2 +4\n"), stdout=io.StringIO()
    )
    assert names(out.selected) == ["A", "C", "D"]
    assert out.overridden is True
    assert out.selected_by == "human"


def test_reselecting_the_same_items_is_not_an_override():
    out = interactive_review(
        RANKED, SELECTED, floor=2.5, stdin=FakeTTY("1 2 3\n"), stdout=io.StringIO()
    )
    assert out.overridden is False


def test_bad_input_reprompts_instead_of_crashing():
    stdout = io.StringIO()
    out = interactive_review(
        RANKED, SELECTED, floor=2.5, stdin=FakeTTY("nope\n-2 +4\n"), stdout=stdout
    )
    assert names(out.selected) == ["A", "C", "D"]
    assert "not an item number" in stdout.getvalue()


def test_q_aborts_the_whole_write():
    with pytest.raises(ReviewAborted):
        interactive_review(
            RANKED, SELECTED, floor=2.5, stdin=FakeTTY("q\n"), stdout=io.StringIO()
        )


def test_no_tty_falls_through_without_blocking():
    """A cron run must not hang forever waiting for input nobody will type."""
    stdout = io.StringIO()
    out = interactive_review(
        RANKED, SELECTED, floor=2.5, stdin=io.StringIO(""), stdout=stdout
    )
    assert out.selected == SELECTED
    assert out.overridden is False
    assert "no terminal" in stdout.getvalue()


def test_eof_is_treated_as_accept():
    out = interactive_review(
        RANKED, SELECTED, floor=2.5, stdin=FakeTTY(""), stdout=io.StringIO()
    )
    assert out.selected == SELECTED
    assert out.overridden is False


def test_review_shows_the_full_ranking_not_just_the_selection():
    stdout = io.StringIO()
    interactive_review(
        RANKED, SELECTED, floor=2.5, stdin=FakeTTY("\n"), stdout=stdout
    )
    text = stdout.getvalue()
    for headline in ("A", "B", "C", "D", "E"):
        assert headline in text
    assert "[5]" in text  # unselected items are numbered and reachable


# --- "nobody reviewed" is not "the human agreed" ------------------------------


def test_no_terminal_is_marked_unreviewed():
    """The distinction that matters for ground truth: a fall-through and a
    genuine acceptance produce an identical selection and mean opposite
    things. Without `reviewed`, capturing the first as the second fabricates
    a human verdict nobody gave."""
    outcome = interactive_review(
        RANKED, SELECTED, floor=2.5, stdin=io.StringIO(""), stdout=io.StringIO()
    )
    assert outcome.reviewed is False
    assert outcome.overridden is False
    assert outcome.selected == SELECTED


def test_an_actual_acceptance_is_marked_reviewed():
    """Enter on a real terminal means a person read the ranking and agreed.
    That IS a label, and must be distinguishable from nobody being there."""
    outcome = interactive_review(
        RANKED, SELECTED, floor=2.5, stdin=FakeTTY("\n"), stdout=io.StringIO()
    )
    assert outcome.reviewed is True
    assert outcome.overridden is False


def test_eof_mid_review_is_not_a_verdict():
    outcome = interactive_review(
        RANKED, SELECTED, floor=2.5, stdin=FakeTTY(""), stdout=io.StringIO()
    )
    assert outcome.reviewed is False


def test_an_override_is_reviewed_and_records_the_command():
    outcome = interactive_review(
        RANKED, SELECTED, floor=2.5, stdin=FakeTTY("-2 +4\n"), stdout=io.StringIO()
    )
    assert outcome.reviewed is True
    assert outcome.overridden is True
    assert outcome.command == "-2 +4"
