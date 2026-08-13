"""Batch capture: `--topics-file`.

The expensive failure mode here is not a crash — it is a session that runs
every topic, bills for every one, and captures nothing because the review fell
through. These tests guard the refusals more than the happy path.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

import pytest

from news_agent.__main__ import main, read_topics
from news_agent.agents.judge.models import ItemVerdict, ScoredItem
from news_agent.agents.research.models import DigestItem, NewsDigest
from news_agent.evals import golden


# --- parsing the file --------------------------------------------------------


def test_comments_and_blanks_are_ignored():
    assert read_topics(
        """
        # a comment

        AI safety research
        computer vision   # trailing comment
        """
    ) == ["AI safety research", "computer vision"]


def test_duplicates_are_dropped_preserving_order():
    """A list edited a few times accumulates them, and paying twice for a
    second fixture that measures the same thing is waste."""
    assert read_topics("AI\nrobotics\nai\nAI\nvision") == ["AI", "robotics", "vision"]


def test_an_empty_file_is_not_a_crash():
    assert read_topics("# only comments\n\n") == []


# --- resume ------------------------------------------------------------------


def _fixture_on_disk(tmp_path, topic: str, day: str = "2026-08-13"):
    digest = NewsDigest(
        topic=topic, overview="o",
        items=[DigestItem(headline="h", summary="s", why_it_matters="w",
                          sources=["https://x.test/1"])],
        coverage_note="c",
    )
    fixture = golden.GoldenFixture(
        topic=topic, captured=day, digest=digest,
        judge_ranked=[0], judge_selected=[0], human_selected=[0],
    )
    path = tmp_path / f"{fixture.slug}.json"
    path.write_text(fixture.model_dump_json(indent=2), encoding="utf-8")
    return path


def test_captured_topics_is_keyed_on_topic_not_filename(tmp_path):
    """The filename carries a date. Matching on it would re-run the whole list
    the next morning — the opposite of resuming."""
    _fixture_on_disk(tmp_path, "AI safety research", day="2026-08-01")
    assert golden.captured_topics(tmp_path) == {"ai safety research"}


def test_resume_matching_ignores_case_and_padding(tmp_path):
    _fixture_on_disk(tmp_path, "  AI Safety Research  ")
    assert "ai safety research" in golden.captured_topics(tmp_path)


# --- the refusals ------------------------------------------------------------


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")


def test_no_terminal_refuses_before_spending_anything(tmp_path, monkeypatch, with_key, capsys):
    """The single most expensive way to get nothing: run every topic, bill for
    each, and capture none of them because nobody was there to review."""
    topics = tmp_path / "t.txt"
    topics.write_text("AI safety research\ncomputer vision\n", encoding="utf-8")

    called = []
    monkeypatch.setattr("news_agent.__main__._wiki", lambda a: called.append(a) or 0)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)

    assert main(["--topics-file", str(topics)]) == 2
    assert called == [], "a run was started despite there being no terminal"
    assert "needs a terminal" in capsys.readouterr().err


def test_a_topic_argument_alongside_the_file_is_rejected(tmp_path, with_key):
    topics = tmp_path / "t.txt"
    topics.write_text("AI\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["some topic", "--topics-file", str(topics)])


def test_a_missing_file_fails_cleanly(tmp_path, with_key, capsys):
    assert main(["--topics-file", str(tmp_path / "nope.txt")]) == 1
    assert "Cannot read" in capsys.readouterr().err


def test_no_key_is_caught_before_reading_anything(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["--topics-file", str(tmp_path / "whatever.txt")]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


# --- the loop ----------------------------------------------------------------


@pytest.fixture
def tty(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)


def test_each_topic_runs_with_the_capture_flags_forced(tmp_path, monkeypatch, with_key, tty):
    """Reusing `_wiki` rather than reimplementing it means the batch cannot
    drift away from what a single run does."""
    topics = tmp_path / "t.txt"
    topics.write_text("alpha\nbeta\n", encoding="utf-8")
    monkeypatch.setattr(golden, "captured_topics", lambda *a, **k: set())

    seen = []

    def fake_wiki(a):
        seen.append((a.topic, a.wiki, a.review, a.capture_golden))
        return 0

    monkeypatch.setattr("news_agent.__main__._wiki", fake_wiki)

    assert main(["--topics-file", str(topics)]) == 0
    assert seen == [("alpha", True, True, True), ("beta", True, True, True)]


def test_already_captured_topics_are_skipped(tmp_path, monkeypatch, with_key, tty, capsys):
    topics = tmp_path / "t.txt"
    topics.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    monkeypatch.setattr(
        "news_agent.evals.golden.captured_topics", lambda *a, **k: {"beta"}
    )

    seen = []
    monkeypatch.setattr("news_agent.__main__._wiki",
                        lambda a: seen.append(a.topic) or 0)

    main(["--topics-file", str(topics)])
    assert seen == ["alpha", "gamma"]
    assert "1 already captured" in capsys.readouterr().err


def test_a_failing_topic_does_not_abandon_the_rest(tmp_path, monkeypatch, with_key, tty):
    """One thin topic must not waste the runs after it."""
    topics = tmp_path / "t.txt"
    topics.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    monkeypatch.setattr(
        "news_agent.evals.golden.captured_topics", lambda *a, **k: set()
    )

    seen = []

    def fake_wiki(a):
        seen.append(a.topic)
        return 1 if a.topic == "beta" else 0

    monkeypatch.setattr("news_agent.__main__._wiki", fake_wiki)
    assert main(["--topics-file", str(topics)]) == 0
    assert seen == ["alpha", "beta", "gamma"]


def test_ctrl_c_stops_the_batch_and_reports_progress(tmp_path, monkeypatch, with_key, tty, capsys):
    topics = tmp_path / "t.txt"
    topics.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    monkeypatch.setattr(
        "news_agent.evals.golden.captured_topics", lambda *a, **k: set()
    )

    seen = []

    def fake_wiki(a):
        seen.append(a.topic)
        if a.topic == "beta":
            raise KeyboardInterrupt
        return 0

    monkeypatch.setattr("news_agent.__main__._wiki", fake_wiki)
    assert main(["--topics-file", str(topics)]) == 0
    assert seen == ["alpha", "beta"]          # gamma never started
    assert "1 captured this session" in capsys.readouterr().err


def test_the_shipped_topics_file_parses_and_is_varied():
    """A dataset of twenty near-identical topics measures the judge on one kind
    of story and says nothing about the rest."""
    import pathlib

    import news_agent

    root = pathlib.Path(news_agent.__file__).resolve().parents[2]
    topics = read_topics((root / "topics.txt").read_text(encoding="utf-8"))
    # Enough to clear MIN_SAMPLE with room for topics that come back empty,
    # and no duplicates — paying twice for the same topic buys a second
    # fixture that measures the same thing.
    assert len(topics) >= 20
    assert len(set(t.casefold() for t in topics)) == len(topics)
