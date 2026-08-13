"""Vault writer tests — including the attacks, since every filename component
here comes from model output."""

from __future__ import annotations

from datetime import date

import pytest

from news_agent.sinks import obsidian as vault
from news_agent.agents.judge import composite
from news_agent.agents.judge.models import ItemVerdict, ScoredItem
from news_agent.agents.research.models import DigestItem

DAY = date(2026, 8, 13)


def scored(headline: str, sig=5, nov=4, rel=5, ev=4) -> ScoredItem:
    v = ItemVerdict(
        item_index=1,
        reasoning="Solid, specific, on topic.",
        significance=sig,
        novelty=nov,
        relevance=rel,
        evidence=ev,
    )
    return ScoredItem(
        item=DigestItem(
            headline=headline,
            summary="What happened, briefly.",
            why_it_matters="It sets precedent.",
            sources=["https://e.test/a", "https://e.test/b"],
        ),
        verdict=v,
        composite=composite(v),
    )


@pytest.fixture
def fake_vault(tmp_path):
    (tmp_path / ".obsidian").mkdir()
    return tmp_path


# --- filename safety ---------------------------------------------------------


def test_windows_illegal_characters_are_stripped():
    assert vault._slug('a<b>c:d"e/f\\g|h?i*j') == "abcdefghij"


def test_path_traversal_in_a_headline_cannot_escape(fake_vault):
    """A headline is model output. `../../.ssh/authorized_keys` must land in
    the News folder as a file, not overwrite something outside the vault."""
    result = vault.write_digest(
        "topic", "o", [scored("../../../.ssh/authorized_keys")],
        vault=fake_vault, day=DAY,
    )
    written = result.item_notes[0].resolve()
    assert written.is_relative_to(fake_vault.resolve())
    assert "News" in written.parts
    assert not (fake_vault.parent / ".ssh").exists()


def test_windows_reserved_names_are_escaped():
    assert vault._slug("CON") == "_CON"
    assert vault._slug("nul.txt") == "_nul.txt"


def test_slug_is_bounded():
    assert len(vault._slug("x" * 500)) <= 60


def test_empty_headline_still_produces_a_filename():
    assert vault._slug("   ") == "untitled"
    assert vault._slug("...") == "untitled"


# --- frontmatter -------------------------------------------------------------


def test_colon_in_headline_does_not_break_frontmatter(fake_vault):
    """Unquoted YAML would silently fail to parse in Obsidian."""
    result = vault.write_digest(
        "topic", "o", [scored("Breaking: the thing happened")], vault=fake_vault, day=DAY
    )
    text = result.item_notes[0].read_text(encoding="utf-8")
    assert 'title: "Breaking: the thing happened"' in text


def test_quotes_in_headline_are_escaped(fake_vault):
    result = vault.write_digest(
        "topic", "o", [scored('He said "no"')], vault=fake_vault, day=DAY
    )
    text = result.item_notes[0].read_text(encoding="utf-8")
    assert r'\"no\"' in text


def test_note_carries_scores_and_sources(fake_vault):
    result = vault.write_digest(
        "AI regulation", "o", [scored("Story")], vault=fake_vault, day=DAY,
        models={"generator": "claude-haiku-4-5", "judge": "claude-sonnet-5"},
    )
    text = result.item_notes[0].read_text(encoding="utf-8")
    assert "significance: 5" in text
    assert "https://e.test/a" in text
    assert 'generator_model: "claude-haiku-4-5"' in text
    assert 'judge_model: "claude-sonnet-5"' in text
    assert "Solid, specific, on topic." in text


def test_wikilinks_connect_topic_and_date(fake_vault):
    """This is what makes it a wiki rather than a folder of files."""
    result = vault.write_digest(
        "AI regulation", "o", [scored("Story")], vault=fake_vault, day=DAY
    )
    text = result.item_notes[0].read_text(encoding="utf-8")
    assert "[[AI regulation]]" in text
    assert "[[2026-08-13]]" in text


# --- layout & idempotency ----------------------------------------------------


def test_writes_items_and_an_index(fake_vault):
    result = vault.write_digest(
        "AI regulation", "Overview text.", [scored("A"), scored("B")],
        vault=fake_vault, day=DAY,
    )
    assert len(result.item_notes) == 2
    assert result.index_note.exists()
    assert result.index_note.parent.name == "Digests"
    assert result.item_notes[0].parent.name == "Items"

    index = result.index_note.read_text(encoding="utf-8")
    assert "Overview text." in index
    assert "[[2026-08-13 A]]" in index  # index links to the item notes


def test_rerunning_the_same_day_overwrites_rather_than_duplicating(fake_vault):
    for _ in range(3):
        vault.write_digest("topic", "o", [scored("Same story")], vault=fake_vault, day=DAY)
    items = list((fake_vault / "News" / "Items").glob("*.md"))
    assert len(items) == 1


def test_missing_vault_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="OBSIDIAN_VAULT"):
        vault.write_digest("t", "o", [scored("A")], vault=tmp_path / "nope", day=DAY)


def test_vault_path_is_env_overridable(monkeypatch, tmp_path):
    monkeypatch.setenv("OBSIDIAN_VAULT", str(tmp_path))
    assert vault.vault_path() == tmp_path
