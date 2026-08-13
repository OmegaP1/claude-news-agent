"""Attribution and the spend ceiling.

Both exist to answer a question after the fact: *why did quality change* and
*why did this cost that much*. Both are useless if they are subtly wrong, and
neither fails loudly when they are — hence tests.
"""

from __future__ import annotations

import pytest

from news_agent.core.budget import BudgetExceeded, check_budget
from news_agent.core.provenance import __version__, prompt_hash, provenance


# --- provenance --------------------------------------------------------------


def test_hash_changes_when_the_prompt_changes():
    assert prompt_hash("be terse") != prompt_hash("be verbose")


def test_hash_is_stable_across_calls():
    """It is a grouping key across runs and processes — a per-process salt
    would make every run look like a new prompt version."""
    assert prompt_hash("same") == prompt_hash("same")


def test_parts_are_separated_so_a_split_cannot_collide():
    """Concatenating without a separator makes ("ab","c") and ("a","bc") hash
    identically — a rubric change could then report as no change at all."""
    assert prompt_hash("ab", "c") != prompt_hash("a", "bc")


def test_judge_weights_are_part_of_its_hash():
    """The composite is computed in code, so re-weighting changes every score
    without changing a single prompt token. Hashing the system prompt alone
    would report 'same prompt' across a change that altered the whole ranking."""
    from news_agent.agents.judge.agent import _judge_prompt_parts
    from news_agent.agents.judge import config

    before = prompt_hash(*_judge_prompt_parts())
    original = dict(config.WEIGHTS)
    try:
        config.WEIGHTS["novelty"] = original["novelty"] + 0.1
        assert prompt_hash(*_judge_prompt_parts()) != before
    finally:
        config.WEIGHTS.clear()
        config.WEIGHTS.update(original)


def test_provenance_carries_version_and_resolved_model():
    payload = provenance(prompt_parts=("p",), model_resolved="claude-sonnet-5-20260401")
    assert payload["app_version"] == __version__
    assert payload["model_resolved"] == "claude-sonnet-5-20260401"
    assert len(payload["prompt_hash"]) == 8


def test_resolved_model_is_omitted_when_unknown():
    """Absent, not None: a key whose value is always None is noise on every
    trace and cannot be filtered on either way."""
    assert "model_resolved" not in provenance(prompt_parts=("p",), model_resolved=None)


# --- budget ------------------------------------------------------------------


def test_no_ceiling_means_no_behaviour_change():
    check_budget(999.0, None, "research")  # must not raise


def test_ceiling_stops_the_run():
    with pytest.raises(BudgetExceeded) as exc:
        check_budget(0.11, 0.10, "research")
    assert exc.value.stage == "research"
    assert "--max-usd" in str(exc.value)  # actionable, not just "exceeded"


def test_a_zero_ceiling_is_honoured_not_treated_as_absent():
    """`if not ceiling` would make --max-usd 0 mean 'unlimited' — the exact
    inversion of what was asked for, and silent."""
    with pytest.raises(BudgetExceeded):
        check_budget(0.0, 0.0, "research")


def test_exactly_at_the_ceiling_stops():
    """A ceiling is a limit, not a target to reach and pass."""
    with pytest.raises(BudgetExceeded):
        check_budget(0.10, 0.10, "research")


# --- one version number, not two ---------------------------------------------


def test_version_comes_from_package_metadata_not_a_second_constant():
    """Regression: `provenance.__version__` was hardcoded while pyproject said
    something else. Two version numbers that can disagree is the duplicate-field
    problem — and the trace would have been the one lying."""
    import inspect
    import tomllib
    from pathlib import Path

    from news_agent.core import provenance as prov

    source = inspect.getsource(prov)
    assert 'version("news-agent")' in source, "must read the installed metadata"

    root = Path(prov.__file__).resolve().parents[3]
    declared = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))
    assert __version__ == declared["project"]["version"], (
        f"provenance reports {__version__}, pyproject declares "
        f"{declared['project']['version']} — reinstall with `pip install -e .`"
    )
