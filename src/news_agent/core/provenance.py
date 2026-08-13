"""What produced this run — so a quality change can be attributed to a cause.

Without this, a drop in `judge_accepted` is unattributable: you cannot tell a
prompt edit from a model alias moving under you from a code change. Three
fields answer it, and all three are free:

- ``app_version``  — bumped by hand when behaviour changes
- ``prompt_hash``  — sha256 of the system prompt text actually sent
- ``model_resolved`` — what the API says it ran, not what we asked for

**Why not the Langfuse prompt registry.** It exists and works, but it moves
prompts out of the code and into a network call on the hot path. The goal here
is *attribution*, not editing: `prompt_hash` changes exactly when the prompt
changes, which is the whole property needed to group scores by prompt version.
A registry would buy a UI at the cost of a runtime dependency.

**Why not a git SHA.** There is no repository — `git rev-parse` fails. Even if
there were, a SHA changes on every commit including ones that touch no prompt,
so it is a noisier key than the hash of the text itself.

This module is in ``core`` and imports no agent: callers pass the prompt text.
"""

from __future__ import annotations

import hashlib

def _installed_version() -> str:
    """Read the version from package metadata — never restate it here.

    Bump it in `pyproject.toml` when behaviour changes in a way that could move
    quality scores: a new rubric, a different default model, a changed
    selection rule. Deliberately manual, because an auto-incrementing number
    would change on edits that cannot affect output and would be useless as a
    grouping key.

    A hardcoded copy here would be a second version number that can disagree
    with the first — and the trace would be the one lying, which is the worst
    place for it. The fallback only fires when the package is not installed
    (running from a source checkout), and says so rather than guessing.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("news-agent")
    except (ImportError, PackageNotFoundError):
        return "unknown"


__version__ = _installed_version()


def prompt_hash(*parts: str) -> str:
    """Stable short hash of the prompt text actually sent.

    Takes every part that shapes the model's behaviour — system prompt, rubric,
    weights rendered into the prompt — because a rubric change with an unchanged
    system prompt must still produce a different hash.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")  # separator: ("ab","c") must differ from ("a","bc")
    return digest.hexdigest()[:8]


def provenance(*, prompt_parts: tuple[str, ...], model_resolved: str | None) -> dict:
    """The attribution payload attached to a generation.

    `model_resolved` comes from the API response, not from what we asked for:
    aliases like `claude-sonnet-5` are pointers, and the whole point is to catch
    the day one moves.
    """
    payload = {
        "app_version": __version__,
        "prompt_hash": prompt_hash(*prompt_parts),
    }
    if model_resolved:
        payload["model_resolved"] = model_resolved
    return payload
