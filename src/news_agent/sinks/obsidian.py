"""Write the top-ranked news into an Obsidian vault as a small wiki.

Layout — one note per item, plus a per-run index that links to them:

    <vault>/News/Items/2026-08-13 Amazon uses Twitch content.md
    <vault>/News/Digests/2026-08-13 AI regulation.md

The item notes carry YAML frontmatter (scores, sources, models) and
`[[wikilinks]]` to the topic and the date, so Obsidian's graph connects runs
over time and a vector plugin like Smart Connections has clean text to index.

Security note: every filename component here derives from **model output**.
`_slug` strips the characters Windows forbids, and `_target` re-checks that the
resolved path is still inside the vault before anything is written. A headline
of `../../.ssh/authorized_keys` must land in the News folder as a file, not
escape the vault.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from ..agents.judge.models import ScoredItem
from ..core.observability import observe
from ..core.telemetry import tool_call

#: Relative to the home directory, never an absolute path with a username in
#: it. A hardcoded personal path is both wrong for everyone else and a small
#: privacy leak in a public repository.
DEFAULT_VAULT = Path.home() / "Obsidian"
NEWS_FOLDER = "News"

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WS = re.compile(r"\s+")
# Windows refuses these as filenames regardless of extension.
_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


@dataclass
class VaultWriteResult:
    vault: Path
    index_note: Path
    item_notes: list[Path]
    skipped: list[str]
    #: Notes that already existed with *different* content and were left alone.
    #: Re-running the same topic on the same day is supposed to converge, but
    #: the judge can rank differently on the second run — and then "converge"
    #: means yesterday's note is gone with no record it ever existed. In a wiki
    #: that is data loss wearing idempotency's clothes.
    conflicts: list[str] = field(default_factory=list)

    @property
    def written(self) -> int:
        return len(self.item_notes) + 1


def vault_path() -> Path:
    """Vault location, overridable with OBSIDIAN_VAULT."""
    return Path(os.getenv("OBSIDIAN_VAULT", str(DEFAULT_VAULT)))


def _slug(text: str, limit: int = 60) -> str:
    """Filesystem-safe note name derived from untrusted model text."""
    flat = _WS.sub(" ", _ILLEGAL.sub("", text)).strip().strip(". ")
    flat = flat[:limit].strip()
    if not flat:
        return "untitled"
    if flat.upper().split(".")[0] in _RESERVED:
        flat = f"_{flat}"
    return flat


def _yaml(value) -> str:
    """Serialise a scalar or list for YAML frontmatter.

    Everything string-ish is quoted and escaped — a headline containing a colon
    would otherwise produce a broken frontmatter block that Obsidian silently
    fails to parse.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_yaml(v) for v in value) + "]"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _frontmatter(fields: dict) -> str:
    lines = ["---"]
    lines += [f"{key}: {_yaml(value)}" for key, value in fields.items()]
    lines.append("---")
    return "\n".join(lines)


def _target(root: Path, subfolder: str, name: str) -> Path:
    """Resolve a note path and refuse anything that escapes the vault."""
    root = root.resolve()
    candidate = (root / NEWS_FOLDER / subfolder / f"{name}.md").resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Refusing to write outside the vault: {candidate}")
    return candidate


def _item_note(
    item: ScoredItem, topic: str, day: str, models: dict[str, str],
    selected_by: str = "judge",
) -> str:
    front = _frontmatter(
        {
            "title": item.item.headline,
            "date": day,
            "topic": topic,
            "composite": item.composite,
            **item.scores,
            "sources": list(item.item.sources),
            "generator_model": models.get("generator", ""),
            "judge_model": models.get("judge", ""),
            "selected_by": selected_by,
            "tags": ["news", _slug(topic).lower().replace(" ", "-")],
        }
    )
    body = [
        "",
        f"# {item.item.headline}",
        "",
        item.item.summary,
        "",
        "## Why it matters",
        "",
        item.item.why_it_matters,
        "",
        "## Editor's assessment",
        "",
        f"> {item.verdict.reasoning}",
        "",
        "| Dimension | Score |",
        "| --- | --- |",
        *[f"| {k} | {v}/5 |" for k, v in item.scores.items()],
        f"| **composite** | **{item.composite}** |",
        "",
        "## Sources",
        "",
        *[f"- {url}" for url in item.item.sources],
        "",
        "---",
        "",
        f"Topic: [[{_slug(topic)}]] · Date: [[{day}]]",
        "",
    ]
    return front + "\n".join(body)


def _index_note(
    topic: str, day: str, items: list[ScoredItem], overview: str, note_names: list[str]
) -> str:
    front = _frontmatter(
        {
            "title": f"{topic} — {day}",
            "date": day,
            "topic": topic,
            "item_count": len(items),
            "tags": ["news", "digest"],
        }
    )
    body = ["", f"# {topic} — {day}", "", overview, "", "## Top stories", ""]
    for scored, name in zip(items, note_names):
        body.append(f"- **{scored.composite}** — [[{name}]]")
        body.append(f"  - {scored.verdict.reasoning}")
    body += ["", "---", "", f"Topic: [[{_slug(topic)}]] · Date: [[{day}]]", ""]
    return front + "\n".join(body)


def write_digest(
    topic: str,
    overview: str,
    items: list[ScoredItem],
    *,
    vault: Path | None = None,
    day: date | None = None,
    models: dict[str, str] | None = None,
    selected_by: str = "judge",
    force: bool = False,
) -> VaultWriteResult:
    """Write `items` as individual notes plus one index note.

    Filenames are deterministic, so re-running the same topic on the same day
    targets the same files. Identical content is rewritten silently — that is
    the idempotent case. **Different** content is reported as a conflict and
    left alone unless `force`, because a second run whose judge ranked
    differently would otherwise erase the first with no trace.
    """
    root = Path(vault) if vault else vault_path()
    if not root.exists():
        raise FileNotFoundError(
            f"Obsidian vault not found: {root}. Set OBSIDIAN_VAULT to override."
        )

    day_str = (day or date.today()).isoformat()
    models = models or {}

    return _write(root, topic, overview, items, day_str, models, selected_by, force)


@observe(name="write_digest", as_type="tool", capture_input=False)
def _write(root, topic, overview, items, day_str, models, selected_by, force):
    """The actual write, in its own span.

    `@observe` must wrap `tool_call`, not the other way round: `tool_call`
    emits in its `finally`, so if the span closed first the Level 2 metadata
    would land on the *parent* span and collide with Level 1 there.
    """
    # side_effect=True: the only operation in the whole pipeline that mutates
    # state outside the process. That flag is the point of Level 2 — it lets
    # you find every run that touched the vault.
    with tool_call(
        "write_digest",
        side_effect=True,
        args={"topic": topic, "day": day_str, "items": len(items)},
    ) as telemetry:
        result = _write_notes(
            root, topic, overview, items, day_str, models, selected_by, force
        )
        telemetry["notes_written"] = len(result.item_notes)
        telemetry["selected_by"] = selected_by
        if result.conflicts:
            telemetry["conflicts"] = len(result.conflicts)
        return result


def _conflicts(path: Path, content: str) -> bool:
    """True if the note exists with different content.

    Byte-identical is not a conflict — re-running an unchanged digest is the
    idempotent case this design wants, and warning about it would train you to
    ignore the warning.
    """
    if not path.exists():
        return False
    try:
        return path.read_text(encoding="utf-8") != content
    except OSError:
        return True  # unreadable: refuse to clobber what we cannot compare


def _write_notes(root, topic, overview, items, day_str, models, selected_by, force):
    note_names: list[str] = []
    paths: list[Path] = []
    skipped: list[str] = []
    conflicts: list[str] = []

    for scored in items:
        name = _slug(f"{day_str} {scored.item.headline}")
        try:
            path = _target(root, "Items", name)
        except ValueError as exc:
            skipped.append(f"{scored.item.headline}: {exc}")
            continue
        content = _item_note(scored, topic, day_str, models, selected_by)
        if not force and _conflicts(path, content):
            conflicts.append(name)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        note_names.append(name)
        paths.append(path)

    index_name = _slug(f"{day_str} {topic}")
    index_path = _target(root, "Digests", index_name)
    index_content = _index_note(
        topic, day_str, items[: len(note_names)], overview, note_names
    )
    # The index is written only if every item made it. A partial index would
    # link to notes that were skipped, which is worse than not updating it.
    if not conflicts and (force or not _conflicts(index_path, index_content)):
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(index_content, encoding="utf-8")
    elif not conflicts:
        conflicts.append(index_name)

    return VaultWriteResult(
        vault=root, index_note=index_path, item_notes=paths,
        skipped=skipped, conflicts=conflicts,
    )
