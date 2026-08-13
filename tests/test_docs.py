"""Documentation fitness tests.

Docs rot silently. A moved file leaves a dead link that nothing complains
about, and a runbook that references a flag which no longer exists is worse
than no runbook — it sends you looking for something that was never there.

These are cheap checks for the failures that actually happen: broken relative
links, and documented flags or exit codes that drifted out of the CLI.
"""

from __future__ import annotations

import pathlib
import re

import pytest

import news_agent

ROOT = pathlib.Path(news_agent.__file__).resolve().parents[2]
DOCS = ROOT / "docs"

#: [text](target) — skipping external URLs and pure anchors.
_LINK = re.compile(r"\[[^\]]+\]\((?!https?://|#)([^)]+)\)")


def _markdown_files() -> list[pathlib.Path]:
    return [ROOT / "README.md", *sorted(DOCS.rglob("*.md"))]


def test_there_are_docs_to_check():
    """Guard the guard: a bad ROOT would make every test below vacuously pass."""
    assert len(_markdown_files()) >= 5


@pytest.mark.parametrize(
    "path", _markdown_files(), ids=lambda p: p.name
)
def test_every_relative_link_resolves(path):
    broken = []
    for target in _LINK.findall(path.read_text(encoding="utf-8")):
        clean = target.split("#", 1)[0]
        if not clean:
            continue
        if not (path.parent / clean).resolve().exists():
            broken.append(target)
    assert not broken, f"{path.name} has dead links: {broken}"


def test_the_observability_doc_lives_in_docs():
    """It started life as `ai-logging-diagram.md` at the repo root, imported
    from another project. Both the name and the location were wrong."""
    assert (DOCS / "observability.md").exists()
    assert not (ROOT / "ai-logging-diagram.md").exists()


def test_code_points_at_the_current_path():
    """A docstring referencing a moved file is a dead link the compiler will
    never catch."""
    telemetry = (ROOT / "src" / "news_agent" / "core" / "telemetry.py").read_text("utf-8")
    assert "docs/observability.md" in telemetry
    assert "ai-logging-diagram" not in telemetry


def test_documented_flags_exist_in_the_cli():
    """The runbook is only useful if its flags are real."""
    cli = (ROOT / "src" / "news_agent" / "__main__.py").read_text("utf-8")
    documented = set()
    for path in _markdown_files():
        documented |= set(re.findall(r"(?<![\w-])(--[a-z][a-z-]+)", path.read_text("utf-8")))

    # `--help` is argparse-generated, never literal in the source; the rest
    # belong to other tools the docs legitimately mention.
    not_ours = {"--help", "--no-deps", "--quiet"}
    missing = {f for f in documented - not_ours if f'"{f}"' not in cli}
    assert not missing, f"documented but not in the CLI: {sorted(missing)}"


def test_documented_exit_codes_are_returned_somewhere():
    """Exit codes are the contract with anything scripting this tool."""
    cli = (ROOT / "src" / "news_agent" / "__main__.py").read_text("utf-8")
    for code in (1, 2, 3, 4, 5, 6, 7):
        assert f"return {code}" in cli, f"exit code {code} is documented but never returned"


def test_every_doc_is_reachable_from_the_index():
    """An unlinked document is one nobody will find, which makes writing it a
    waste rather than a nice-to-have."""
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    for doc in DOCS.glob("*.md"):
        if doc.name == "README.md":
            continue
        assert doc.name in index, f"{doc.name} is not linked from docs/README.md"


# --- nothing personal leaks into a public repository -------------------------

#: Things that must never appear in committed source. A hardcoded home
#: directory is wrong for every other user *and* publishes a username.
_PERSONAL = re.compile(
    r"[A-Za-z]:" + re.escape("\\") + r"Users" + re.escape("\\") + r"[^\\\s\"']+"
    r"|/(?:home|Users)/[^/\s\"']+"
)


def _committed_text_files() -> list[pathlib.Path]:
    keep = {".py", ".md", ".toml", ".txt", ".yml", ".yaml"}
    skip = {".venv", "__pycache__", ".pytest_cache", "node_modules", ".git"}
    return [
        p for p in ROOT.rglob("*")
        if p.suffix in keep
        and p.is_file()
        and not any(part in skip for part in p.parts)
    ]


def test_no_absolute_home_directory_is_committed():
    offenders = []
    for path in _committed_text_files():
        for match in _PERSONAL.findall(path.read_text(encoding="utf-8", errors="replace")):
            # The regex itself and the docs describing the rule are exempt.
            if path.name in {"test_docs.py"}:
                continue
            offenders.append(f"{path.relative_to(ROOT)}: {match}")
    assert not offenders, f"personal paths in committed files: {offenders}"


def test_the_env_file_is_ignored():
    """The one mistake with real consequences."""
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    assert ".env" in ignored


def test_the_env_example_carries_no_real_key():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        if "=" in line and not line.startswith("#"):
            value = line.split("=", 1)[1].strip()
            assert value.endswith("...") or value.startswith("https://"), (
                f"{line} looks like a real value, not a placeholder"
            )


def test_docs_do_not_present_the_deprecated_api_as_current():
    """The README's architecture section had drifted: it showed
    `output_format=NewsDigest` and `.parsed_output` as the way to do structured
    output, months after the migration to `output_config.format`. A code sample
    that does not work is worse than no code sample."""
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "output_format=" in line or ".parsed_output" in line:
                assert "deprecated" in text.lower(), (
                    f"{path.name} shows the deprecated structured-output API "
                    f"without saying so: {line.strip()}"
                )
