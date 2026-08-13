"""Architecture fitness tests.

A folder structure is a claim about dependencies. Without something checking
it, the claim quietly stops being true — someone adds one convenient import
and the layering is gone with every other test still green.

The claim::

    core  <-  agents/research  <-  agents/judge  <-  sinks  <-  orchestrator

Arrows point the way imports are allowed to go. `core` knows about nobody;
`orchestrator` may know about everybody.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import news_agent

SRC = pathlib.Path(news_agent.__file__).parent

#: layer -> the layers it may import from (besides itself).
ALLOWED: dict[str, set[str]] = {
    "core": set(),
    "agents.research": {"core"},
    "agents.judge": {"core", "agents.research"},
    "sinks": {"core", "agents.research", "agents.judge"},
    "review": {"core", "agents.research", "agents.judge"},
    # `evals` may NOT import sinks or orchestrator. That is the economic
    # argument encoded as a rule: replay re-runs the judge alone against a
    # frozen digest, so it costs a cent. An orchestrator import would silently
    # drag research and vault writes back into every replay.
    "evals": {"core", "agents.research", "agents.judge"},
    "orchestrator": {"core", "agents.research", "agents.judge", "sinks", "review"},
}


def _layer(path: pathlib.Path) -> str | None:
    rel = path.relative_to(SRC).as_posix()
    if rel.startswith("core/"):
        return "core"
    if rel.startswith("agents/research/"):
        return "agents.research"
    if rel.startswith("agents/judge/"):
        return "agents.judge"
    if rel.startswith("sinks/"):
        return "sinks"
    if rel.startswith("evals/"):
        return "evals"
    if rel == "review.py":
        return "review"
    if rel == "orchestrator.py":
        return "orchestrator"
    return None  # __init__.py and __main__.py compose everything: exempt


def _imported_layers(path: pathlib.Path) -> set[str]:
    """Resolve this module's relative imports to layer names."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parts = path.relative_to(SRC).parent.as_posix().split("/") if path.parent != SRC else []
    parts = [p for p in parts if p]

    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.level:
            continue
        base = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
        target = ".".join([*base, *(node.module.split(".") if node.module else [])])
        for layer in ALLOWED:
            if target == layer or target.startswith(layer + "."):
                found.add(layer)
    return found


def _modules():
    return [p for p in SRC.rglob("*.py") if _layer(p) is not None]


def test_there_are_modules_to_check():
    """Guard the guard: a path typo would silently make every test below pass."""
    assert len(_modules()) >= 10


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_module_respects_the_layering(path):
    layer = _layer(path)
    allowed = ALLOWED[layer] | {layer}
    violations = _imported_layers(path) - allowed
    assert not violations, (
        f"{path.relative_to(SRC)} is in '{layer}' and may only import from "
        f"{sorted(allowed)}, but imports from {sorted(violations)}."
    )


def test_core_depends_on_nothing_internal():
    """The load-bearing one. The moment core imports an agent, 'shared' has
    become 'everything', and the folder names stop meaning anything."""
    for path in SRC.glob("core/*.py"):
        assert _imported_layers(path) <= {"core"}, f"{path.name} escapes core"


def test_scored_item_lives_with_the_judge_not_in_core():
    """It composes a research DigestItem with a judge ItemVerdict — putting it
    in core would drag both agents into the shared layer."""
    from news_agent.agents.judge.models import ScoredItem  # noqa: F401

    core_types = (SRC / "core" / "types.py").read_text(encoding="utf-8")
    assert "ScoredItem" not in core_types.split('"""')[2]  # not in the code body


def test_prompts_are_not_buried_in_implementation_files():
    """Prompts are the highest-iteration artefact in an LLM project; they get
    their own module so their diffs are readable on their own."""
    for agent_dir in (SRC / "agents").iterdir():
        if not agent_dir.is_dir() or agent_dir.name.startswith("_"):
            continue
        assert (agent_dir / "instructions.py").exists(), (
            f"{agent_dir.name} has no instructions.py"
        )
        body = (agent_dir / "agent.py").read_text(encoding="utf-8")
        assert 'SYSTEM = """' not in body and 'PROMPT = """' not in body


def test_every_agent_has_the_same_shape():
    """Consistency is the point of the folder layout: you always know where to
    look, whether or not that agent happens to use tools."""
    for agent_dir in (SRC / "agents").iterdir():
        if not agent_dir.is_dir() or agent_dir.name.startswith("_"):
            continue
        for required in ("__init__.py", "agent.py", "config.py", "instructions.py", "models.py"):
            assert (agent_dir / required).exists(), (
                f"agents/{agent_dir.name} is missing {required}"
            )


def test_secrets_stay_in_one_place():
    """Per-agent .env files would multiply where a key can leak."""
    assert not list(SRC.rglob(".env")), "secrets must live in the single root .env"
    for agent_dir in (SRC / "agents").iterdir():
        if agent_dir.is_dir():
            config = agent_dir / "config.py"
            if config.exists():
                text = config.read_text(encoding="utf-8")
                assert "API_KEY" not in text, f"{agent_dir.name}/config.py mentions a key"
