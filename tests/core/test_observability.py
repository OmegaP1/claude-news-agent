"""Tracing must never be able to break the agent."""

from __future__ import annotations

from news_agent.core.observability import _identity, flush, status


def test_noop_supports_bare_decorator():
    @_identity
    def f(x):
        return x * 2

    assert f(3) == 6


def test_noop_supports_decorator_factory():
    """Regression: `@observe(name=...)` must work when Langfuse is absent."""

    @_identity(name="anything", as_type="span")
    def f(x):
        return x * 2

    assert f(3) == 6


def test_flush_is_safe_without_client():
    flush()  # must not raise


def test_status_reports_disabled_without_keys(monkeypatch):
    """Hermetic: `enabled` is decided at import time, and on a machine with a
    real .env it is already True. Patch the module state as well as the env,
    or this test just mirrors whoever's laptop it runs on."""
    import news_agent.core.observability as obs

    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr(obs, "enabled", False)
    assert "disabled" in obs.status()


def test_status_reports_enabled_when_configured(monkeypatch):
    import news_agent.core.observability as obs

    monkeypatch.setattr(obs, "enabled", True)
    assert "enabled" in obs.status()


def test_render_survives_cp1252_console(capsys, monkeypatch):
    """Regression: the CLI printed '≈' and '→' and died with
    UnicodeEncodeError on a default Windows console, *after* the API call had
    already been billed."""
    import io
    import sys

    from news_agent.__main__ import _force_utf8, render
    from news_agent.agents.research.models import DigestItem, NewsDigest

    digest = NewsDigest(
        topic="café",
        overview="Ünicode — everywhere…",
        items=[
            DigestItem(
                headline="→ arrow",
                summary="≈ approx",
                why_it_matters="—",
                sources=["https://e.test/ü"],
            )
        ],
        coverage_note="ok",
    )
    text = render(digest)

    # A cp1252 stream must not raise once we have reconfigured it.
    buf = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", buf)
    _force_utf8()
    buf.write(text)  # would raise UnicodeEncodeError without the fix
    buf.flush()


def test_dotenv_loads_before_observability_is_imported():
    """Regression: observability decides on Langfuse at import time and
    @observe is applied at def-time, so .env must load first. When it didn't,
    tracing silently stayed off despite valid keys."""
    import news_agent

    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(news_agent.__file__).read_text(encoding="utf-8"))
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]

    # Asserted structurally rather than on literal strings, so moving modules
    # around cannot silently retire this guard.
    env_line = next(
        n.lineno for n in imports if any(a.name == "load_dotenv" for a in n.names)
    )
    first_agent_line = min(
        (n.lineno for n in imports if n.module and "agents" in n.module),
        default=float("inf"),
    )
    assert env_line < first_agent_line
    assert hasattr(news_agent, "DOTENV_LOADED")


def test_tracing_is_disabled_during_tests():
    """conftest sets the flag before news_agent is imported, so a developer
    with real Langfuse keys does not get junk traces from every pytest run."""
    import news_agent.core.observability as obs

    assert obs.enabled is False
    assert obs._client is None
    assert "disabled" in obs.status()


def test_disable_flag_beats_configured_keys(monkeypatch):
    import news_agent.core.observability as obs

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv(obs.DISABLE_FLAG, "1")
    assert obs._configured() is False


def test_report_generation_is_a_noop_without_client():
    from news_agent.core.observability import report_generation

    report_generation(model="x", usage_details={"input": 1})  # must not raise


def test_run_digest_does_not_serialise_the_client():
    """capture_input=False: the default would ship the whole Anthropic client
    (with its auth) into the trace payload."""
    import inspect

    from news_agent.agents.research import agent

    src = inspect.getsource(agent)
    decorator = src[src.index('@observe(name="news-digest"') : src.index("def run_digest")]
    assert "capture_input=False" in decorator
