"""Tests for .env loading and the --check preflight."""

from __future__ import annotations

import anthropic
import pytest

from news_agent.core.doctor import check, mask
from news_agent.core.env import find_dotenv, load_dotenv


def test_loads_pairs_and_strips_quotes(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        '# comment\n\nANTHROPIC_API_KEY="sk-ant-abc"\n'
        "export LANGFUSE_HOST=https://x.test\n"
        "MALFORMED\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)

    loaded = load_dotenv(env)
    assert set(loaded) == {"ANTHROPIC_API_KEY", "LANGFUSE_HOST"}
    import os
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-abc"      # quotes stripped
    assert os.environ["LANGFUSE_HOST"] == "https://x.test"      # `export ` stripped


def test_real_env_wins_over_file(tmp_path, monkeypatch):
    """An explicit override must never be clobbered by the file."""
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=from-file", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")

    load_dotenv(env)
    import os
    assert os.environ["ANTHROPIC_API_KEY"] == "from-shell"


def test_missing_file_is_not_an_error(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == []


def test_find_walks_up_to_project_root(tmp_path):
    (tmp_path / ".env").write_text("A=1", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_dotenv(nested) == tmp_path / ".env"


def test_mask_never_reveals_the_key():
    masked = mask("sk-ant-api03-SECRETSECRETSECRET-1234")
    assert "SECRETSECRET" not in masked
    assert masked.startswith("sk-ant-api")
    assert masked.endswith("1234")


# --- doctor ------------------------------------------------------------------


class FakeClient:
    def __init__(self, on_retrieve=None, on_create=None):
        outer = self

        class Models:
            def retrieve(self, model):
                if on_retrieve:
                    raise on_retrieve
                return type("M", (), {"id": model})()

        class Messages:
            def create(self, **kwargs):
                outer.create_kwargs = kwargs
                if on_create:
                    raise on_create
                return object()

        self.models = Models()
        self.messages = Messages()
        self.create_kwargs = None


def _response(status):
    import httpx

    return httpx.Response(status, request=httpx.Request("GET", "https://api.test"))


def test_missing_key_fails_with_guidance(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = check()
    assert not result.ok
    assert ".env" in result.report()


def test_happy_path_probes_with_zero_tokens(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-123456")
    client = FakeClient()
    result = check(client=client)
    assert result.ok
    # The probe must not generate output tokens.
    assert client.create_kwargs["max_tokens"] == 0
    assert "Billing works" in result.report()


def test_invalid_key_reported_as_auth_failure(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-bad")
    err = anthropic.AuthenticationError("bad", response=_response(401), body=None)
    result = check(client=FakeClient(on_retrieve=err))
    assert not result.ok
    assert "401" in result.report()


def test_empty_balance_is_distinguished_from_auth(monkeypatch):
    """The whole point of stage 2: a valid key with no credits."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-123456")
    err = anthropic.BadRequestError(
        "Your credit balance is too low to access the API",
        response=_response(400),
        body=None,
    )
    result = check(client=FakeClient(on_create=err))
    assert not result.ok
    assert "cannot bill" in result.report()
    assert "credits" in result.report().lower()


def test_unrelated_400_does_not_claim_billing_failure(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-123456")
    err = anthropic.BadRequestError(
        "max_tokens: must be >= 1", response=_response(400), body=None
    )
    result = check(client=FakeClient(on_create=err))
    assert result.ok  # auth verified; balance simply unproven
    assert "unverified" in result.report()
