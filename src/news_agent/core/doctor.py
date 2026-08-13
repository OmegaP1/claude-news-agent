"""`--check`: verify credentials before spending anything on a real run.

Two stages, cheapest first:

1. ``models.retrieve`` — proves the key is valid and has model access.
   The Models API bills no tokens, so this stage is free.
2. ``messages.create`` with ``max_tokens=0`` — proves the account can actually
   bill. The API runs prefill and returns immediately with no output tokens,
   so this costs a handful of input tokens (~$0.00001 on Haiku).

Stage 1 alone cannot detect an empty balance, which is exactly the failure a
new account hits — hence stage 2.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import anthropic

from .pricing import DEFAULT_MODEL


@dataclass
class CheckResult:
    ok: bool
    lines: list[str]

    def report(self) -> str:
        return "\n".join(self.lines)


def mask(key: str) -> str:
    """Never print a key in full — logs and screenshots leak."""
    if len(key) <= 12:
        return "***"
    return f"{key[:11]}…{key[-4:]}"


def check(model: str = DEFAULT_MODEL, client: anthropic.Anthropic | None = None) -> CheckResult:
    lines: list[str] = []

    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return CheckResult(
            False,
            [
                "FAIL  No ANTHROPIC_API_KEY found.",
                "      Put it in a .env file next to pyproject.toml:",
                "          ANTHROPIC_API_KEY=sk-ant-...",
                "      (.env is gitignored — never commit it.)",
            ],
        )

    if not key.startswith("sk-ant-"):
        lines.append(f"WARN  Key {mask(key)} does not look like an Anthropic key.")
    else:
        lines.append(f"OK    Key found: {mask(key)}")

    client = client or anthropic.Anthropic()

    # --- Stage 1: is the key valid? (free) ---
    try:
        info = client.models.retrieve(model)
        lines.append(f"OK    Key is valid and can see {info.id}.")
    except anthropic.AuthenticationError:
        lines.append("FAIL  Key rejected (401). It may be revoked or mistyped.")
        return CheckResult(False, lines)
    except anthropic.PermissionDeniedError:
        lines.append(f"FAIL  Key is valid but lacks access to {model} (403).")
        return CheckResult(False, lines)
    except anthropic.NotFoundError:
        lines.append(f"FAIL  Model {model} not found (404). Check the model id.")
        return CheckResult(False, lines)
    except anthropic.APIConnectionError as exc:
        lines.append(f"FAIL  Could not reach the API: {exc}")
        return CheckResult(False, lines)

    # --- Stage 2: can the account bill? (~$0.00001) ---
    try:
        client.messages.create(
            model=model,
            max_tokens=0,
            messages=[{"role": "user", "content": "ping"}],
        )
        lines.append("OK    Billing works — the account can serve requests.")
    except anthropic.BadRequestError as exc:
        message = str(exc).lower()
        if "credit" in message or "billing" in message or "balance" in message:
            lines.append("FAIL  Key is valid but the account cannot bill:")
            lines.append(f"      {exc}")
            lines.append("      Add credits, or check whether your org's tier covers usage.")
            return CheckResult(False, lines)
        # max_tokens=0 is rejected in some configurations — not a billing fault.
        lines.append(f"WARN  Zero-token probe rejected ({exc}).")
        lines.append("      Auth is fine; balance unverified. A real run will tell you.")
    except anthropic.RateLimitError:
        lines.append("WARN  Rate limited on the probe. Auth is fine; try again shortly.")

    lines.append("")
    lines.append(f"Ready. Try:  python -m news_agent \"AI regulation\" --model {model}")
    return CheckResult(True, lines)
