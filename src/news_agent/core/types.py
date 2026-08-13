"""Types shared across every agent.

Deliberately tiny, and deliberately free of any agent import. `core` sits at
the bottom of the dependency graph — the moment something here needs to know
about a specific agent's models, it belongs in that agent instead.

That is why `ScoredItem` is *not* here: it composes a research `DigestItem`
with a judge `ItemVerdict`, so it lives in `agents/judge/models.py` as the
judge's output contract. Sinks import it from there, which is a downstream
dependency rather than an inverted one.
"""

from __future__ import annotations

from pydantic import BaseModel


class TokenUsage(BaseModel):
    """Accumulated token spend for one model call or one multi-turn loop.

    Not a structured-output model — this is ours, filled in from the API's
    usage fields, so ordinary defaults are fine here.
    """

    #: Uncached input only. The Anthropic API reports cached tokens in their
    #: own fields, so these four are mutually exclusive buckets — which is also
    #: what Langfuse requires: every token counted exactly once, or cost
    #: display silently goes wrong.
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    turns: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_creation_tokens
        )

    def usage_details(self) -> dict[str, int]:
        """Token buckets in Langfuse's shape.

        `total` is optional there (derived as the sum when omitted); we send it
        explicitly so the value shown is the one we computed. Zero buckets are
        dropped rather than sent as 0 — a cost bucket that never applies is
        noise on every trace.
        """
        buckets = {
            "input": self.input_tokens,
            "output": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_tokens,
            "cache_creation_input_tokens": self.cache_creation_tokens,
        }
        details = {k: v for k, v in buckets.items() if v}
        details["total"] = self.total_tokens
        return details
