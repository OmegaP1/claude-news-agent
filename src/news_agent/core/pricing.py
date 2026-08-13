"""Model choice and cost estimation.

Kept separate from the agent so the cost/quality tradeoff is one obvious file
rather than a constant buried in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import TokenUsage


@dataclass(frozen=True)
class ModelPricing:
    """USD per million tokens, per the public Anthropic price list."""

    model_id: str
    input_per_mtok: float
    output_per_mtok: float
    note: str

    def breakdown_usd(self, usage: TokenUsage) -> dict[str, float]:
        """Per-bucket cost, in the shape Langfuse's `cost_details` wants.

        The keys **mirror `usage_details` exactly**. Langfuse requires the
        buckets to be mutually exclusive and to line up with the token buckets;
        folding cache-read cost into `input` (as this used to) breaks that
        contract and makes the per-type cost display wrong.

        Rates: cache reads bill at ~0.1x the input rate, cache writes at ~1.25x
        (5-minute TTL).
        """
        per_bucket = {
            "input": usage.input_tokens * self.input_per_mtok,
            "output": usage.output_tokens * self.output_per_mtok,
            "cache_read_input_tokens": usage.cache_read_tokens * self.input_per_mtok * 0.1,
            "cache_creation_input_tokens": (
                usage.cache_creation_tokens * self.input_per_mtok * 1.25
            ),
        }
        costs = {k: v / 1_000_000 for k, v in per_bucket.items() if v}
        costs["total"] = sum(costs.values())
        return costs

    def estimate_usd(self, usage: TokenUsage) -> float:
        return self.breakdown_usd(usage)["total"]


PRICING: dict[str, ModelPricing] = {
    "claude-haiku-4-5": ModelPricing(
        "claude-haiku-4-5", 1.00, 5.00, "cheapest; plenty for summarising headlines"
    ),
    "claude-sonnet-5": ModelPricing(
        "claude-sonnet-5", 3.00, 15.00, "step up if digests feel shallow"
    ),
    "claude-opus-4-8": ModelPricing("claude-opus-4-8", 5.00, 25.00, "overkill here"),
    "claude-opus-5": ModelPricing("claude-opus-5", 5.00, 25.00, "overkill here"),
}

# Haiku 4.5 supports tool use and structured outputs, which is everything this
# agent needs. The hard work is done by the JSON schema, not by model IQ.
DEFAULT_MODEL = "claude-haiku-4-5"


def pricing_for(model_id: str) -> ModelPricing | None:
    """None for an unknown model — we report tokens but not a dollar figure
    rather than inventing a price."""
    return PRICING.get(model_id)


def format_cost(model_id: str, usage: TokenUsage) -> str:
    parts = [
        f"{usage.turns} turn{'s' if usage.turns != 1 else ''}",
        f"{usage.input_tokens:,} in",
        f"{usage.output_tokens:,} out",
    ]
    if usage.cache_read_tokens:
        parts.append(f"{usage.cache_read_tokens:,} cached")

    price = pricing_for(model_id)
    if price is None:
        return f"{model_id}: {', '.join(parts)} (no price on file)"

    usd = price.estimate_usd(usage)
    return f"{model_id}: {', '.join(parts)} ≈ ${usd:.4f}"
