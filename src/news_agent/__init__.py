"""A minimal Claude agent system that reads the news and files the best of it.

Layering (dependencies point one way only)::

    core  <-  agents/research  <-  agents/judge  <-  sinks  <-  orchestrator
"""

# MUST come first. `core.observability` decides whether to initialise Langfuse
# when it is imported, and `@observe` is applied at def-time, so it cannot be
# deferred. If .env loads after that import, the keys arrive too late and
# tracing silently stays off.
from .core.env import load_dotenv

DOTENV_LOADED = load_dotenv()

from .agents.judge import (  # noqa: E402
    MIN_COMPOSITE,
    ItemVerdict,
    JudgeError,
    JudgeResult,
    ScoredItem,
    judge_digest,
)
from .agents.research import (  # noqa: E402
    Article,
    Category,
    DigestError,
    DigestItem,
    DigestResult,
    HeadlineQuery,
    NewsDigest,
    run_digest,
)
from .core.pricing import DEFAULT_MODEL, PRICING  # noqa: E402
from .core.types import TokenUsage  # noqa: E402
from .orchestrator import PipelineResult, run_pipeline  # noqa: E402

__all__ = [
    "DEFAULT_MODEL",
    "DOTENV_LOADED",
    "MIN_COMPOSITE",
    "PRICING",
    "Article",
    "Category",
    "DigestError",
    "DigestItem",
    "DigestResult",
    "HeadlineQuery",
    "ItemVerdict",
    "JudgeError",
    "JudgeResult",
    "NewsDigest",
    "PipelineResult",
    "ScoredItem",
    "TokenUsage",
    "judge_digest",
    "run_digest",
    "run_pipeline",
]
