"""Tuning knobs for the research agent.

Everything that changes *how this agent behaves* lives here — one obvious file
to open when you want to make it cheaper, deeper, or slower. Secrets do not:
those stay in the single `.env` at the project root, loaded by `core.env`.
"""

from __future__ import annotations

import os

from ...core.pricing import DEFAULT_MODEL

#: Override per-run with --model, or globally with NEWS_AGENT_MODEL.
MODEL = os.getenv("NEWS_AGENT_MODEL", DEFAULT_MODEL)

MAX_TOKENS = 4_000

#: Default ceiling on tool-calling rounds — the main guard on cost, since each
#: round resends the whole conversation.
MAX_ITERATIONS = 6

#: One turn is consumed by each tool call, and the model needs a further turn
#: to write the digest. Below 3 it can spend the whole budget searching and
#: never get to answer — which bills you for nothing.
MIN_ITERATIONS = 3
