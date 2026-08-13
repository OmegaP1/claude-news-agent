"""Judge: scores and ranks digest items. A single structured call —
no tools, no loop — but packaged like an agent for consistency."""

from .agent import JudgeError, JudgeResult, composite, judge_digest
from .config import MIN_COMPOSITE, WEIGHTS
from .models import ItemVerdict, JudgeVerdicts, ScoredItem

__all__ = [
    "MIN_COMPOSITE",
    "WEIGHTS",
    "ItemVerdict",
    "JudgeError",
    "JudgeResult",
    "JudgeVerdicts",
    "ScoredItem",
    "composite",
    "judge_digest",
]
