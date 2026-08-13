"""Models owned by the research agent.

Three layers, each doing real work:

1. ``HeadlineQuery`` — the *tool input*. Its JSON Schema is what Claude sees
   when deciding how to call the tool, so the field descriptions here are
   prompt engineering, not documentation.
2. ``Article`` / ``HeadlineSearchResult`` — the *tool output*, serialised back
   into the conversation as the tool result.
3. ``NewsDigest`` — the *structured final answer*, handed to the API via
   ``output_config.format`` so the last message is guaranteed-valid JSON
   rather than prose we have to regex.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Category(str, Enum):
    """Feed categories the tool knows how to query.

    Exposed to Claude as an enum so it cannot invent a category name.
    """

    #: Dedicated AI publications. Use this for anything AI-centred: general
    #: news feeds carry too little AI to filter usefully, which shows up as
    #: honest-but-thin digests.
    AI = "ai"
    TOP = "top"
    WORLD = "world"
    BUSINESS = "business"
    TECHNOLOGY = "technology"
    SCIENCE = "science"


class HeadlineQuery(BaseModel):
    """Input schema for the ``search_headlines`` tool."""

    category: Category = Field(
        description=(
            "Which news category to pull from. Pick the single category most "
            "likely to carry the story; call the tool again with a different "
            "category if the first pass is thin. For anything AI-related use "
            "'ai' — it queries dedicated AI publications, whereas 'technology' "
            "is general tech news that merely mentions AI sometimes."
        )
    )
    keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Optional case-insensitive keywords. An article matches if ANY "
            "keyword appears in its title or summary. Leave empty to get the "
            "category's full front page. Prefer 2-4 broad keywords over one "
            "narrow phrase — RSS titles are short and over-filtering returns "
            "nothing."
        ),
    )
    limit: int = Field(
        default=8,
        ge=1,
        le=25,
        description="Maximum number of articles to return (1-25).",
    )


class Article(BaseModel):
    """A single headline as returned by the tool."""

    title: str
    source: str
    url: str
    published: str = Field(description="Publication date, ISO YYYY-MM-DD.")
    summary: str = Field(description="Feed-provided blurb, truncated.")


class HeadlineSearchResult(BaseModel):
    """What ``search_headlines`` hands back to Claude."""

    # Deliberately does NOT echo the query back (category aside, for grounding):
    # the model just sent those arguments, and every field here is re-sent on
    # every subsequent turn.
    category: str
    article_count: int
    articles: list[Article]
    note: str = Field(
        default="",
        description="Set when something went wrong or nothing matched.",
    )


# --- Structured output -------------------------------------------------------
#
# For `output_config.format` the schema must have `additionalProperties: false`
# and every property listed in `required`. `extra="forbid"` gives us the former;
# declaring no defaults gives us the latter. Do NOT add ge/le/min_length to
# these models — numeric and string constraints are not supported by the
# structured-output schema compiler.


class DigestItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headline: str = Field(description="One-line restatement of the story.")
    summary: str = Field(description="Two or three sentences of what happened.")
    why_it_matters: str = Field(
        description="One sentence of significance. Say so plainly if it is routine."
    )
    sources: list[str] = Field(description="URLs of the articles this item draws on.")


class NewsDigest(BaseModel):
    """The research agent's final, validated answer."""

    model_config = ConfigDict(extra="forbid")

    topic: str = Field(description="The topic the user asked about.")
    overview: str = Field(
        description="Two or three sentences summarising the whole picture."
    )
    items: list[DigestItem] = Field(
        description="The individual stories, most important first."
    )
    coverage_note: str = Field(
        description=(
            "Honest note on coverage: which categories were searched, and "
            "whether the feeds actually had much on this topic. Say plainly "
            "if coverage was thin rather than padding the digest."
        )
    )
