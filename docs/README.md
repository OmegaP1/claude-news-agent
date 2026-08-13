# Documentation

A minimal Claude agent that reads the news, judges it, and files the best of it
into an Obsidian vault. Two models, one tool, one human checkpoint.

## Reading order

| Start here if you want to… | Document |
|---|---|
| **run it** | [../README.md](../README.md) — setup, flags, costs |
| **understand the shape** | [architecture.md](architecture.md) — layers, data flow, what enforces them |
| **debug a trace** | [observability.md](observability.md) — three telemetry levels, what to filter on |
| **change a prompt safely** | [evaluation.md](evaluation.md) — golden fixtures, the replay gate |
| **fix something broken** | [operations.md](operations.md) — exit codes, failure modes, runbook |

## The short version

```
research (Haiku 4.5, agentic) → judge (Sonnet 5) → select (code)
                                     → review (human) → vault
```

**One stage is agentic.** Research is a `tool_runner` loop where the model
decides what to search for. Everything after it is deterministic — selection is
a rule, persistence is a file write — so there is nothing for a model to
decide.

**Three things run on every request without being asked for:** a grounding
check (did the model cite a URL the tool never returned?), a cost ledger split
into four token buckets, and levelled telemetry that degrades to a no-op with
no keys configured.

**Two things run when you ask:** a spend ceiling that aborts between stages,
and a regression gate that re-judges frozen digests against your own past
decisions.

## Conventions worth knowing before editing

- **Every agent has the same five files** — `__init__`, `agent`, `config`,
  `instructions`, `models` — whether or not it uses tools. A test enforces it.
- **Prompts live in `instructions.py`.** In an LLM project the prompt iterates
  more than anything else; its diffs should be readable on their own.
- **`config.py` is tuning; `.env` is secrets.** One `.env`, at the root.
- **`core` imports no agent.** It takes primitives and callers do the mapping.
- **`evals` imports no sink or orchestrator** — that restriction is what keeps
  a replay costing a cent instead of a full run.

All five are checked by [`tests/test_architecture.py`](../tests/test_architecture.py),
because a folder structure is a claim about dependencies and an unchecked claim
quietly stops being true.

## Known gaps

Stated here rather than discovered later:

- **Prompt injection is mitigated, not solved.** Feed text is fenced with a
  per-process nonce and both prompts state that fenced content is data. The
  judge's `Literal[1..5]` schema means no injection can produce an out-of-range
  score. But a sufficiently persuasive headline can still influence a
  judgement, which is why a human sees the ranking before anything is filed.
- **Claim checking is a smoke detector, not a proof.** Figures are verified
  against the cited article and lexical overlap is measured, but a fabricated
  *qualitative* claim using the source's own vocabulary would pass.
- **Feed health is visible but not alerting.** A dead feed is named in the
  trace; nothing tells you unprompted.
- **No scheduled runs.** Everything is manual — which is also why fixtures
  accumulate slowly.
- **The floor is deliberately left strict.** 21 fixtures say `MIN_COMPOSITE`
  cuts items worth filing; keeping it is a considered choice, not an oversight.
