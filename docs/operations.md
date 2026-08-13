# Operations

**Runbook: what each flag does, what each exit code means, and what to do when
something breaks.**

For how to *use* the tool, see the [README](../README.md). This is for when it
misbehaves.

---

## Exit codes

Every one of these is deliberate. A pipeline consuming this tool can branch on
them.

| Code | Meaning | What to do |
|---|---|---|
| `0` | Success | — |
| `1` | Research or judge failed | Read the message; usually `--max-iterations` too low |
| `2` | No `ANTHROPIC_API_KEY` | Put it in `.env`, or `ant auth login` |
| `3` | **Fabricated sources** | The model cited URLs the tool never returned. Do not trust the digest |
| `4` | Vault not found | Set `OBSIDIAN_VAULT` |
| `5` | Vault conflicts | Notes exist with different content. Inspect, then `--force` |
| `6` | Budget ceiling hit | Raise `--max-usd` or lower `--max-iterations` |
| `7` | Judge regression | `--replay` found agreement fell. Revert the rubric change |

**Exit 3 is not a warning.** A digest citing sources that do not exist is a
correctness failure, not a cosmetic one, and it exits non-zero so nothing
downstream treats the output as clean.

---

## Cost control

A run is roughly **$0.03** — about $0.02 research, $0.01 judge. Three
independent guards:

```bash
--max-iterations 6      # bounds tool-calling ROUNDS (default 6, minimum 3)
--max-usd 0.05          # bounds DOLLARS, aborts between stages
--dry-run               # judge and rank, write nothing
```

**`--max-iterations` has a floor of 3 and argparse rejects anything lower
before any API call.** Each round is one model turn: a tool call costs one, and
writing the digest costs one more. A budget that cannot finish bills you for a
run that returns nothing — this was a real bug that burned money and produced
nothing, which is why the rejection happens at parse time.

Turning it *down* is a false economy for the same reason. The levers that
actually work are the tool's `limit` (fewer articles per call) and a narrower
topic.

**`--max-usd` is checked after research and before the judge.** That is the
only point where stopping still saves money: research is already billed, the
judge runs on a pricier model and has not started. A check after the judge
would be a report, not a guard.

### Prompt caching does not apply here

Measured, not assumed: the cacheable prefix is 993 tokens and Haiku 4.5's
minimum is 4,096. Padding to reach it would cost **more** than not caching
(6,758 vs 4,965 tokens). Do not add cache breakpoints hoping they help.

---

## Failure modes

### A topic returns nothing

**Symptom:** the digest is thin or the run is worth cancelling with `q`.

**Why:** an RSS feed holds what it holds. Measured on the AI feeds, that is
about **four days** — so a narrow topic can miss news that existed but scrolled
off before you asked. There is no "daily" setting to widen; the window is a
property of the feed.

**Mitigation:** `--window-days` (default 7) draws on a local cache of articles
seen in previous runs, at `%LOCALAPPDATA%\news-agent\articles.jsonl` or
`NEWS_AGENT_CACHE`. Cached articles are appended after the live fetch, so a
broad query still gets today's news first and the window only shows through
when the keyword filter would otherwise return nothing.

**It starts empty.** The cache cannot recover last week — the benefit accrues
from the next run onward, and a topic that fails today will still fail today.

**When the window is not the problem:** research topics like *reinforcement
learning* or *synthetic data* are thin in industry news feeds no matter how far
back you look. `from_cache` in the tool span tells you which case you are in —
a topic answered entirely from cache means the live feeds do not cover it, and
the fix is a research-oriented feed in `agents/research/tools.py`, not a longer
window.

### A feed died

**Symptom:** digests feel thinner than usual; no error anywhere.

**Why it is hard to see:** `article_count` is per *category*, aggregated after
five feeds are merged round-robin, and `_fetch` swallows network errors by
design so one dead feed cannot take down a multi-feed query. Four healthy feeds
make the count look fine.

**How to check:** Langfuse → Filters → Metadata → `feeds_failed` exists. The
tool span names the specific feeds:

```json
{"pipeline_level": 2, "tool_name": "search_headlines",
 "feeds_ok": 4, "feeds_failed": ["MIT Tech Review"]}
```

`feeds_empty` is the quieter one — reachable and well-formed but zero
articles. A feed that silently stops publishing looks like a slow news day
until it has been empty for a week.

**Fix:** update the URL in `agents/research/tools.py`.

### Vault conflicts (exit 5)

**Symptom:** `N note(s) already exist with different content and were left
alone.`

**Why:** filenames are deterministic, so re-running the same topic on the same
day targets the same files. Identical content is rewritten silently — that is
the intended idempotent case. **Different** content means the judge ranked
differently on the second run, and overwriting would erase the first with no
record it existed.

**Fix:** read the existing note. If the new one is better, `--force`.

**Known wart:** when only the *index* conflicts, the item notes are still
written, so the vault briefly holds notes the index does not link to. The index
is a per-run record of what was picked, and silently rewriting it would lose
the earlier run's decisions — so refusing is the lesser evil, but the
inconsistency is real.

### The model fabricated a source (exit 3)

**Symptom:** `WARNING: N cited source(s) were never returned by the tool`.

**Why:** the system prompt forbids inventing URLs; the model did it anyway.
This is what the grounding check exists to catch.

**Fix:** nothing automatic. Check the `grounding` score trend in Langfuse — if
it is drifting down, the research prompt needs tightening. An isolated instance
is noise.

### `--max-iterations` exhausted (exit 1)

**Symptom:** `stop_reason='tool_use' but produced no digest`.

**Why:** the model spent every round searching and never got to write.

**Fix:** the error message tells you the number to retry with. Raising it costs
money; a narrower topic usually works better.

### Traces stopped appearing

**Check in order:**

1. `NEWS_AGENT_DISABLE_TRACING` set? `tests/conftest.py` sets it, so a stray
   export from a shell where you ran pytest will silence a real run.
2. Keys in `.env`? The CLI prints `[Langfuse tracing enabled → ...]` or
   `disabled` on every run — read that line first.
3. **Import order.** `observability.py` decides whether to initialise Langfuse
   *at import time*, and `@observe` is applied at def-time. If `.env` loads
   after that import, the keys arrive too late and tracing silently stays off.
   `news_agent/__init__.py` loads `.env` first for exactly this reason, and a
   test asserts the order via AST. Do not reorder those imports.

**Telemetry never breaks a run.** Every adapter swallows exceptions. That is
also why they are tested — nothing else will ever tell you they stopped
working.

---

## Regular maintenance

| Cadence | Action | Why |
|---|---|---|
| Every review | `--capture-golden` | Ground truth is free at the point of use |
| ~20 fixtures | `--feedback` | Decide whether `--review` still earns its place |
| Before shipping a rubric edit | `--replay` | Exit 7 means revert |
| Monthly | Check `feeds_failed` in Langfuse | Dead feeds degrade quality invisibly |
| Monthly | Check `model_resolved` | Aliases move; `claude-haiku-4-5` → `...-20251001` |
| On a quality drop | Group scores by `prompt_hash` | Separates a prompt change from a model change |

---

## Environment

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | Billed per token |
| `OBSIDIAN_VAULT` | no | Overrides the default vault path |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | no | Tracing; absent = silent no-op |
| `LANGFUSE_BASE_URL` | no | Self-hosted Langfuse |
| `NEWS_AGENT_DISABLE_TRACING` | no | Force tracing off |
| `NEWS_AGENT_CACHE` | no | Rolling article window; default `%LOCALAPPDATA%\news-agent\articles.jsonl` |

**An API subscription does not cover this.** Claude Code and Claude.ai are
separate products; this bills your API account per token. `--check` verifies
key, model access and billing for about $0.00001.

---

## Windows notes

Two things bite specifically here:

**Console encoding.** Windows consoles default to cp1252, which cannot encode
`— → ≈ …` or non-Latin headlines. `_force_utf8()` reconfigures stdout/stderr
with `errors="replace"` — without it the CLI died with `UnicodeEncodeError` at
print time, *after* the API call had been paid for. It runs before
`parse_args` because `--help` exits from inside it.

**Filename safety.** Every vault filename derives from model output. `_slug`
strips characters Windows forbids and escapes reserved device names (`CON`,
`NUL`, `COM1`…); `_target` re-checks that the resolved path is still inside the
vault. A headline of `../../.ssh/authorized_keys` lands in the News folder as a
file rather than escaping.
