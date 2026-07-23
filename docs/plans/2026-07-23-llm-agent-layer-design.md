# Soccer Data Hub — LLM Agent Layer (Layer 3) — Design Spec

_Plan · Project #2 addendum · created 2026-07-23_

## Overview

`soccerhub`'s design doc (`2026-07-14-soccerhub-data-layer-design.md`) scoped a
three-layer architecture and explicitly deferred Layer 3 ("LLM agent over MCP")
as future work. Layer 1 (readers/pipelines) and Layer 2 (`mcp_server.py`) are
built and in production use (player card xG, hub queries). This spec adds
Layer 3: a Gemini-backed agent, `soccerhub.agent.ask()`, that answers
natural-language questions over hub data, interprets charts (data + rendered
image), and generates summaries — by calling the *existing* MCP tools, no new
data pipeline.

## Goals

- One function, `ask(prompt, images=None) -> str`, covering all three use
  cases (data query, chart interpretation, summary generation) — they're the
  same shape (text/image in, model decides whether to call a tool, text out).
- Zero duplication of tool logic: the agent calls `mcp_server.py`'s tools as
  an MCP client, it does not re-import Layer 1 or redefine schemas.
- Backend-first: importable/CLI-runnable now; site wiring (`player.html`
  button → HTTP endpoint → `ask()`) is explicitly out of scope for this spec.

## Non-goals (explicit YAGNI)

- HTTP endpoint / web wiring for `site/*.html`.
- Conversation memory or multi-turn session state across separate `ask()` calls.
- Streaming responses.
- Non-Gemini providers / fallback routing.
- Renaming existing MCP tool names (`hub_table`, `fbref_season`, etc.) — Layer 2
  is shipped; renaming breaks nothing to fix, so it stays snake_case as-is.

## Architecture

```
Layer 1  soccerhub readers/pipelines        ← unchanged
Layer 2  mcp_server.py (FastMCP, stdio)     ← unchanged tools, adds annotations
                                               + ToolError wrapping (this spec)
Layer 3  soccerhub/agent.py  ask()          ← NEW: Gemini + MCP client
```

`ask()` spawns `mcp_server.py` as a stdio subprocess, opens an
`mcp.ClientSession`, and passes that session directly into
`google-genai`'s `generate_content(tools=[session])`. The SDK's built-in MCP
support handles `tools/list` discovery, schema translation, and the
`FunctionCall` ↔ `tools/call` round-trip — no hand-written tool schemas.

## Data flow

```
CALLER
  ask(prompt, images=[])
        │
        ▼
  build contents: [text Part(prompt), image Part(png)...]   ← only if images passed
        │
        ▼
  spawn mcp_server.py (stdio subprocess) ──► mcp.ClientSession
        │                                         │
        │                              lists 4 registered tools:
        │                              hub_table / fbref_season /
        │                              statsbomb_events / transfermarkt_values
        ▼
  genai.Client.generate_content(model, contents, tools=[session])
        │
        ▼
  Gemini decides: answer directly, or call a tool? ──┐
        │                                            │
   NO TOOL NEEDED                              TOOL CALL NEEDED
   (chart/data interpretation,                  (data-hub query, e.g.
    summary from data in prompt)                 "compare these 2 wingers")
        │                                            │
        │                                            ▼
        │                              MCP routes call to matching tool fn
        │                                            │
        │                       ┌────────────────────┼─────────────────────┐
        │                       ▼                    ▼                     ▼
        │                 hub_table(...)      fbref_season(...)   statsbomb_events(...)
        │                 read_hub()           fetch_fbref_season() fetch_statsbomb_events()
        │                       │                    │                     │
        │                 Supabase REST        Layer-1 reader        Layer-1 reader
        │                 (source of truth)    cache_key() lookup    cache_key() lookup
        │                       │              ┌─────┴─────┐         ┌─────┴─────┐
        │                       │           HIT│           │MISS  HIT│           │MISS
        │                       │              ▼           ▼         ▼           ▼
        │                       │        read parquet  soccerdata  read parquet kloppy
        │                       │        (instant)     call+cache  (instant)    call+cache
        │                       │              │           │         │           │
        │                       │              └─────┬─────┘         └─────┬─────┘
        │                       │                     ▼                    ▼
        │                       │                Manifest/dict      Manifest/dict
        │                       └──────────┬──────────┴────────────────────┘
        │                                  ▼
        │                       tool result returned to Gemini
        │                                  │
        │                                  ▼
        │                       Gemini reads result → may call
        │                       another tool, or produce final answer
        │                                  │
        └──────────────────┬───────────────┘
                            ▼
                  final text response
                            │
                            ▼
                       return str

ERROR PATHS (either branch):
  tool raises SoccerhubError → mcp_server.py re-raises as ToolError(str(e))
      → surfaced to Gemini as an actionable message (not masked), model may
      retry with corrected args or explain the failure to the caller
  MCP subprocess dies / transport error → ask() raises SoccerhubError
  Gemini API error (rate limit / auth) → ask() raises SoccerhubError
  cache write mid-fetch fails → existing Layer-1 atomic-write guarantee holds
      (unchanged from Layer 1 design, no poisoned cache)
```

## Interface

```python
# src/soccerhub/agent.py
def ask(
    prompt: str,
    images: list[bytes] | None = None,
    model: str = "gemini-3-flash-preview",
) -> str:
    """Ask the soccerhub agent a question. Calls MCP tools as needed."""
```

- `images`: optional list of PNG bytes (rendered chart screenshots), sent as
  multimodal `Part`s alongside the text prompt — Gemini reads both the
  underlying data (passed in `prompt`) and the visual pattern in the image.
- Raises `SoccerhubError` on any failure (API, transport, or an unrecovered
  tool error) — same convention as Layer 1 readers.
- One function covers all three use cases; which one happens is a function of
  what the caller puts in `prompt`/`images`, not a code branch.

## MCP server changes (`mcp_server.py`)

Two additions to the 4 existing tools, no behavior change to their logic:

1. **Tool annotations** — all 4 tools are pure reads:
   ```python
   from mcp.types import ToolAnnotations

   @mcp.tool(annotations=ToolAnnotations(
       readOnlyHint=True, destructiveHint=False, openWorldHint=False))
   def hub_table(...):   # reads own Supabase — closed world

   @mcp.tool(annotations=ToolAnnotations(
       readOnlyHint=True, destructiveHint=False, openWorldHint=True))
   def fbref_season(...):        # scrapes external site — open world
   def statsbomb_events(...):    # same
   def transfermarkt_values(...): # same
   ```
2. **Error wrapping** — catch `SoccerhubError`, re-raise as
   `mcp.server.fastmcp.exceptions.ToolError` so Gemini gets an actionable
   message instead of FastMCP's default masked generic error:
   ```python
   from mcp.server.fastmcp.exceptions import ToolError

   @mcp.tool(annotations=...)
   def statsbomb_events(match_id: str) -> dict:
       try:
           return asdict(fetch_statsbomb_events(match_id))
       except SoccerhubError as e:
           raise ToolError(str(e)) from e
   ```

Confirmed against installed `mcp==1.28.1`: `FastMCP.tool()` accepts
`annotations: ToolAnnotations`, and `ToolError` is importable — no version
bump needed for this.

## Config & dependencies

- New env var `GEMINI_API_KEY`, same pattern as existing `SUPABASE_*` vars
  (exported in shell / sourced `.env`, no dotenv library — matches repo
  convention, see `pipelines/supa.py`).
- `pyproject.toml`: add `google-genai>=1.9` (first version with stable MCP
  `ClientSession` passthrough).

## Testing

One smoke test, `tests/test_agent.py`, matching the repo's "one test per unit"
pattern (`tests/test_cache.py`, `tests/test_understat.py`): mock the
`genai.Client` and `mcp.ClientSession`, assert `ask()` builds the correct
`Part` list (text, or text+image) and returns the mocked model's text. No real
API calls in tests — this is a wiring test, not an eval of Gemini's answers.

## Key decisions

| Concern | Decision | Rationale |
|---|---|---|
| LLM provider | Google Gemini free tier | AWS Bedrock / Azure OpenAI have no real free inference tier (trial credits only); Gemini free tier + native multimodal covers chart-image interpretation |
| Tool wiring | MCP client against existing `mcp_server.py` | Matches original Layer-3-over-MCP design; zero duplicate tool schemas; same server also usable by Claude Desktop/other MCP clients later |
| Interface shape | One function `ask(prompt, images=None)` | All three use cases (query/chart/summary) are the same text-and-optional-image-in, text-out shape — no reason to branch in code |
| Chart interpretation input | Both structured data (in prompt) and rendered screenshot (image) | User wants the model reading numbers and visual pattern, not one or the other |
| MCP tool naming | Unchanged (`hub_table`, `fbref_season`, ...) | Already shipped; renaming has no functional benefit and risks breaking other MCP clients |
| New code naming | snake_case, prefer one-word names (`ask`) | Matches repo's PEP8 convention; user asked for concise naming, not camelCase |
| Site wiring | Deferred | User chose backend-first; needs a real backend host since `site/*.html` is static + direct Supabase calls today, no server to add an endpoint to yet |
| Tool annotations / error wrapping | Add to `mcp_server.py` now | Prerequisite for a *model* being the caller instead of a human — masked/vague errors block the model's ability to self-correct (e.g. retry a bad `match_id`) |
