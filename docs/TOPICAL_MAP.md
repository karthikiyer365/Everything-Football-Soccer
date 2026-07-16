# Topical Map

Generated 2026-07-16 at `99fb72e` · `⟶` = dependency edge · `▓ planned ▓` = claimed in
docs/UI but absent from code · regenerate stale sections, don't patch prose.

---

## MAP A — Product Topology

```
                       ┌────────────────────────┐
                       │ A0 Landing (pitch)     │
                       │ site/index.html        │
                       └───────────┬────────────┘
              ┌────────────────────┼──────────────────────┐
              v                    v                       v
┌──────────────────────┐ ┌─────────────────────┐ ┌──────────────────────┐
│ A1 Player Dashboard  │ │ ▓ A2 player screens │ │ ▓ A3 market screens  │
│ site/player.html     │ │ scouting, value-vs- │ │ transfer market,     │
└──────────┬───────────┘ │ output              │ │ inflation, age curves│
           │             └─────────────────────┘ └──────────────────────┘
           v                        ┌──────────────────────┐
┌──────────────────────┐            │ ▓ A4 event screens   │
│ Supabase (read, anon)│            │ match center, shot   │
└──────────────────────┘            │ maps, xG studio      │
                                    └──────────────────────┘
┌──────────────────────────┐   ┌───────────────────────────┐
│ A5 FootballScout Machine │   │ A6 MCP server (agents)    │
│ legacy, standalone       │   │ soccerhub/mcp_server.py   │
└──────────────────────────┘   └───────────────────────────┘
```

**[A0] Landing page**
- What: pitch-themed hub; formation grid of 9 dashboard cards — 1 live, 8 `▓ planned ▓`
  ("In training" / "Next window" in the UI).
- Where: `site/index.html` (167 LOC, static — no JS beyond SMIL ball animation).
- Docs: none dedicated; deployed by `.github/workflows/deploy-pages.yml`.
- Edges: ⟶ A1 (only live link: `href="player.html"` ×2).
- Drift: footer claims data "refreshed twice weekly" — true only for EPL player stats
  (see C2/C3); transfers and the other four leagues have no scheduled refresh.

---

**[A1] Player Dashboard**
- What: player search → career page: market-value chart (transfer + synthetic club-change
  markers), G+A chart with minutes and G+A/90 overlay lines, transfer history table,
  season log.
- Where: `site/player.html` (429 LOC, single file, inline JS; functions `hub()`,
  `runSearch()`, `load()`, `render()`, `valueChart()`, `gaChart()`, `wireTooltips()`).
- Docs: none.
- Edges: ⟶ Supabase PostgREST (anon key, RLS select-only) reading `player_season`,
  `transfers`; ⟵ A0 (nav).

---

**[▓ A2] Player screens — Scouting Screens, Value vs Output**
- What: moneyball filters; production-vs-price views. UI cards only, zero code.
- Where: `site/index.html:114-123` (cards marked `pos soon`).
- Edges: would read same `player_season` table as A1.

---

**[▓ A3] Market screens — Transfer Market, League Inflation, Age Curves**
- What: fees/moves explorer, league value growth, positional age arcs. Cards only.
- Where: `site/index.html:126-140`.
- Edges: would read `transfers` + `player_season`; blocked on C3 (transfers refresh gap).

---

**[▓ A4] Event screens — Match Center, Shot Maps, xG Studio**
- What: fixtures/results, event-level shot data, xG-vs-goals. Cards only.
- Where: `site/index.html:143-157`.
- Edges: would need schedule data (not ingested) and StatsBomb events (B1 reader exists,
  nothing downstream of it).

---

**[A5] FootballScout Machine (legacy)**
- What: FIFA-attribute scouting toolkit 2015–2022 — ETL to canonical CSVs, 3-tab Dash
  app on localhost:8080, offline EDA/statistics scripts.
- Where: `Player Performance Analysis/` — `player_data_generate.py` (ETL, 124 LOC),
  `player_static_analysis.py` (Dash app, 422 LOC), `football_data_cleaner.py`
  (EDA, 469 LOC).
- Docs: `docs/product/football-scout-machine.md` (notes the filename/content swap:
  cleaner = EDA script, static_analysis = dashboard).
- Edges: none — reads FIFA CSVs from a GitHub raw URL; shares nothing with soccerhub.

---

**[A6] MCP server (agent surface)**
- What: FastMCP wrapper exposing three reader tools — `fbref_season`,
  `statsbomb_events`, `transfermarkt_values` — each returns a manifest dict.
- Where: `Soccer Data Hub/src/soccerhub/mcp_server.py` (31 LOC).
- Docs: `Soccer Data Hub/README.md` "Use (MCP server)".
- Edges: ⟶ B1 readers only. Does NOT expose `read_hub` — agents can't reach the
  Supabase source of truth through MCP yet.

---

## MAP B — Developer Topology

```
┌─────────────────────────── soccerhub package ("Soccer Data Hub/src") ─────────────┐
│                                                                                   │
│  ┌──────────────────┐      ┌─────────────────────────────────────────┐            │
│  │ B0 cache core    │<─────│ B1 readers: fbref / transfermarkt /     │            │
│  │ cache, manifest, │      │ statsbomb (fetch → parquet → Manifest)  │            │
│  │ errors           │      └───────────────┬─────────────────────────┘            │
│  └──────────────────┘                      │                                      │
│                          ┌─────────────────┴───────────┐                          │
│                          v                             v                          │
│               ┌────────────────────┐        ┌─────────────────────┐               │
│               │ B2 xref pipeline   │───────>│ B3 player_season    │               │
│               │ (entity resolution)│  ids   │ (merge + clean())   │               │
│               └─────────┬──────────┘        └──────────┬──────────┘               │
│                         │       ┌──────────────────────┤                          │
│                         v       v                      v                          │
│               ┌────────────────────┐        ┌─────────────────────┐               │
│               │ B6 run_season      │───────>│ B4 supa (upsert,    │               │
│               │ orchestrator       │        │ service role)       │               │
│               └────────────────────┘        └──────────┬──────────┘               │
│                                                        v                          │
│  ┌────────────────────┐                     ┌─────────────────────┐               │
│  │ B5 query.read_hub  │<────────────────────│   Supabase Postgres │               │
│  │ (anon, paginated)  │                     │   B7 migrations     │               │
│  └────────────────────┘                     └──────────┬──────────┘               │
└────────────────────────────────────────────────────────┼──────────────────────────┘
                                                         v
   ┌──────────────┐  ┌──────────────────┐     ┌────────────────────┐
   │ B8 CI: cron  │  │ B9 tests (13     │     │ B10 site frontend  │
   │ + pages      │  │ files, 817 LOC)  │     │ (inline JS, no     │
   │ deploy       │  └──────────────────┘     │ build step)        │
   └──────────────┘                           └────────────────────┘
```

**[B0] Cache core**
- What: content-addressed parquet cache + fetch memoizer; every reader routes through
  `cached_fetch(source, dataset, params, produce, force)`.
- Where: `soccerhub/cache.py` (76 LOC), `manifest.py` (36 — `Manifest` dataclass,
  `infer_date_range`), `errors.py` (2 — `SoccerhubError`).
- Docs: `Soccer Data Hub/README.md` (cache dir: `SOCCERHUB_CACHE`, default `./data`).
- Edges: ⟵ B1, B2, B3 (all builds are `cached_fetch` closures).

---

**[B1] Readers**
- What: source fetchers, one file per source; return `Manifest`, never DataFrames.
- Where: `soccerhub/readers/fbref.py` (67 LOC — `_patch_league_config()` writes
  soccerdata's `league_dict.json` before import; `_season_to_code()` disambiguates
  "2021"→"2122"), `transfermarkt.py` (93 — players/transfers/values from
  transfermarkt-datasets), `statsbomb.py` (15 — events via kloppy).
- Docs: hub README "Use (package)".
- Edges: ⟶ B0; ⟵ B2/B3 (`refetch` param forwards as `force`), ⟵ A6, ⟵ B11.
- Drift: `fetch_transfermarkt_transfers` (transfermarkt.py:55) has **no downstream
  pipeline** — `grep transfers pipelines/` = 0 hits. The `transfers` table was loaded
  by a one-off script since deleted. See C3.

---

**[B2] xref pipeline (entity resolution)**
- What: FBref (player, team) → Transfermarkt id; ladder OVERRIDES → mapping_file →
  exact → fuzzy, every rung birth-year-guarded; token blocking for fuzzy speed.
- Where: `soccerhub/pipelines/xref.py` (174 LOC) — `build_player_xref(league, season,
  force, refetch)`, `normalize()`, `_score()`.
- Docs: `docs/plans/2026-07-14-soccerhub-implementation-plan.md` (pre-dates OVERRIDES
  and the mononym rule — stale on matching details).
- Edges: ⟶ B1 (fbref + TM registry), ⟶ B0; ⟵ B6.

---

**[B3] player_season pipeline**
- What: flatten FBref stats, join xref ids + TM values, disambiguate same-name-same-team,
  then `clean()`: rate stats nulled < 450 min, age from birth_year, primary_position,
  value_is_stale flag.
- Where: `soccerhub/pipelines/player_season.py` (150 LOC) — `build_player_season()`,
  `flatten_fbref()`, `clean()`, `season_end()`.
- Docs: same plan doc as B2 (clean() rules only exist in code + tests).
- Edges: ⟶ B1, B2 output parquet, B0; ⟵ B6.

---

**[B4] Supabase write path**
- What: chunked upserts (500/req) with float→Int64 repair; service-role key, bypasses
  RLS. `CONFLICT_KEY = "league,season,team,player_name"`.
- Where: `soccerhub/pipelines/supa.py` (36 LOC) — `push_to_supabase()`, `upsert_df()`.
- Edges: ⟵ B6 (`push_to_supabase`, `push_xref`); ⟶ Supabase.
- Note: upsert can't delete — re-labeled seasons leave orphans; manual `DELETE` needed
  (happened once for the mislabeled 2021 season).

---

**[B5] Supabase read path**
- What: `read_hub(table, select, **eq_filters)` — paginated PostgREST reads (1000/page),
  anon key, RLS select-only. The API for all downstream analysis phases.
- Where: `soccerhub/pipelines/query.py` (36 LOC).
- Edges: ⟵ future analysis code; the site (B10) duplicates this logic in JS (`hub()`).

---

**[B6] run_season orchestrator**
- What: the cron preset: xref (refetch) → push_xref → player_season (refetch) → upsert.
  `force=True` means re-download sources, not just re-merge.
- Where: `soccerhub/pipelines/__init__.py` (46 LOC) — `run_season()`, `push_xref()`,
  `XREF_CONFLICT_KEY`.
- Edges: ⟶ B2, B3, B4; ⟵ B8 cron.

---

**[B7] Migrations (manual apply)**
- What: 4 SQL files — schema + RLS anon-read policies. **No migration runner**: user
  applies each in the Supabase SQL editor by hand; files are the record, not the tool.
- Where: `Soccer Data Hub/supabase/migrations/0001_player_season.sql … 0004_player_xref.sql`.
- Edges: defines tables B4 writes and B5/B10 read.

---

**[B8] CI / automation**
- What: two workflows. `run-season.yml`: cron Mon+Thu 06:00 UTC → `run_season(LEAGUE,
  SEASON, force=True)`; scheduled runs have no inputs so env defaults apply —
  **EPL 2025 only**. `deploy-pages.yml`: push to main touching `site/**` → GitHub Pages.
- Where: `.github/workflows/run-season.yml`, `.github/workflows/deploy-pages.yml`.
- Drift: ESP/GER/ITA/FRA are backfilled but never auto-refreshed; current-season rows
  for those leagues go stale between manual `workflow_dispatch` runs.

---

**[B9] Tests**
- What: pytest, 13 files / 817 LOC, all monkeypatch-based (no network). Cover season
  codes, xref ladder, disambiguation, clean() rules, upsert chunking, read_hub
  pagination, run_season refetch propagation, public exports.
- Where: `Soccer Data Hub/tests/test_*.py`.
- Edges: ⟶ every B0–B6 unit.

---

**[B10] Site frontend**
- What: two static HTML files, inline CSS/JS, zero dependencies and no build step.
  Talks straight to PostgREST with the publishable anon key (safe: RLS select-only).
- Where: `site/index.html` (167 LOC), `site/player.html` (429 LOC).
- Edges: ⟶ Supabase REST; deployed by B8. Duplicates B5's pagination/read logic in JS —
  intentional (no shared runtime between Python and browser).

---

**[B11] preview_app (dev toy)**
- What: self-described "throwaway local preview" — stdlib HTTP server rendering reader
  output as HTML tables on :8765. Superseded by site/ for anything user-facing.
- Where: `Soccer Data Hub/preview_app.py` (135 LOC, outside `src/`).
- Edges: ⟶ B1 readers only.

---

**[B12] Legacy: Player Performance Analysis**
- What: pre-soccerhub standalone project (FIFA CSVs, Dash). No imports in or out of
  the soccerhub package; frozen since 2024.
- Where: `Player Performance Analysis/` (1,015 LOC across 3 scripts).
- Docs: `docs/product/football-scout-machine.md`, root `README.md`.
- Drift: root `README.md` describes ONLY this project — Soccer Data Hub, the site and
  the workflows are absent from it entirely.

---

## MAP C — Per-feature backend processing

**C1 · Player dashboard read path (live)**

```
player.html search box
  └──> runSearch(q)  ── 300ms debounce ──> hub("player_season",
                                            {player_name: "ilike.*q*"})
         │                                  GET {SUPABASE_URL}/rest/v1/… apikey=anon
         v
       load()  ── ?tm=<id> or ?name= ──> hub("player_season", …)   [career rows]
         │                          └──> hub("transfers", {tm_id: eq})  [moves]
         v
       render(rows, moves)
         ├──> valueChart(rows, moves)   amber line; dashed markers = recorded
         │                              transfers, dotted = synthetic club changes
         ├──> gaChart(rows)             green bars + minutes / G+A/90 overlays
         ├──> transfers table           fee null -> "free / undisclosed"
         └──> season log                per-90 "—" under 450 min; stale-value ⚠
```
No writes. Anon key only; RLS blocks everything but SELECT.

---

**C2 · Cron season refresh (write path)**

```
run-season.yml (cron Mon+Thu / workflow_dispatch)
  └──> run_season(league, season, force=True)          pipelines/__init__.py:34
         ├──> build_player_xref(refetch=True)          pipelines/xref.py:69
         │      ├──> fetch_fbref_season(force=True)    readers/fbref.py:54
         │      ├──> fetch_transfermarkt_players(force=True)
         │      └──> ladder: OVERRIDES → mapping → exact → fuzzy (year-guarded)
         ├──> push_xref(m, league, season)             upsert player_xref
         │                                             key: league,season,team,fbref_name
         ├──> build_player_season(refetch=True)        pipelines/player_season.py:81
         │      ├──> flatten_fbref() → join xref → join TM values
         │      ├──> same-name disambiguation (birth-year suffix)
         │      └──> clean()  rates<450min→null · age · primary_position · stale flag
         └──> push_to_supabase(m, "player_season")     upsert, 500-row chunks
                                                       key: league,season,team,player_name
```
Side effects live in app code only — no DB triggers. Upsert never deletes (orphan risk
on key changes).

---

**C3 · Transfers load — ⚠ manual gap**

```
fetch_transfermarkt_transfers()        readers/transfermarkt.py:55
  └──> (rename player_id→tm_id, drop future-dated placeholders, dedup)
         └──> ▓ no pipeline ▓  one-off script (deleted) pushed to `transfers` via
              upsert_df; run_season never touches it → table is a snapshot, not a feed
```
Fix when needed: one `push_transfers()` preset in `pipelines/__init__.py` + a line in
the cron. Blocks A3 (Transfer Market screen) from being trustworthy.

---

**C4 · Landing/site deploy**

```
git push main (paths site/**) ──> deploy-pages.yml ──> configure-pages
  └──> upload-pages-artifact(site/) ──> deploy-pages ──> github.io
```
One-time repo setting required (Pages source = GitHub Actions) — done 2026-07-16.

---

**C5 · Legacy FIFA ETL (frozen)**

```
player_data_generate.py ──> FIFA CSVs from raw.githubusercontent ──> clean/explode/
impute ──> Male_Players.csv / Female_Players.csv (local) ──> player_static_analysis.py
(Dash :8080) / football_data_cleaner.py (EDA)
```

---

## Cross-map bridges

| Product node | Runs on | Key seam to check when touching it |
|---|---|---|
| A0 Landing | B10, B8 (pages) | card stats hard-coded (49,692 / 35,123) — restate after big loads |
| A1 Player Dashboard | B10 ⟶ B7 tables ⟵ C2 | JS `hub()` vs Python `read_hub()` — same PostgREST semantics, changed separately |
| A1 value markers | C2 + C3 | synthetic markers assume `transfers` may be stale — stays true until C3 is wired |
| ▓ A2/A3 screens | B5 or B10, C3 | A3 needs transfers refresh (C3) first |
| ▓ A4 screens | B1 statsbomb (unused) | no schedule/event ingestion exists yet |
| A6 MCP server | B1 only | does not expose read_hub — agents bypass source of truth |
| A5 legacy | nothing shared | safe to archive; only root README references it |
