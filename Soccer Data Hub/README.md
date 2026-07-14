# Soccer Data Hub (`soccerhub`)

Unified fetch layer for open-source soccer data. Fetch → cache to parquet →
return a manifest. Primary interface is the Python package; an MCP server wraps
the same functions for agent use.

## Install

    cd "Soccer Data Hub"
    pip install -e ".[dev]"

## Use (package)

    from soccerhub import fetch_fbref_season, fetch_statsbomb_events, fetch_transfermarkt_values

    m = fetch_fbref_season("ENG-Premier League", "2023")
    print(m.path, m.rows, m.cols)      # parquet path + shape

    import pandas as pd
    df = pd.read_parquet(m.path)        # projects read the cached parquet directly

Cache directory: `SOCCERHUB_CACHE` env var, default `./data`.

## Use (MCP server)

    python -m soccerhub.mcp_server

Tools: `fbref_season`, `statsbomb_events`, `transfermarkt_values` — each returns
a manifest dict.

## Readers (v1 tracer bullet)

| Function | Source | Returns |
|---|---|---|
| `fetch_fbref_season(league, season)` | FBref (soccerdata) | player season stats |
| `fetch_statsbomb_events(match_id)` | StatsBomb open data (kloppy) | match events |
| `fetch_transfermarkt_values(competition)` | Transfermarkt (pre-scraped) | player valuations |

Adding a source is one new file in `readers/` reusing `cached_fetch` — see
`docs/plans/2026-07-14-soccerhub-data-layer-design.md`.
