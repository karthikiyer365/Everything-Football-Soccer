# FootballScout Machine

_Product doc · Project #1 · folder: `Player Performance Analysis/`_

## Overview

A rudimentary football (soccer) scouting tool built on FIFA/EA player-attribute
data spanning 2015–2022, male and female. It turns 22 raw per-player attributes
into interactive evaluation views for a scout or team manager: outlier and
normality inspection of any attribute by position, and head-to-head player
comparison. The original prototype was framed as a multivariate regression scout
(player valuation + playmaking scores) developed with gameplay-derived attributes.

## Capabilities

- **ETL pipeline** — pulls per-year FIFA datasets by gender, selects 22 core
  attributes, explodes multi-position players into one row per position, adds
  `year`/`sex`, and imputes missing values into two canonical datasets
  (`Male_Players.csv`, `Female_Players.csv`).
- **Interactive Dash dashboard** (`localhost:8080`) — three tabs:
  - _Data Download_ — sample and export a chosen percentage of the data.
  - _Outlier / Normality view_ — boxplots before/after IQR outlier removal, plus
    Shapiro / Kolmogorov–Smirnov / D'Agostino normality tests with QQ plots.
  - _Player vs Player_ — radar (Scatterpolar) comparison across pace, dribbling,
    defending, physic, shooting, passing.
- **Static exploratory analysis** — PCA, Pearson correlation heatmaps, Box-Cox
  transforms, KDE / pair / joint / regression plots, and position, nationality
  and league-level distributions.

## Components

| File | Role |
|---|---|
| `player_data_generate.py` | ETL — fetch, clean, explode, impute → canonical CSVs |
| `player_static_analysis.py` | Dash dashboard app (3 tabs, callbacks) |
| `football_data_cleaner.py` | Offline EDA / statistics (matplotlib, seaborn, plotly) |
| `requirements.txt` | Runtime dependencies |

> Note: filenames vs content are historically swapped — `football_data_cleaner.py`
> is the EDA/stats script; `player_static_analysis.py` is the dashboard app.

## Data

- Male attributes retain `club_name`, `league_level`, `value_eur`; female data
  drops those three by design, so the two datasets have different schemas.
- `player_positions` is exploded, so a multi-position player appears in multiple
  rows — downstream counts double-count players.

## Running

```bash
cd "Player Performance Analysis"
pip install -r requirements.txt
python player_data_generate.py     # regenerate Male_Players.csv / Female_Players.csv
python player_static_analysis.py   # serve dashboard on localhost:8080
```

## Current status & known gaps

- Generated `Male_Players.csv` / `Female_Players.csv` are not committed (data is
  gitignored); they must be regenerated before the dashboard runs.
- The ETL source URL points at the old repo name and needs updating to the
  current repo before `player_data_generate.py` can fetch raw data.
- The regression/valuation scoring described in the original README is scope, not
  yet implemented in committed code.

## Roadmap link

Future data-science work (valuation models, similarity search, aging curves,
xG, dashboards) is tracked in the local `SOCCER_DS_BACKLOG.md`. The next project
(#2, _Soccer Data Hub_) provides the unified data-fetch layer those projects
build on — see `docs/plans/`.
