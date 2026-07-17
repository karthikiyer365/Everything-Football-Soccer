"""Preset pipelines: arbitration layer over the raw readers.

Supabase is the source of truth; local parquet manifests are build cache.
"""
from soccerhub.manifest import Manifest
from soccerhub.pipelines.player_season import build_player_season
from soccerhub.pipelines.query import read_hub
from soccerhub.pipelines.supa import push_to_supabase
from soccerhub.pipelines.xref import build_player_xref

__all__ = [
    "build_player_xref",
    "build_player_season",
    "push_to_supabase",
    "push_transfers",
    "push_age_curve",
    "read_hub",
    "run_season",
]

XREF_CONFLICT_KEY = "league,season,team,fbref_name"
TRANSFERS_CONFLICT_KEY = "tm_id,transfer_date"  # matches 0002 PK
AGE_CURVE_CONFLICT_KEY = "primary_position,age"  # matches 0005 PK


def push_transfers(force: bool = False) -> int:
    """Transfers snapshot -> transfers table. Cron runs this so the table
    stays a feed, not a one-off load."""
    import pandas as pd

    from soccerhub.pipelines.supa import upsert_df
    from soccerhub.readers.transfermarkt import fetch_transfermarkt_transfers

    m = fetch_transfermarkt_transfers(force=force)
    return upsert_df(pd.read_parquet(m.path), "transfers", TRANSFERS_CONFLICT_KEY)


def push_age_curve(min_minutes: int = 450, min_n: int = 30) -> int:
    """player_season -> avg market value by (position, age) for the dashboard
    overlay. Derived from the hub itself (PostgREST aggregates are disabled),
    so cron must run it after the season jobs."""
    from soccerhub.pipelines.query import read_hub
    from soccerhub.pipelines.supa import upsert_df

    df = read_hub(
        "player_season",
        select="primary_position,age,market_value_in_eur,minutes",
    )
    df = df[
        (df["minutes"].fillna(0) >= min_minutes)
        & df["market_value_in_eur"].notna()
        & df["age"].notna()
        & df["primary_position"].notna()
    ]
    g = (
        df.groupby(["primary_position", "age"])["market_value_in_eur"]
        .agg(avg_value_eur="mean", n="count")
        .reset_index()
    )
    g = g[g["n"] >= min_n]  # thin age buckets (16, 40+) make a jumpy curve
    g["avg_value_eur"] = g["avg_value_eur"].round().astype("Int64")
    g["age"] = g["age"].astype(int)
    return upsert_df(g, "age_curve", AGE_CURVE_CONFLICT_KEY)


def push_xref(manifest: Manifest, league: str, season: str) -> int:
    """Xref parquet -> player_xref table (adds the league/season key cols)."""
    import pandas as pd

    from soccerhub.pipelines.supa import upsert_df

    df = pd.read_parquet(manifest.path)
    df.insert(0, "league", league)
    df.insert(1, "season", season)
    return upsert_df(df, "player_xref", XREF_CONFLICT_KEY)


def run_season(
    league: str, season: str, force: bool = False, table: str = "player_season"
) -> Manifest:
    """Phase A preset: xref -> merged player-season -> Supabase upsert.

    force here means "refresh this season": sources are re-downloaded, not
    just re-merged — the cron's whole point is new fbref numbers.
    """
    mx = build_player_xref(league, season, force=force, refetch=force)
    push_xref(mx, league, season)
    manifest = build_player_season(league, season, force=force, refetch=force)
    push_to_supabase(manifest, table)
    return manifest
