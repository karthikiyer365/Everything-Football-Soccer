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
    "read_hub",
    "run_season",
]

XREF_CONFLICT_KEY = "league,season,team,fbref_name"


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
    """Phase A preset: xref -> merged player-season -> Supabase upsert."""
    mx = build_player_xref(league, season, force=force)
    push_xref(mx, league, season)
    manifest = build_player_season(league, season, force=force)
    push_to_supabase(manifest, table)
    return manifest
