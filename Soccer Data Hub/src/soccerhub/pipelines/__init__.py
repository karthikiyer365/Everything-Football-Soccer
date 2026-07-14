"""Preset pipelines: arbitration layer over the raw readers."""
from soccerhub.manifest import Manifest
from soccerhub.pipelines.player_season import build_player_season
from soccerhub.pipelines.supa import push_to_supabase
from soccerhub.pipelines.xref import build_player_xref

__all__ = [
    "build_player_xref",
    "build_player_season",
    "push_to_supabase",
    "run_season",
]


def run_season(
    league: str, season: str, force: bool = False, table: str = "player_season"
) -> Manifest:
    """Phase A preset: xref -> merged player-season -> Supabase upsert."""
    build_player_xref(league, season, force=force)
    manifest = build_player_season(league, season, force=force)
    push_to_supabase(manifest, table)
    return manifest
