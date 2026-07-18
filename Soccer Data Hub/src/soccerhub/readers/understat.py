"""Understat xG data (plain HTTP, no browser) via soccerdata."""
import soccerdata as sd

from soccerhub.cache import cached_fetch
from soccerhub.manifest import Manifest
from soccerhub.readers.fbref import _season_to_code

DATASETS = {
    "player_season": "read_player_season_stats",
    "team_match": "read_team_match_stats",
    "shots": "read_shot_events",
}


def fetch_understat(league: str, season: str, dataset: str = "player_season",
                    force: bool = False) -> Manifest:
    """One league-season of Understat data. Coverage: Big-5 from 2014.

    ``dataset``: player_season (per-player xG/xA totals), team_match
    (per-match team xG, PPDA, expected points), shots (every shot with
    coordinates — one page per match upstream, so much slower to fetch).
    """
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {sorted(DATASETS)}")

    def produce():
        u = sd.Understat(leagues=league, seasons=_season_to_code(season))
        return getattr(u, DATASETS[dataset])().reset_index()

    return cached_fetch(
        "understat", dataset, {"league": league, "season": season}, produce, force
    )
