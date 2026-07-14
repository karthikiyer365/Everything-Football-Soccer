import soccerdata as sd

from soccerhub.cache import cached_fetch
from soccerhub.manifest import Manifest


def fetch_fbref_season(league: str, season: str, force: bool = False) -> Manifest:
    """Player season stats for one league-season from FBref."""

    def produce():
        return sd.FBref(leagues=league, seasons=season).read_player_season_stats()

    return cached_fetch(
        "fbref", "player_season", {"league": league, "season": season}, produce, force
    )
