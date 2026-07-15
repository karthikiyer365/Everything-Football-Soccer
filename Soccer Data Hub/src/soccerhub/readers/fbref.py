import json
import os
from pathlib import Path

from soccerhub.cache import cached_fetch
from soccerhub.manifest import Manifest

# FBref's comps index gender-suffixes any league that has a same-named
# women's competition ("Serie A (M)" / "Serie A (W)"); soccerdata 1.9.0
# still looks for plain "Serie A" and silently returns zero leagues.
# soccerdata merges ~/soccerdata/config/league_dict.json OVER its built-ins
# at import time, so we write the corrected entry there before importing.
_SERIE_A_ENTRY = {
    "ClubElo": "ITA_1",
    "MatchHistory": "I1",
    "FBref": "Serie A (M)",
    "ESPN": "ita.1",
    "Sofascore": "Serie A",
    "SoFIFA": "[Italy] Serie A",
    "Understat": "Serie A",
    "WhoScored": "Italy - Serie A",
    "season_start": "Aug",
    "season_end": "May",
}


def _patch_league_config() -> None:
    cfg_dir = (
        Path(os.environ.get("SOCCERDATA_DIR", Path.home() / "soccerdata")) / "config"
    )
    cfg = cfg_dir / "league_dict.json"
    data = json.loads(cfg.read_text()) if cfg.exists() else {}
    if data.get("ITA-Serie A", {}).get("FBref") != _SERIE_A_ENTRY["FBref"]:
        data["ITA-Serie A"] = _SERIE_A_ENTRY
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps(data, indent=2))


_patch_league_config()

import soccerdata as sd  # noqa: E402  — config file must exist before this import


def fetch_fbref_season(league: str, season: str, force: bool = False) -> Manifest:
    """Player season stats for one league-season from FBref."""

    def produce():
        return sd.FBref(leagues=league, seasons=season).read_player_season_stats()

    return cached_fetch(
        "fbref", "player_season", {"league": league, "season": season}, produce, force
    )
