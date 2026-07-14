"""Player entity resolution: FBref names -> Transfermarkt ids."""
import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

from soccerhub.cache import cached_fetch
from soccerhub.manifest import Manifest
from soccerhub.readers.fbref import fetch_fbref_season
from soccerhub.readers.transfermarkt import fetch_transfermarkt_players

MAPPING_URL = (
    "https://raw.githubusercontent.com/JaseZiv/worldfootballR_data/"
    "master/raw-data/fbref-tm-player-mapping/output/fbref_to_tm_mapping.csv"
)

LEAGUE_TO_TM = {
    "ENG-Premier League": "GB1",
    "ESP-La Liga": "ES1",
    "GER-Bundesliga": "L1",
    "ITA-Serie A": "IT1",
    "FRA-Ligue 1": "FR1",
}

FUZZY_FLOOR = 0.85  # ponytail: single global threshold, tune per-league if noisy


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


def _load_mapping() -> pd.DataFrame:
    # upstream file is utf-8 with a few stray latin-1 bytes; strict utf-8
    # raises, latin-1 mojibakes every accent -> utf-8 + replace stray bytes
    return pd.read_csv(MAPPING_URL, encoding="utf-8", encoding_errors="replace")


def _tm_id_from_url(url: str):
    m = re.search(r"/spieler/(\d+)", str(url))
    return int(m.group(1)) if m else None


def _score(a: str, b: str) -> float:
    """Name similarity. Token-subset beats raw ratio: 'angel di maria' vs
    'angel di maria hernandez' is the same person (short name inside legal
    name) but SequenceMatcher alone scores it 0.74 — below the floor."""
    ta, tb = set(a.split()), set(b.split())
    if ta and tb and (ta <= tb or tb <= ta):
        return 0.95
    return SequenceMatcher(None, a, b).ratio()


def build_player_xref(league: str, season: str, force: bool = False) -> Manifest:
    """One row per FBref (player, team) in a league-season -> tm_id + method."""

    def produce():
        fbref = pd.read_parquet(fetch_fbref_season(league, season).path).reset_index()
        if isinstance(fbref.columns, pd.MultiIndex):
            # real fbref frames have ('Performance','Gls')-style columns;
            # we only need the index-derived flat ones (player/team/born)
            fbref.columns = [c[0] for c in fbref.columns]
        players = fbref[["player", "team"]].drop_duplicates()
        born = (
            fbref.groupby("player")["born"].first()
            if "born" in fbref.columns
            else pd.Series(dtype="object")
        )

        mapping = _load_mapping()
        mapping["tm_id"] = mapping["UrlTmarkt"].map(_tm_id_from_url)
        map_by_name = mapping.set_index("PlayerFBref")["tm_id"].to_dict()

        tm = pd.read_parquet(fetch_transfermarkt_players(LEAGUE_TO_TM[league]).path)
        tm["norm"] = tm["name"].map(normalize)
        exact_by_norm = tm.set_index("norm")["player_id"].to_dict()
        tm_year = pd.to_datetime(tm["date_of_birth"], errors="coerce").dt.year

        rows = []
        for _, r in players.iterrows():
            name = r["player"]
            # 1. mapping file
            tm_id = map_by_name.get(name)
            if tm_id is not None and pd.notna(tm_id):
                rows.append((name, r["team"], int(tm_id), "mapping_file", 1.0))
                continue
            # 2. exact normalized
            tm_id = exact_by_norm.get(normalize(name))
            if tm_id is not None:
                rows.append((name, r["team"], int(tm_id), "exact", 1.0))
                continue
            # 3. fuzzy within competition, birth-year guard
            best_ratio, best_id = 0.0, None
            b = born.get(name)
            for i, cand in enumerate(tm["norm"]):
                ratio = _score(normalize(name), cand)
                if ratio <= best_ratio:
                    continue
                ty = tm_year.iloc[i]
                if pd.notna(b) and pd.notna(ty) and abs(int(b) - int(ty)) > 1:
                    continue
                best_ratio, best_id = ratio, int(tm["player_id"].iloc[i])
            if best_ratio >= FUZZY_FLOOR:
                rows.append((name, r["team"], best_id, "fuzzy", round(best_ratio, 3)))
            else:
                rows.append((name, r["team"], None, "unmatched", 0.0))

        out = pd.DataFrame(
            rows, columns=["fbref_name", "team", "tm_id", "method", "confidence"]
        )
        out["tm_id"] = out["tm_id"].astype("Int64")
        return out

    return cached_fetch(
        "xref", "players", {"league": league, "season": season}, produce, force
    )
