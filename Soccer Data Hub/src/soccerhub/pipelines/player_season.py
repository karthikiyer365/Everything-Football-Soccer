"""Merged player-season table: FBref stats + Transfermarkt market value."""
import pandas as pd

from soccerhub.cache import cached_fetch
from soccerhub.manifest import Manifest
from soccerhub.pipelines.xref import LEAGUE_TO_TM, build_player_xref
from soccerhub.readers.fbref import fetch_fbref_season
from soccerhub.readers.transfermarkt import fetch_transfermarkt_values

# FBref ('group', 'stat') -> canonical DB column. One namespace forever;
# Understat/ClubElo columns land here later, not in source jargon.
CANON = {
    ("nation", ""): "nationality",
    ("pos", ""): "position",
    ("age", ""): "age",
    ("born", ""): "birth_year",
    ("Playing Time", "MP"): "matches_played",
    ("Playing Time", "Starts"): "starts",
    ("Playing Time", "Min"): "minutes",
    ("Playing Time", "90s"): "nineties",
    ("Performance", "Gls"): "goals",
    ("Performance", "Ast"): "assists",
    ("Performance", "G+A"): "goals_assists",
    ("Performance", "G-PK"): "non_penalty_goals",
    ("Performance", "PK"): "penalties_scored",
    ("Performance", "PKatt"): "penalties_attempted",
    ("Performance", "CrdY"): "yellow_cards",
    ("Performance", "CrdR"): "red_cards",
    ("Per 90 Minutes", "Gls"): "goals_per90",
    ("Per 90 Minutes", "Ast"): "assists_per90",
    ("Per 90 Minutes", "G+A"): "goals_assists_per90",
    ("Per 90 Minutes", "G-PK"): "non_penalty_goals_per90",
    ("Per 90 Minutes", "G+A-PK"): "non_penalty_goals_assists_per90",
}


def flatten_fbref(df: pd.DataFrame) -> pd.DataFrame:
    """MultiIndex fbref frame -> flat frame with canonical column names."""
    flat = df.copy()
    flat.columns = [
        CANON.get(tuple(c), "_".join(filter(None, c)).lower()) for c in flat.columns
    ]
    flat = flat.reset_index().rename(columns={"player": "player_name"})
    return flat


def season_end(season: str) -> str:
    """'2023' (2023-24 season) -> '2024-06-30'."""
    return f"{int(season) + 1}-06-30"


def build_player_season(league: str, season: str, force: bool = False) -> Manifest:
    """FBref season stats + latest market value on/before season end."""

    def produce():
        stats = flatten_fbref(pd.read_parquet(fetch_fbref_season(league, season).path))
        stats["season"] = season  # canonical start-year label, not fbref's '2324'

        xref = pd.read_parquet(build_player_xref(league, season).path).rename(
            columns={"method": "xref_method", "confidence": "xref_confidence"}
        )
        merged = stats.merge(
            xref,
            left_on=["player_name", "team"],
            right_on=["fbref_name", "team"],
            how="left",
        ).drop(columns=["fbref_name"])
        merged["xref_method"] = merged["xref_method"].fillna("unmatched")

        vals = pd.read_parquet(fetch_transfermarkt_values(LEAGUE_TO_TM[league]).path)
        vals = vals[vals["date"] <= season_end(season)]
        latest = (
            vals.sort_values("date")
            .groupby("player_id")
            .tail(1)[["player_id", "date", "market_value_in_eur"]]
            .rename(columns={"date": "value_date"})
        )
        merged = merged.merge(
            latest, left_on="tm_id", right_on="player_id", how="left"
        ).drop(columns=["player_id"])
        return merged

    return cached_fetch(
        "hub", "player_season", {"league": league, "season": season}, produce, force
    )
