"""Merged player-season table: FBref stats + Transfermarkt market value."""
import pandas as pd

from soccerhub.cache import cached_fetch
from soccerhub.manifest import Manifest
from soccerhub.pipelines.xref import build_player_xref
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

        # None = all competitions: valuations are keyed by CURRENT club, so a
        # league filter would drop every player who transferred out since
        vals = pd.read_parquet(fetch_transfermarkt_values(None).path)
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

        # two different players, same name, same team (Adrián López x2,
        # Deportivo 2009): name-keyed xref matched both to one tm_id and the
        # DB primary key rejects the pair. Disambiguate the name with birth
        # year; trust the tm match only for the biggest-minutes row.
        key = ["league", "season", "team", "player_name"]
        dups = merged.duplicated(key, keep=False)
        if dups.any():
            top = merged.loc[dups].groupby(key)["minutes"].transform("max")
            minor = pd.Series(False, index=merged.index)
            minor.loc[dups] = merged.loc[dups, "minutes"] < top
            merged.loc[minor, ["tm_id", "market_value_in_eur", "value_date"]] = pd.NA
            merged.loc[minor, "xref_method"] = "ambiguous"
            merged.loc[minor, "xref_confidence"] = 0.0
            merged.loc[dups, "player_name"] = (
                merged.loc[dups, "player_name"] + " ("
                + merged.loc[dups, "birth_year"].astype("Int64").astype(str) + ")"
            )
            # ponytail: same name + team + birth year would still collide;
            # keep the bigger spell if that ever happens
            merged = (
                merged.sort_values("minutes", ascending=False)
                .drop_duplicates(key, keep="first")
                .sort_index()
            )
        return merged

    return cached_fetch(
        "hub", "player_season", {"league": league, "season": season}, produce, force
    )
