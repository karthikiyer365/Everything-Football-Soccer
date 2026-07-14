import pandas as pd

from soccerhub.cache import cached_fetch
from soccerhub.manifest import Manifest

# Pre-scraped Transfermarkt valuations (dcaribou/transfermarkt-datasets,
# published on Kaggle; git repo stores data via DVC so raw.githubusercontent 404s).
# Calibration knob: update if the upstream path moves.
TM_VALUATIONS_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "davidcariboo/player-scores?fileName=player_valuations.csv"
)


TM_PLAYERS_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "davidcariboo/player-scores?fileName=players.csv"
)

PLAYER_COLS = [
    "player_id", "name", "date_of_birth", "country_of_citizenship",
    "position", "sub_position", "current_club_id", "current_club_name",
    "current_club_domestic_competition_id", "market_value_in_eur",
]


def fetch_transfermarkt_players(competition: str, force: bool = False) -> Manifest:
    """Player identity registry (name, DOB, club) for one domestic competition."""

    def produce():
        df = pd.read_csv(TM_PLAYERS_URL)
        df = df[df["current_club_domestic_competition_id"] == competition]
        return df[PLAYER_COLS].reset_index(drop=True)

    return cached_fetch(
        "transfermarkt", "players", {"competition": competition}, produce, force
    )


def fetch_transfermarkt_values(competition: str, force: bool = False) -> Manifest:
    """Player market valuations filtered to one domestic competition."""

    def produce():
        df = pd.read_csv(TM_VALUATIONS_URL)
        return df[df["player_club_domestic_competition_id"] == competition]

    return cached_fetch(
        "transfermarkt", "valuations", {"competition": competition}, produce, force
    )
