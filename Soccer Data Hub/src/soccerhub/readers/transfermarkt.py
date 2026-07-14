import pandas as pd

from soccerhub.cache import cached_fetch
from soccerhub.manifest import Manifest

# Pre-scraped Transfermarkt valuations (dcaribou/transfermarkt-datasets).
# Calibration knob: update if the upstream path moves.
TM_VALUATIONS_URL = (
    "https://raw.githubusercontent.com/dcaribou/transfermarkt-datasets/"
    "master/data/prep/player_valuations.csv"
)


def fetch_transfermarkt_values(competition: str, force: bool = False) -> Manifest:
    """Player market valuations filtered to one domestic competition."""

    def produce():
        df = pd.read_csv(TM_VALUATIONS_URL)
        return df[df["player_club_domestic_competition_id"] == competition]

    return cached_fetch(
        "transfermarkt", "valuations", {"competition": competition}, produce, force
    )
