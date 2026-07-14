from kloppy import statsbomb

from soccerhub.cache import cached_fetch
from soccerhub.manifest import Manifest


def fetch_statsbomb_events(match_id: str, force: bool = False) -> Manifest:
    """Event stream for one StatsBomb open-data match, flattened to a DataFrame."""

    def produce():
        return statsbomb.load_open_data(match_id=match_id).to_df()

    return cached_fetch(
        "statsbomb", "events", {"match_id": match_id}, produce, force
    )
