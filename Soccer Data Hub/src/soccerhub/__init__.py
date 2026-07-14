"""soccerhub — unified fetch layer for open-source soccer data."""

from soccerhub.errors import SoccerhubError
from soccerhub.manifest import Manifest
from soccerhub.readers.fbref import fetch_fbref_season
from soccerhub.readers.statsbomb import fetch_statsbomb_events
from soccerhub.readers.transfermarkt import fetch_transfermarkt_values

__all__ = [
    "SoccerhubError",
    "Manifest",
    "fetch_fbref_season",
    "fetch_statsbomb_events",
    "fetch_transfermarkt_values",
]
