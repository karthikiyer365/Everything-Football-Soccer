"""soccerhub — unified fetch layer for open-source soccer data."""

from soccerhub.errors import SoccerhubError
from soccerhub.manifest import Manifest
from soccerhub.readers.fbref import fetch_fbref_season
from soccerhub.readers.statsbomb import fetch_statsbomb_events
from soccerhub.readers.transfermarkt import (
    fetch_transfermarkt_players,
    fetch_transfermarkt_transfers,
    fetch_transfermarkt_values,
)
from soccerhub.pipelines import (
    build_player_season,
    build_player_xref,
    push_to_supabase,
    push_transfers,
    read_hub,
    run_season,
)

__all__ = [
    "SoccerhubError",
    "Manifest",
    "fetch_fbref_season",
    "fetch_statsbomb_events",
    "fetch_transfermarkt_players",
    "fetch_transfermarkt_transfers",
    "fetch_transfermarkt_values",
    "build_player_xref",
    "build_player_season",
    "push_to_supabase",
    "push_transfers",
    "read_hub",
    "run_season",
]
