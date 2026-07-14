from dataclasses import asdict

from mcp.server.fastmcp import FastMCP

from soccerhub.readers.fbref import fetch_fbref_season
from soccerhub.readers.statsbomb import fetch_statsbomb_events
from soccerhub.readers.transfermarkt import fetch_transfermarkt_values

mcp = FastMCP("soccerhub")


@mcp.tool()
def fbref_season(league: str, season: str) -> dict:
    """Fetch + cache FBref player season stats; returns a manifest."""
    return asdict(fetch_fbref_season(league, season))


@mcp.tool()
def statsbomb_events(match_id: str) -> dict:
    """Fetch + cache StatsBomb open-data match events; returns a manifest."""
    return asdict(fetch_statsbomb_events(match_id))


@mcp.tool()
def transfermarkt_values(competition: str) -> dict:
    """Fetch + cache Transfermarkt valuations for a competition; returns a manifest."""
    return asdict(fetch_transfermarkt_values(competition))


if __name__ == "__main__":
    mcp.run()
