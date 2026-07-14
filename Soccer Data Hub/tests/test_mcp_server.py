from dataclasses import asdict

from soccerhub.manifest import Manifest


def _fake_manifest(source):
    return Manifest(
        path=f"data/{source}/x.parquet",
        source=source,
        dataset="events",
        params={},
        rows=1,
        cols=1,
        date_range=None,
        fetched_at="2026-07-14T00:00:00+00:00",
    )


def test_fbref_tool_returns_manifest_dict(monkeypatch):
    import soccerhub.mcp_server as server

    monkeypatch.setattr(
        server, "fetch_fbref_season", lambda league, season: _fake_manifest("fbref")
    )
    result = server.fbref_season("ENG-Premier League", "2023")
    assert result == asdict(_fake_manifest("fbref"))
    assert result["source"] == "fbref"


def test_server_registers_three_tools():
    import soccerhub.mcp_server as server

    for name in ("fbref_season", "statsbomb_events", "transfermarkt_values"):
        assert callable(getattr(server, name))
