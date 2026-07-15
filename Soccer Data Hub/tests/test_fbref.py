import json

import pandas as pd
import pytest


def test_patch_league_config_writes_serie_a_fix(tmp_path, monkeypatch):
    monkeypatch.setenv("SOCCERDATA_DIR", str(tmp_path))
    from soccerhub.readers.fbref import _patch_league_config

    _patch_league_config()
    data = json.loads((tmp_path / "config" / "league_dict.json").read_text())
    assert data["ITA-Serie A"]["FBref"] == "Serie A (M)"
    assert data["ITA-Serie A"]["Understat"] == "Serie A"  # full entry, not partial


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SOCCERHUB_CACHE", str(tmp_path))


def test_fetch_fbref_season_caches_reader_output(monkeypatch):
    import soccerhub.readers.fbref as fbref

    class FakeFBref:
        def __init__(self, leagues, seasons):
            self.leagues = leagues
            self.seasons = seasons

        def read_player_season_stats(self):
            return pd.DataFrame({"player": ["A", "B"], "goals": [3, 1]})

    monkeypatch.setattr(fbref.sd, "FBref", FakeFBref)

    m = fbref.fetch_fbref_season("ENG-Premier League", "2023")
    assert m.source == "fbref"
    assert m.dataset == "player_season"
    assert m.rows == 2
    assert m.params == {"league": "ENG-Premier League", "season": "2023"}
    assert pd.read_parquet(m.path).shape == (2, 2)
