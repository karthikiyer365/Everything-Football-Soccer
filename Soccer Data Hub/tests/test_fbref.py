import pandas as pd
import pytest


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
