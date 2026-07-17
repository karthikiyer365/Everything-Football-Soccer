import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SOCCERHUB_CACHE", str(tmp_path))


def test_fetch_club_elo_snapshot_renames_and_tags_date(monkeypatch):
    import soccerhub.readers.clubelo as ce

    fake = pd.DataFrame(
        {"rank": [1.0], "country": ["ENG"], "level": [1], "elo": [2077.3],
         "from": [pd.Timestamp("2023-06-11")], "to": [pd.Timestamp("2023-08-11")],
         "league": ["ENG-Premier League"]},
        index=pd.Index(["Man City"], name="team"),
    )

    class FakeClubElo:
        def read_by_date(self, date):
            captured["date"] = date
            return fake

    captured = {}
    monkeypatch.setattr(ce.sd, "ClubElo", FakeClubElo)

    m = ce.fetch_club_elo_snapshot("2023-08-11")
    assert captured["date"] == "2023-08-11"
    assert m.params == {"date": "2023-08-11"}
    df = pd.read_parquet(m.path)
    assert "from" not in df.columns and "to" not in df.columns
    assert df.iloc[0]["elo_from"] == "2023-06-11"  # ISO string, JSON-safe for upsert
    assert df.iloc[0]["snapshot_date"] == "2023-08-11"
    assert df.iloc[0]["team"] == "Man City"


def test_fetch_club_elo_snapshot_default_date_is_todays_cache_key(monkeypatch):
    import soccerhub.readers.clubelo as ce

    fake = pd.DataFrame(
        {"rank": [1.0], "country": ["ENG"], "level": [1], "elo": [2000.0],
         "from": [pd.Timestamp("2026-01-01")], "to": [pd.Timestamp("2026-01-08")],
         "league": ["ENG-Premier League"]},
        index=pd.Index(["Arsenal"], name="team"),
    )
    monkeypatch.setattr(ce.sd, "ClubElo",
                        lambda: type("X", (), {"read_by_date": lambda self, d: fake})())

    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    m = ce.fetch_club_elo_snapshot()
    assert m.params == {"date": today}  # resolved eagerly, not a constant "today" key
