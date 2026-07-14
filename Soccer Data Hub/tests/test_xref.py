import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SOCCERHUB_CACHE", str(tmp_path))


def test_normalize():
    from soccerhub.pipelines.xref import normalize
    assert normalize("Ángel Di María") == "angel di maria"


def test_build_player_xref_ladder(monkeypatch, tmp_path):
    import soccerhub.pipelines.xref as xr
    from soccerhub.cache import cached_fetch

    # fake fbref parquet: 3 players, one team
    fbref = pd.DataFrame(
        {"born": [2001, 1987, 1999]},
        index=pd.MultiIndex.from_tuples(
            [
                ("ENG-Premier League", "2324", "Arsenal", "Bukayo Saka"),
                ("ENG-Premier League", "2324", "Arsenal", "Angel Di Maria"),
                ("ENG-Premier League", "2324", "Arsenal", "Totally Unknown"),
            ],
            names=["league", "season", "team", "player"],
        ),
    )
    m_fbref = cached_fetch(
        "fbref",
        "player_season",
        {"league": "ENG-Premier League", "season": "2023"},
        lambda: fbref,
    )
    monkeypatch.setattr(xr, "fetch_fbref_season", lambda l, s, force=False: m_fbref)

    # fake tm players registry
    tm = pd.DataFrame(
        {
            "player_id": [10, 20],
            "name": ["Bukayo Saka", "Ángel Di María Hernández"],
            "date_of_birth": ["2001-09-05", "1988-02-14"],
        }
    )
    m_tm = cached_fetch("transfermarkt", "players", {"competition": "GB1"}, lambda: tm)
    monkeypatch.setattr(
        xr, "fetch_transfermarkt_players", lambda c, force=False: m_tm
    )

    # fake mapping file: covers Saka only
    mapping = pd.DataFrame(
        {
            "PlayerFBref": ["Bukayo Saka"],
            "UrlTmarkt": [
                "https://www.transfermarkt.com/bukayo-saka/profil/spieler/10"
            ],
        }
    )
    monkeypatch.setattr(xr, "_load_mapping", lambda: mapping)

    m = xr.build_player_xref("ENG-Premier League", "2023")
    df = pd.read_parquet(m.path).set_index("fbref_name")

    assert df.loc["Bukayo Saka", "method"] == "mapping_file"
    assert df.loc["Bukayo Saka", "tm_id"] == 10
    assert df.loc["Angel Di Maria", "method"] == "fuzzy"
    assert df.loc["Angel Di Maria", "tm_id"] == 20
    assert df.loc["Totally Unknown", "method"] == "unmatched"
    assert pd.isna(df.loc["Totally Unknown", "tm_id"])
