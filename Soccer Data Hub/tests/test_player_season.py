import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SOCCERHUB_CACHE", str(tmp_path))


def _fake_fbref():
    cols = pd.MultiIndex.from_tuples([
        ("nation", ""), ("pos", ""), ("age", ""), ("born", ""),
        ("Playing Time", "MP"), ("Playing Time", "Min"),
        ("Performance", "Gls"), ("Performance", "Ast"),
    ])
    idx = pd.MultiIndex.from_tuples(
        [("ENG-Premier League", "2324", "Arsenal", "Bukayo Saka"),
         ("ENG-Premier League", "2324", "Arsenal", "Totally Unknown")],
        names=["league", "season", "team", "player"],
    )
    return pd.DataFrame(
        [["ENG", "FW", 21, 2001, 38, 3300, 14, 9],
         ["ENG", "MF", 30, 1993, 2, 90, 0, 0]],
        index=idx, columns=cols,
    )


def test_flatten_fbref_canonical_names():
    from soccerhub.pipelines.player_season import flatten_fbref
    flat = flatten_fbref(_fake_fbref())
    assert {"player_name", "goals", "assists", "matches_played", "minutes",
            "nationality", "position", "birth_year"} <= set(flat.columns)
    assert flat.loc[flat.player_name == "Bukayo Saka", "goals"].iloc[0] == 14


def test_season_end():
    from soccerhub.pipelines.player_season import season_end
    assert season_end("2023") == "2024-06-30"


def test_build_player_season_merges_value(monkeypatch):
    import soccerhub.pipelines.player_season as ps
    from soccerhub.cache import cached_fetch

    m_fbref = cached_fetch("fbref", "player_season",
                           {"league": "ENG-Premier League", "season": "2023"},
                           _fake_fbref)
    monkeypatch.setattr(ps, "fetch_fbref_season", lambda l, s, force=False: m_fbref)

    xref = pd.DataFrame({
        "fbref_name": ["Bukayo Saka", "Totally Unknown"],
        "team": ["Arsenal", "Arsenal"],
        "tm_id": pd.array([10, None], dtype="Int64"),
        "method": ["mapping_file", "unmatched"],
        "confidence": [1.0, 0.0],
    })
    m_xref = cached_fetch("xref", "players",
                          {"league": "ENG-Premier League", "season": "2023"},
                          lambda: xref)
    monkeypatch.setattr(ps, "build_player_xref", lambda l, s, force=False: m_xref)

    vals = pd.DataFrame({
        "player_id": [10, 10, 10],
        "date": ["2023-11-01", "2024-05-30", "2024-08-01"],  # last one after season end
        "market_value_in_eur": [90_000_000, 120_000_000, 140_000_000],
    })
    m_vals = cached_fetch("transfermarkt", "valuations", {"competition": "GB1"},
                          lambda: vals)
    monkeypatch.setattr(ps, "fetch_transfermarkt_values",
                        lambda c, force=False: m_vals)

    m = ps.build_player_season("ENG-Premier League", "2023")
    df = pd.read_parquet(m.path).set_index("player_name")

    saka = df.loc["Bukayo Saka"]
    assert saka["market_value_in_eur"] == 120_000_000  # latest <= 2024-06-30
    assert saka["season"] == "2023"
    unk = df.loc["Totally Unknown"]
    assert pd.isna(unk["market_value_in_eur"])  # kept, not dropped
    assert unk["xref_method"] == "unmatched"
