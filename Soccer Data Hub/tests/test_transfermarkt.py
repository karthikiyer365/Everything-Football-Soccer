import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SOCCERHUB_CACHE", str(tmp_path))


def test_fetch_transfermarkt_values_filters_by_competition(monkeypatch):
    import soccerhub.readers.transfermarkt as tm

    fake = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "market_value_in_eur": [100, 200, 300],
            "player_club_domestic_competition_id": ["GB1", "ES1", "GB1"],
        }
    )
    monkeypatch.setattr(tm.pd, "read_csv", lambda url: fake)

    m = tm.fetch_transfermarkt_values("GB1")
    assert m.source == "transfermarkt"
    assert m.dataset == "valuations"
    assert m.rows == 2  # only GB1 rows
    assert m.params == {"competition": "GB1"}
    assert set(pd.read_parquet(m.path)["player_id"]) == {1, 3}
