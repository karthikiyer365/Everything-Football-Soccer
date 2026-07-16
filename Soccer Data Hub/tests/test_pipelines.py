def test_run_season_chains_and_pushes(monkeypatch):
    import soccerhub.pipelines as pl

    order = []
    refetches = []
    fake_manifest = object()
    monkeypatch.setattr(
        pl, "build_player_xref",
        lambda l, s, force=False, refetch=False:
            (order.append("xref"), refetches.append(refetch))[0])
    monkeypatch.setattr(pl, "push_xref",
                        lambda m, l, s: order.append("push:xref"))
    monkeypatch.setattr(
        pl, "build_player_season",
        lambda l, s, force=False, refetch=False:
            (order.append("merge"), refetches.append(refetch), fake_manifest)[2],
    )
    monkeypatch.setattr(pl, "push_to_supabase",
                        lambda m, table: order.append(f"push:{table}"))

    out = pl.run_season("ENG-Premier League", "2023", force=True)
    assert order == ["xref", "push:xref", "merge", "push:player_season"]
    assert out is fake_manifest
    assert refetches == [True, True]  # cron force must reach the leaf readers


def test_read_hub_paginates_and_filters(monkeypatch):
    import io
    import json as jsonlib
    import urllib.request

    import soccerhub.pipelines.query as q

    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "anon")
    monkeypatch.setattr(q, "PAGE", 2)

    pages = [[{"a": 1}, {"a": 2}], [{"a": 3}]]  # 2 full + short page -> stop
    urls = []

    def fake_urlopen(req):
        urls.append(req.full_url)
        return io.BytesIO(jsonlib.dumps(pages[len(urls) - 1]).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    df = q.read_hub("player_season", league="ITA-Serie A")
    assert list(df.a) == [1, 2, 3]
    assert len(urls) == 2
    assert "league=eq.ITA-Serie+A" in urls[0] or "league=eq.ITA-Serie%20A" in urls[0]
    assert "offset=2" in urls[1]


def test_push_xref_adds_key_columns(monkeypatch, tmp_path):
    import pandas as pd

    import soccerhub.pipelines as pl
    from soccerhub.manifest import Manifest

    xdf = pd.DataFrame({"fbref_name": ["A"], "team": ["T"],
                        "tm_id": [1], "method": ["exact"], "confidence": [1.0]})
    p = tmp_path / "x.parquet"
    xdf.to_parquet(p)
    m = Manifest(path=str(p), source="xref", dataset="players",
                 params={}, rows=1, cols=5, date_range=None, fetched_at="t")

    captured = {}

    def fake_upsert(df, table, on_conflict):
        captured["df"], captured["table"], captured["key"] = df, table, on_conflict
        return len(df)

    import soccerhub.pipelines.supa as sp
    monkeypatch.setattr(sp, "upsert_df", fake_upsert)

    n = pl.push_xref(m, "ITA-Serie A", "2023")
    assert n == 1
    assert captured["table"] == "player_xref"
    assert captured["df"].iloc[0]["league"] == "ITA-Serie A"
    assert captured["df"].iloc[0]["season"] == "2023"
    assert captured["key"] == "league,season,team,fbref_name"


def test_public_exports():
    import soccerhub
    for fn in ("build_player_xref", "build_player_season",
               "push_to_supabase", "run_season"):
        assert callable(getattr(soccerhub, fn))
