def test_run_season_chains_and_pushes(monkeypatch):
    import soccerhub.pipelines as pl

    order = []
    fake_manifest = object()
    monkeypatch.setattr(pl, "build_player_xref",
                        lambda l, s, force=False: order.append("xref"))
    monkeypatch.setattr(
        pl, "build_player_season",
        lambda l, s, force=False: (order.append("merge"), fake_manifest)[1],
    )
    monkeypatch.setattr(pl, "push_to_supabase",
                        lambda m, table: order.append(f"push:{table}"))

    out = pl.run_season("ENG-Premier League", "2023")
    assert order == ["xref", "merge", "push:player_season"]
    assert out is fake_manifest


def test_public_exports():
    import soccerhub
    for fn in ("build_player_xref", "build_player_season",
               "push_to_supabase", "run_season"):
        assert callable(getattr(soccerhub, fn))
