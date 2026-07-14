"""Throwaway local preview for soccerhub readers — testing only.

Run:  .venv/bin/python preview_app.py   →  http://localhost:8765
No deps beyond the package itself (stdlib http.server + pandas to_html).
"""
import html
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

from soccerhub import (
    fetch_fbref_season,
    fetch_statsbomb_events,
    fetch_transfermarkt_values,
)

# source → (fn, [(arg, default), …], note)
READERS = {
    "fbref": (
        fetch_fbref_season,
        [("league", "ENG-Premier League"), ("season", "2023")],
        "player season stats — first fetch scrapes FBref, can take minutes",
    ),
    "statsbomb": (
        fetch_statsbomb_events,
        [("match_id", "3869685")],  # WC 2022 final
        "open-data match events via kloppy",
    ),
    "transfermarkt": (
        fetch_transfermarkt_values,
        [("competition", "GB1")],  # Premier League
        "player valuations from pre-scraped dataset — first fetch downloads a large CSV",
    ),
}

FILTER_FIELDS = """
  <fieldset><legend>peruse</legend>
    filter col <input name="fcol" size="18">
    op <select name="fop"><option>contains</option><option>==</option><option>&gt;=</option><option>&lt;=</option></select>
    value <input name="fval" size="10">
    &nbsp;|&nbsp; sort col <input name="scol" size="18">
    desc <input type="checkbox" name="desc" checked>
    &nbsp;|&nbsp; limit <input name="limit" value="50" size="4">
    &nbsp;|&nbsp; force refetch <input type="checkbox" name="force">
  </fieldset>
"""


def index_page() -> str:
    forms = []
    for source, (_, args, note) in READERS.items():
        inputs = " ".join(
            f'{a} <input name="{a}" value="{html.escape(d)}" size="20">' for a, d in args
        )
        forms.append(
            f"<h3>{source}</h3><p><i>{note}</i></p>"
            f'<form action="/fetch"><input type="hidden" name="source" value="{source}">'
            f"{inputs}{FILTER_FIELDS}<button>fetch</button></form><hr>"
        )
    return (
        "<h1>soccerhub preview (testing only)</h1>"
        "<p>Results cache to parquet under ./data — repeat fetches are instant.</p>"
        + "".join(forms)
    )


def flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(p) for p in t if p and str(p) != "nan") for t in df.columns]
    if isinstance(df.index, pd.MultiIndex) or df.index.name:
        df = df.reset_index()
    return df


def peruse(df: pd.DataFrame, q: dict) -> pd.DataFrame:
    fcol, fop, fval = q.get("fcol", [""])[0], q.get("fop", ["contains"])[0], q.get("fval", [""])[0]
    if fcol and fval and fcol in df.columns:
        s = df[fcol]
        if fop == "contains":
            df = df[s.astype(str).str.contains(fval, case=False, na=False)]
        elif fop == "==":
            df = df[s.astype(str) == fval]
        elif fop == ">=":
            df = df[pd.to_numeric(s, errors="coerce") >= float(fval)]
        elif fop == "<=":
            df = df[pd.to_numeric(s, errors="coerce") <= float(fval)]
    scol = q.get("scol", [""])[0]
    if scol and scol in df.columns:
        df = df.sort_values(scol, ascending="desc" not in q)
    return df.head(int(q.get("limit", ["50"])[0]))


def fetch_page(q: dict) -> str:
    source = q["source"][0]
    fn, args, _ = READERS[source]
    kwargs = {a: q.get(a, [d])[0] for a, d in args}
    m = fn(**kwargs, force="force" in q)
    df = flatten(pd.read_parquet(m.path))
    cols = ", ".join(df.columns)
    shown = peruse(df, q)
    return (
        f'<p><a href="/">&larr; back</a></p><h2>{source} — {m.rows} rows cached</h2>'
        f"<pre>manifest: {html.escape(m.to_json())}</pre>"
        f"<p><b>columns:</b> {html.escape(cols)}</p>"
        f"<p>showing {len(shown)} rows after filters</p>"
        + shown.to_html(index=False, border=1)
    )


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == "/fetch":
                body = fetch_page(parse_qs(url.query))
            else:
                body = index_page()
            code = 200
        except Exception:
            body, code = f"<pre>{html.escape(traceback.format_exc())}</pre>", 500
        payload = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):  # quiet
        pass


if __name__ == "__main__":
    print("soccerhub preview → http://localhost:8765")
    ThreadingHTTPServer(("localhost", 8765), Handler).serve_forever()
