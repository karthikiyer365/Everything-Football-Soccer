"""Upsert pipeline outputs into Supabase Postgres (service role, RLS bypassed)."""
import os

import pandas as pd
from supabase import create_client

from soccerhub.manifest import Manifest

CHUNK = 500
CONFLICT_KEY = "league,season,team,player_name"


def push_to_supabase(manifest: Manifest, table: str) -> int:
    client = create_client(
        os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    )
    df = pd.read_parquet(manifest.path)
    records = df.astype(object).where(pd.notna(df), None).to_dict("records")
    for i in range(0, len(records), CHUNK):
        client.table(table).upsert(
            records[i : i + CHUNK], on_conflict=CONFLICT_KEY
        ).execute()
    return len(records)
