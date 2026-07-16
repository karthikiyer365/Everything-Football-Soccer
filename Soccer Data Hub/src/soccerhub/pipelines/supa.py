"""Upsert pipeline outputs into Supabase Postgres (service role, RLS bypassed)."""
import os

import pandas as pd
from supabase import create_client

from soccerhub.manifest import Manifest

CHUNK = 500
CONFLICT_KEY = "league,season,team,player_name"


def push_to_supabase(
    manifest: Manifest, table: str, on_conflict: str = CONFLICT_KEY
) -> int:
    return upsert_df(pd.read_parquet(manifest.path), table, on_conflict)


def upsert_df(df: pd.DataFrame, table: str, on_conflict: str) -> int:
    client = create_client(
        os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    )
    df = df.copy()
    for col in df.columns:
        # pandas upcasts nullable ints to float ('25000000.0' breaks bigint
        # columns in postgres) — send whole-number floats back as ints
        if pd.api.types.is_float_dtype(df[col]):
            s = df[col].dropna()
            if len(s) and (s % 1 == 0).all():
                df[col] = df[col].astype("Int64")
    records = df.astype(object).where(pd.notna(df), None).to_dict("records")
    for i in range(0, len(records), CHUNK):
        client.table(table).upsert(
            records[i : i + CHUNK], on_conflict=on_conflict
        ).execute()
    return len(records)
