"""
scripts/ingest.py
Reads the raw CSV and bulk-inserts it into MySQL raw_tweets table.
Uses chunked inserts for memory efficiency.

Run:
    python scripts/ingest.py
"""
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_RAW, DB_URL

CSV_PATH   = DATA_RAW / "labeled_data.csv"
CHUNK_SIZE = 500   # rows per INSERT batch


def load_and_rename(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # The Davidson CSV has an unnamed index column; drop it
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Standardise column names to match our schema
    rename = {
        "count":          "count",
        "hate_speech":    "hate_speech",
        "offensive_language": "offensive_lang",
        "neither":        "neither",
        "class":          "raw_label",
        "tweet":          "tweet_text",
    }
    df = df.rename(columns=rename)

    # Add a surrogate tweet_id (the dataset has no real IDs)
    df.insert(0, "tweet_id", range(1, len(df) + 1))

    return df[["tweet_id", "count", "hate_speech", "offensive_lang",
               "neither", "raw_label", "tweet_text"]]


def ingest():
    if not CSV_PATH.exists():
        print(f"[error] Dataset not found at {CSV_PATH}")
        print("        Run: python scripts/download_data.py first")
        sys.exit(1)

    df = load_and_rename(CSV_PATH)
    print(f"[info] Loaded {len(df):,} rows from CSV")

    engine = create_engine(DB_URL)

    # Wipe existing raw data so ingest is idempotent
    with engine.begin() as conn:
        conn.execute(__import__("sqlalchemy").text("DELETE FROM raw_tweets"))

    chunks = [df[i:i + CHUNK_SIZE] for i in range(0, len(df), CHUNK_SIZE)]
    for chunk in tqdm(chunks, desc="Inserting to MySQL"):
        chunk.to_sql("raw_tweets", con=engine, if_exists="append", index=False)

    print(f"\n[ok] Ingested {len(df):,} tweets into raw_tweets")
    print("Next: python src/etl/pipeline.py")


if __name__ == "__main__":
    ingest()
