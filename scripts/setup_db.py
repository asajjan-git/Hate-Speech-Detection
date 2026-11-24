"""
scripts/setup_db.py
Creates the MySQL database and the raw tweets table.
Run ONCE before ingestion.

Requirements: MySQL running locally, credentials in .env

Run:
    python scripts/setup_db.py
"""
import sys
from pathlib import Path

import pymysql
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_URL

# ── SQL schema ────────────────────────────────────────────────
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS raw_tweets (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    tweet_id        BIGINT,
    count           INT COMMENT 'number of CrowdFlower coders',
    hate_speech     INT COMMENT 'votes for hate speech',
    offensive_lang  INT COMMENT 'votes for offensive language',
    neither         INT COMMENT 'votes for neither',
    raw_label       INT COMMENT '0=hate 1=offensive 2=safe',
    tweet_text      TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS clean_tweets (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    raw_id          INT,
    clean_text      TEXT,
    token_count     INT,
    label           VARCHAR(20),
    label_int       INT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (raw_id) REFERENCES raw_tweets(id)
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    tweet_text      TEXT,
    predicted_label VARCHAR(20),
    confidence      FLOAT,
    xgb_prob_hate   FLOAT,
    rf_prob_hate    FLOAT,
    svm_prob_hate   FLOAT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def create_database():
    """Create the DB if it doesn't exist yet."""
    conn = pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD
    )
    with conn.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    conn.commit()
    conn.close()
    print(f"[ok] Database `{DB_NAME}` ready")


def create_tables():
    """Create tables from schema SQL."""
    engine = create_engine(DB_URL)
    with engine.begin() as conn:
        for statement in SCHEMA_SQL.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))
    print("[ok] Tables created: raw_tweets, clean_tweets, predictions")


if __name__ == "__main__":
    create_database()
    create_tables()
    print("\nDatabase setup complete. Next: python scripts/ingest.py")
