"""
src/etl/pipeline.py
Full ETL pipeline — reads raw_tweets from MySQL, applies 12+ cleaning
operations, and writes clean_tweets back.

Operations applied (in order):
  1.  Decode HTML entities            (&amp; → &)
  2.  Remove URLs
  3.  Remove @mentions
  4.  Remove hashtag symbol (keep word)
  5.  Remove RT prefix
  6.  Lowercase
  7.  Expand contractions             (don't → do not)
  8.  Remove punctuation & special chars
  9.  Remove extra whitespace
  10. Remove numeric tokens
  11. Tokenise
  12. Remove stopwords (NLTK English)
  13. Lemmatise (WordNetLemmatizer)
  14. Drop tokens shorter than 2 chars
  15. Rejoin into clean string

Run:
    python src/etl/pipeline.py
"""
import re
import sys
import html
import logging
from pathlib import Path

import nltk
import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm

# ── Bootstrap path & config ───────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    DB_URL, DATA_PROCESSED, LABEL_MAP, LOG_LEVEL
)

logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Download NLTK assets once ─────────────────────────────────
for pkg in ("stopwords", "wordnet", "punkt", "punkt_tab", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg != "punkt" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

STOP_WORDS  = set(stopwords.words("english"))
LEMMATIZER  = WordNetLemmatizer()

# Common English contractions
CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "won't":  "will not", "can't": "cannot", "couldn't": "could not",
    "isn't":  "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "wouldn't": "would not", "shouldn't": "should not",
    "i'm":   "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "he's":  "he is", "she's": "she is", "it's": "it is",
    "we're": "we are", "we've": "we have", "we'll": "we will",
    "they're": "they are", "they've": "they have", "they'll": "they will",
    "that's": "that is", "what's": "what is", "who's": "who is",
    "there's": "there is", "here's": "here is",
}


# ─────────────────────────────────────────────────────────────
#  Individual cleaning steps  (each returns a string)
# ─────────────────────────────────────────────────────────────

def step_decode_html(text: str) -> str:
    """Op 1 — decode &amp; &lt; etc."""
    return html.unescape(text)

def step_remove_urls(text: str) -> str:
    """Op 2 — strip http/https URLs"""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)

def step_remove_mentions(text: str) -> str:
    """Op 3 — strip @username"""
    return re.sub(r"@\w+", " ", text)

def step_remove_hashtag_symbol(text: str) -> str:
    """Op 4 — keep hashtag word, drop #"""
    return re.sub(r"#(\w+)", r"\1", text)

def step_remove_rt(text: str) -> str:
    """Op 5 — strip leading RT"""
    return re.sub(r"^RT\s+", "", text.strip())

def step_lowercase(text: str) -> str:
    """Op 6"""
    return text.lower()

def step_expand_contractions(text: str) -> str:
    """Op 7 — expand common contractions"""
    for contraction, expanded in CONTRACTIONS.items():
        text = text.replace(contraction, expanded)
    return text

def step_remove_punctuation(text: str) -> str:
    """Op 8 — keep only letters, digits, spaces"""
    return re.sub(r"[^a-z0-9\s]", " ", text)

def step_clean_whitespace(text: str) -> str:
    """Op 9 — collapse multiple spaces"""
    return re.sub(r"\s+", " ", text).strip()

def step_remove_numbers(text: str) -> str:
    """Op 10 — remove standalone numeric tokens"""
    return re.sub(r"\b\d+\b", " ", text)

def step_tokenise(text: str) -> list[str]:
    """Op 11 — split into word tokens"""
    return word_tokenize(text)

def step_remove_stopwords(tokens: list[str]) -> list[str]:
    """Op 12"""
    return [t for t in tokens if t not in STOP_WORDS]

def step_lemmatise(tokens: list[str]) -> list[str]:
    """Op 13 — reduce to root forms"""
    return [LEMMATIZER.lemmatize(t) for t in tokens]

def step_min_length(tokens: list[str], min_len: int = 2) -> list[str]:
    """Op 14 — drop very short tokens"""
    return [t for t in tokens if len(t) >= min_len]


# ─────────────────────────────────────────────────────────────
#  Main cleaning function
# ─────────────────────────────────────────────────────────────

def clean_tweet(raw_text: str) -> str:
    """Apply all 15 steps to a single tweet string."""
    if not isinstance(raw_text, str):
        return ""

    t = step_decode_html(raw_text)
    t = step_remove_urls(t)
    t = step_remove_mentions(t)
    t = step_remove_hashtag_symbol(t)
    t = step_remove_rt(t)
    t = step_lowercase(t)
    t = step_expand_contractions(t)
    t = step_remove_punctuation(t)
    t = step_clean_whitespace(t)
    t = step_remove_numbers(t)

    tokens = step_tokenise(t)
    tokens = step_remove_stopwords(tokens)
    tokens = step_lemmatise(tokens)
    tokens = step_min_length(tokens)

    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────
#  ETL orchestrator
# ─────────────────────────────────────────────────────────────

def run_pipeline():
    engine = create_engine(DB_URL)

    log.info("Reading raw_tweets from MySQL …")
    df = pd.read_sql("SELECT id, raw_label, tweet_text FROM raw_tweets", engine)
    log.info(f"  → {len(df):,} rows loaded")

    # ── Apply cleaning ────────────────────────────────────────
    log.info("Applying 15-step cleaning pipeline …")
    tqdm.pandas(desc="Cleaning tweets")
    df["clean_text"] = df["tweet_text"].progress_apply(clean_tweet)

    # ── Derived columns ───────────────────────────────────────
    df["token_count"] = df["clean_text"].str.split().str.len()
    df["label"]       = df["raw_label"].map(LABEL_MAP)
    df["label_int"]   = df["raw_label"]

    # ── Noise reduction metric ────────────────────────────────
    orig_chars  = df["tweet_text"].str.len().sum()
    clean_chars = df["clean_text"].str.len().sum()
    reduction   = (1 - clean_chars / orig_chars) * 100
    log.info(f"  → Text noise reduced by {reduction:.1f}%")

    # ── Write to clean_tweets ─────────────────────────────────
    out = df[["id", "clean_text", "token_count", "label", "label_int"]].copy()
    out = out.rename(columns={"id": "raw_id"})

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM clean_tweets"))

    out.to_sql("clean_tweets", con=engine, if_exists="append", index=False)
    log.info(f"  → {len(out):,} clean rows written to clean_tweets")

    # ── Also save as parquet for fast ML loading ──────────────
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    parquet_path = DATA_PROCESSED / "clean_tweets.parquet"
    out.to_parquet(parquet_path, index=False)
    log.info(f"  → Parquet saved → {parquet_path}")

    # ── Class distribution ────────────────────────────────────
    dist = df["label"].value_counts()
    log.info("\nLabel distribution:")
    for label, count in dist.items():
        log.info(f"  {label:12s}  {count:,}  ({count/len(df)*100:.1f}%)")

    log.info("\nETL complete. Next: python src/models/train.py")


if __name__ == "__main__":
    run_pipeline()
