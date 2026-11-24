"""
config/settings.py
Central configuration — loads from .env and exposes typed constants.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ── Paths ─────────────────────────────────────────────────────
DATA_RAW        = ROOT / "data" / "raw"
DATA_PROCESSED  = ROOT / "data" / "processed"
MODELS_DIR      = ROOT / "models"          # created in Phase 4
MODELS_DIR.mkdir(exist_ok=True)

# ── Database ──────────────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = int(os.getenv("DB_PORT", 3306))
DB_USER     = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME     = os.getenv("DB_NAME", "hate_speech_db")

# SQLAlchemy connection string
DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── ML ────────────────────────────────────────────────────────
RANDOM_STATE    = 42
TEST_SIZE       = 0.2
VAL_SIZE        = 0.1

TFIDF_MAX_FEATURES  = 50_000
TFIDF_NGRAM_RANGE   = (1, 2)   # unigrams + bigrams

# Label mapping  (matches the Davidson dataset)
LABEL_MAP = {0: "hate", 1: "offensive", 2: "safe"}
LABEL_REVERSE = {"hate": 0, "offensive": 1, "safe": 2}

# ── API ───────────────────────────────────────────────────────
FLASK_PORT  = int(os.getenv("FLASK_PORT", 5000))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"

# ── Logging ───────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
