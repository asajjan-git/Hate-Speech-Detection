"""
scripts/download_data.py
Downloads the Davidson et al. (2017) hate speech dataset — the same one used
to train the original model.  31K+ tweets, 3 classes: hate / offensive / safe.

Run:
    python scripts/download_data.py
"""
import sys
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_RAW

RAW_URL = (
    "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language"
    "/master/data/labeled_data.csv"
)
DEST = DATA_RAW / "labeled_data.csv"


def download():
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if DEST.exists():
        print(f"[skip] Already downloaded → {DEST}")
        return

    print(f"Downloading dataset from GitHub …")
    r = requests.get(RAW_URL, timeout=30)
    r.raise_for_status()

    DEST.write_bytes(r.content)
    lines = DEST.read_text().count("\n")
    print(f"[ok] Saved {lines - 1:,} rows → {DEST}")


if __name__ == "__main__":
    download()
