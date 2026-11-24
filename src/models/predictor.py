"""
src/models/predictor.py
Loads trained artefacts and exposes a clean predict() interface.
Used by both the Flask API and the Streamlit dashboard.
"""
import sys
import logging
from pathlib import Path
from dataclasses import dataclass

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import MODELS_DIR, LABEL_MAP

log = logging.getLogger(__name__)


@dataclass
class Prediction:
    label:       str          # "hate" | "offensive" | "safe"
    confidence:  float        # 0.0 – 1.0
    probabilities: dict       # {"hate": 0.82, "offensive": 0.12, "safe": 0.06}
    label_int:   int          # 0 | 1 | 2


class HateSpeechPredictor:
    """
    Thin wrapper around the saved ensemble.

    Usage:
        predictor = HateSpeechPredictor()
        result = predictor.predict("some tweet text")
        print(result.label, result.confidence)
    """

    def __init__(self):
        self._tfidf    = None
        self._ensemble = None
        self._xgb      = None
        self._rf       = None
        self._svm      = None
        self._loaded   = False

    def _load(self):
        if self._loaded:
            return
        required = ["tfidf.pkl", "ensemble.pkl", "xgboost.pkl",
                    "random_forest.pkl", "svm.pkl"]
        missing = [f for f in required if not (MODELS_DIR / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing model files: {missing}\n"
                f"Run:  python src/models/train.py"
            )
        self._tfidf    = joblib.load(MODELS_DIR / "tfidf.pkl")
        self._ensemble = joblib.load(MODELS_DIR / "ensemble.pkl")
        self._xgb      = joblib.load(MODELS_DIR / "xgboost.pkl")
        self._rf       = joblib.load(MODELS_DIR / "random_forest.pkl")
        self._svm      = joblib.load(MODELS_DIR / "svm.pkl")
        self._loaded   = True
        log.info("Models loaded from %s", MODELS_DIR)

    def predict(self, text: str) -> Prediction:
        """Predict a single tweet string."""
        self._load()
        vec = self._tfidf.transform([text])

        # Ensemble probabilities  →  shape (1, 3)
        probs = self._ensemble.predict_proba(vec)[0]
        label_int   = int(np.argmax(probs))
        label       = LABEL_MAP[label_int]
        confidence  = float(probs[label_int])

        return Prediction(
            label=label,
            confidence=confidence,
            probabilities={LABEL_MAP[i]: float(p) for i, p in enumerate(probs)},
            label_int=label_int,
        )

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        """Predict a list of tweets (more efficient than looping predict())."""
        self._load()
        vecs  = self._tfidf.transform(texts)
        probs = self._ensemble.predict_proba(vecs)   # shape (N, 3)

        results = []
        for prob_row in probs:
            label_int  = int(np.argmax(prob_row))
            label      = LABEL_MAP[label_int]
            confidence = float(prob_row[label_int])
            results.append(Prediction(
                label=label,
                confidence=confidence,
                probabilities={LABEL_MAP[i]: float(p) for i, p in enumerate(prob_row)},
                label_int=label_int,
            ))
        return results

    def individual_probs(self, text: str) -> dict:
        """
        Returns per-model probabilities — useful for the dashboard's
        model comparison view.
        """
        self._load()
        vec = self._tfidf.transform([text])
        return {
            "xgboost":       self._xgb.predict_proba(vec)[0].tolist(),
            "random_forest": self._rf.predict_proba(vec)[0].tolist(),
            "svm":           self._svm.predict_proba(vec)[0].tolist(),
        }


# Singleton — import this everywhere
predictor = HateSpeechPredictor()
