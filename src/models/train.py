"""
src/models/train.py
Trains an ensemble of XGBoost + Random Forest + SVM on TF-IDF features.

Steps:
  1. Load clean_tweets.parquet
  2. Build TF-IDF feature matrix (unigrams + bigrams, 50K vocab)
  3. Handle class imbalance with SMOTE
  4. Train each base model with RandomizedSearchCV
  5. Build soft-voting ensemble
  6. Evaluate — F1, precision, recall, confusion matrix
  7. Save all artefacts to models/

Run:
    python src/models/train.py
"""
import sys
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    DATA_PROCESSED, MODELS_DIR, RANDOM_STATE, TEST_SIZE,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, LOG_LEVEL
)

logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  1. Load data
# ─────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.Series, pd.Series]:
    path = DATA_PROCESSED / "clean_tweets.parquet"
    if not path.exists():
        log.error(f"Not found: {path}  →  Run ETL pipeline first")
        sys.exit(1)

    df = pd.read_parquet(path)
    df = df.dropna(subset=["clean_text", "label_int"])
    df = df[df["clean_text"].str.strip() != ""]

    log.info(f"Loaded {len(df):,} clean tweets")
    return df["clean_text"], df["label_int"]


# ─────────────────────────────────────────────────────────────
#  2. TF-IDF vectorizer
# ─────────────────────────────────────────────────────────────

def build_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,      # log-scale TF
        min_df=2,               # ignore extremely rare tokens
        max_df=0.95,            # ignore near-universal tokens
        strip_accents="unicode",
    )


# ─────────────────────────────────────────────────────────────
#  3. Individual model builders with tuning grids
# ─────────────────────────────────────────────────────────────

def tune_xgboost(X_train, y_train) -> XGBClassifier:
    log.info("Tuning XGBoost …")
    param_dist = {
        "n_estimators":     [200, 400, 600],
        "max_depth":        [4, 6, 8],
        "learning_rate":    [0.05, 0.1, 0.2],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }
    base = XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=20, cv=3, scoring="f1_weighted",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    log.info(f"  Best XGB params: {search.best_params_}")
    log.info(f"  CV F1 (weighted): {search.best_score_:.4f}")
    return search.best_estimator_


def tune_random_forest(X_train, y_train) -> RandomForestClassifier:
    log.info("Tuning Random Forest …")
    param_dist = {
        "n_estimators":  [200, 400],
        "max_depth":     [None, 20, 40],
        "min_samples_split": [2, 5],
        "max_features":  ["sqrt", "log2"],
        "class_weight":  ["balanced", "balanced_subsample"],
    }
    base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=15, cv=3, scoring="f1_weighted",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    log.info(f"  Best RF params: {search.best_params_}")
    log.info(f"  CV F1 (weighted): {search.best_score_:.4f}")
    return search.best_estimator_


def tune_svm(X_train, y_train) -> CalibratedClassifierCV:
    """LinearSVC wrapped in CalibratedClassifierCV to get predict_proba."""
    log.info("Tuning SVM …")
    param_dist = {
        "base_estimator__C":        [0.1, 0.5, 1.0, 5.0, 10.0],
        "base_estimator__max_iter": [1000, 2000],
    }
    base = CalibratedClassifierCV(
        LinearSVC(random_state=RANDOM_STATE, class_weight="balanced"),
        cv=3
    )
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=10, cv=3, scoring="f1_weighted",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    log.info(f"  Best SVM params: {search.best_params_}")
    log.info(f"  CV F1 (weighted): {search.best_score_:.4f}")
    return search.best_estimator_


# ─────────────────────────────────────────────────────────────
#  4. Ensemble + evaluation
# ─────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, name: str):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    log.info(f"\n{'='*50}")
    log.info(f"  {name}  —  F1 (weighted): {f1:.4f}")
    log.info(f"{'='*50}")
    log.info("\n" + classification_report(y_test, y_pred,
             target_names=["hate", "offensive", "safe"]))
    return f1


def run_training():
    # 1. Load
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 2. Fit TF-IDF on training data only
    log.info("Fitting TF-IDF vectorizer …")
    tfidf = build_tfidf()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)
    log.info(f"  → Feature matrix: {X_train_tfidf.shape}")

    # 3. SMOTE to balance classes
    log.info("Applying SMOTE to balance classes …")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_tfidf, y_train)
    log.info(f"  → Balanced training set: {X_train_bal.shape[0]:,} rows")

    # 4. Train each model
    xgb = tune_xgboost(X_train_bal, y_train_bal)
    rf  = tune_random_forest(X_train_bal, y_train_bal)
    svm = tune_svm(X_train_bal, y_train_bal)

    # 5. Evaluate individuals
    evaluate(xgb, X_test_tfidf, y_test, "XGBoost")
    evaluate(rf,  X_test_tfidf, y_test, "Random Forest")
    evaluate(svm, X_test_tfidf, y_test, "SVM")

    # 6. Soft-voting ensemble
    log.info("\nBuilding soft-voting ensemble …")
    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf), ("svm", svm)],
        voting="soft",
        weights=[2, 1, 1],   # XGB gets double weight (strongest model)
    )
    ensemble.fit(X_train_bal, y_train_bal)
    ensemble_f1 = evaluate(ensemble, X_test_tfidf, y_test, "ENSEMBLE")

    # 7. Save all artefacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf,    MODELS_DIR / "tfidf.pkl")
    joblib.dump(xgb,      MODELS_DIR / "xgboost.pkl")
    joblib.dump(rf,       MODELS_DIR / "random_forest.pkl")
    joblib.dump(svm,      MODELS_DIR / "svm.pkl")
    joblib.dump(ensemble, MODELS_DIR / "ensemble.pkl")

    # Save confusion matrix data for dashboard
    y_pred = ensemble.predict(X_test_tfidf)
    cm = confusion_matrix(y_test, y_pred)
    np.save(MODELS_DIR / "confusion_matrix.npy", cm)

    log.info(f"\nAll models saved to {MODELS_DIR}/")
    log.info(f"Final ensemble F1: {ensemble_f1:.4f}")
    log.info("Next: python src/api/app.py")


if __name__ == "__main__":
    run_training()
