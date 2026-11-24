"""
src/api/app.py
Flask REST API — exposes model predictions over HTTP.

Endpoints:
  POST /predict          single tweet
  POST /predict/batch    list of tweets
  GET  /history          last N predictions from DB
  GET  /stats            corpus statistics
  GET  /health           liveness probe

Run:
    python src/api/app.py
"""
import sys
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import DB_URL, FLASK_PORT, FLASK_DEBUG, LOG_LEVEL
from src.models.predictor import predictor
from src.etl.pipeline import clean_tweet

logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

app   = Flask(__name__)
CORS(app)
db    = create_engine(DB_URL)


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def save_prediction(tweet_text: str, pred) -> None:
    """Persist prediction to DB for history / audit."""
    try:
        sql = text("""
            INSERT INTO predictions
              (tweet_text, predicted_label, confidence,
               xgb_prob_hate, rf_prob_hate, svm_prob_hate)
            VALUES
              (:tweet, :label, :conf, :xgb, :rf, :svm)
        """)
        individual = predictor.individual_probs(clean_tweet(tweet_text))
        with db.begin() as conn:
            conn.execute(sql, {
                "tweet": tweet_text[:500],
                "label": pred.label,
                "conf":  pred.confidence,
                "xgb":   individual["xgboost"][0],
                "rf":    individual["random_forest"][0],
                "svm":   individual["svm"][0],
            })
    except Exception as e:
        log.warning(f"Could not save prediction: {e}")


# ─────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Body: { "text": "some tweet" }
    Returns: label, confidence, probabilities
    """
    body = request.get_json(force=True)
    text = body.get("text", "").strip()

    if not text:
        return jsonify({"error": "text field is required"}), 400
    if len(text) > 560:
        return jsonify({"error": "text too long (max 560 chars)"}), 400

    cleaned = clean_tweet(text)
    pred    = predictor.predict(cleaned)

    save_prediction(text, pred)

    return jsonify({
        "input":    text,
        "label":    pred.label,
        "confidence": round(pred.confidence, 4),
        "probabilities": {k: round(v, 4) for k, v in pred.probabilities.items()},
    })


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Body: { "texts": ["tweet1", "tweet2", ...] }
    Max 100 texts per call.
    """
    body  = request.get_json(force=True)
    texts = body.get("texts", [])

    if not texts or not isinstance(texts, list):
        return jsonify({"error": "texts must be a non-empty list"}), 400
    if len(texts) > 100:
        return jsonify({"error": "max 100 texts per batch"}), 400

    cleaned  = [clean_tweet(t) for t in texts]
    preds    = predictor.predict_batch(cleaned)

    results = []
    for original, pred in zip(texts, preds):
        results.append({
            "input":        original,
            "label":        pred.label,
            "confidence":   round(pred.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in pred.probabilities.items()},
        })

    return jsonify({"results": results, "count": len(results)})


@app.route("/history")
def history():
    """Returns the last N predictions from the DB."""
    limit = min(int(request.args.get("limit", 50)), 200)
    try:
        with db.connect() as conn:
            rows = conn.execute(text(
                "SELECT tweet_text, predicted_label, confidence, created_at "
                "FROM predictions ORDER BY created_at DESC LIMIT :lim"
            ), {"lim": limit}).fetchall()
        return jsonify([
            {"text": r[0], "label": r[1], "confidence": r[2],
             "timestamp": str(r[3])}
            for r in rows
        ])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/stats")
def stats():
    """Basic corpus stats for the dashboard."""
    try:
        with db.connect() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM raw_tweets")).scalar()
            dist  = conn.execute(text(
                "SELECT raw_label, COUNT(*) as cnt FROM raw_tweets GROUP BY raw_label"
            )).fetchall()
            pred_count = conn.execute(text(
                "SELECT COUNT(*) FROM predictions"
            )).scalar()
        label_map = {0: "hate", 1: "offensive", 2: "safe"}
        return jsonify({
            "total_tweets": total,
            "predictions_made": pred_count,
            "distribution": {label_map.get(r[0], str(r[0])): r[1] for r in dist},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info(f"Starting API on port {FLASK_PORT} …")
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=FLASK_DEBUG)
