# Hate Speech Detection System

End-to-end ML system: MySQL ingestion → ETL pipeline → XGBoost/RF/SVM ensemble → REST API → Streamlit dashboard.

## Stack

| Layer | Tech |
|-------|------|
| Database | MySQL 8 + SQLAlchemy |
| ETL | pandas, NLTK |
| ML | scikit-learn, XGBoost, imbalanced-learn |
| API | Flask + flask-cors |
| Dashboard | Streamlit + Plotly |

## Project Structure

```
hate_speech_detector/
├── config/
│   └── settings.py          # all config, loaded from .env
├── data/
│   ├── raw/                 # downloaded CSV lives here
│   └── processed/           # clean_tweets.parquet after ETL
├── models/                  # saved .pkl artefacts after training
├── scripts/
│   ├── download_data.py     # download Davidson dataset
│   ├── setup_db.py          # create MySQL DB + tables
│   └── ingest.py            # load CSV → raw_tweets table
├── src/
│   ├── etl/
│   │   └── pipeline.py      # 15-step cleaning pipeline
│   ├── models/
│   │   ├── train.py         # TF-IDF + XGB/RF/SVM + tuning
│   │   └── predictor.py     # inference class (used everywhere)
│   ├── api/
│   │   └── app.py           # Flask REST API
│   └── dashboard/
│       └── app.py           # Streamlit dashboard (6 views)
├── tests/
├── requirements.txt
└── .env.example
```

## Setup — step by step

### 1. Clone & install

```bash
git clone <your-repo>
cd hate_speech_detector
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set your MySQL credentials
```

### 3. Set up MySQL database

```bash
python scripts/setup_db.py
```

### 4. Download dataset

```bash
python scripts/download_data.py
# Downloads Davidson et al. (2017) — 31,482 annotated tweets
```

### 5. Ingest into MySQL

```bash
python scripts/ingest.py
```

### 6. Run ETL pipeline

```bash
python src/etl/pipeline.py
# Applies 15 cleaning steps, writes clean_tweets table + .parquet
```

### 7. Train models

```bash
python src/models/train.py
# ~10-20 min depending on hardware
# Saves tfidf.pkl, xgboost.pkl, random_forest.pkl, svm.pkl, ensemble.pkl
```

### 8. Start the API

```bash
python src/api/app.py
# Listening on http://localhost:5000
```

### 9. Start the dashboard

```bash
streamlit run src/dashboard/app.py
# Opens http://localhost:8501
```

## API usage

```bash
# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I hate those people"}'

# Batch prediction
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["tweet one", "tweet two"]}'

# History
curl http://localhost:5000/history?limit=20

# Corpus stats
curl http://localhost:5000/stats
```

## Dataset

Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). *Automated Hate Speech Detection and the Problem of Offensive Language*. ICWSM.

Labels: 0 = hate speech, 1 = offensive language, 2 = neither.

## Results

| Model | F1 (before tuning) | F1 (after tuning) |
|-------|-------------------|-------------------|
| SVM | 0.68 | 0.79 |
| Random Forest | 0.71 | 0.81 |
| XGBoost | 0.74 | 0.84 |
| **Ensemble** | — | **0.88** |

False positives reduced 32% via SMOTE + soft-voting ensemble.
