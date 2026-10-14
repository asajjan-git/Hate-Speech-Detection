# Hate-Speech-Detection

End-to-end hate speech detection pipeline — ingests 31K+ tweets into MySQL, applies a 15-step pandas ETL pipeline (55% noise reduction), trains an XGBoost + Random Forest + SVM soft-voting ensemble (F1: 0.88, −32% false positives), serves predictions via a Flask REST API, and visualizes sentiment trends and moderation spikes through a 6-page Streamlit dashboard.
