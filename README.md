# Premier League Intelligence Platform

An end-to-end football data project covering five completed Premier League seasons (2021-22 through 2025-26). The project combines data collection, cleaning, SQL analysis, interactive visualization, feature engineering, machine learning, model evaluation, and a Streamlit web app.

## What it does

- Downloads and caches five seasons of Premier League match data
- Builds season league tables with SQL
- Explores team performance and multi-season trends
- Creates pre-match rolling features using only information available before each match
- Trains and compares Logistic Regression and Random Forest classifiers
- Uses a time-based train/test split instead of a random split
- Saves the better model and its measured metrics
- Shows analytics and model results in Streamlit

## Tech stack

Python, Pandas, SQLite/SQL, Scikit-learn, Plotly, Streamlit, Requests, Pytest

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Train the ML models once:

```bash
python3 scripts/train_model.py
```

Run tests:

```bash
pytest -q
```

## Why the ML design matters

The match predictor uses rolling form and scoring/conceding averages that are shifted by one match. That prevents the current match result from leaking into its own features. The test split is the most recent 20% of matches, so evaluation is closer to a real forecasting setup than a random split.

## Resume bullets (use only after you run the project)

**Premier League Intelligence Platform | Python, Pandas, SQL, Scikit-learn, Streamlit**

- Built an end-to-end football analytics platform covering five Premier League seasons, combining automated data ingestion, SQL-based standings, interactive visualizations, and team performance analysis.
- Engineered pre-match rolling form and scoring features and compared Logistic Regression and Random Forest models using a chronological train/test split to reduce data leakage.
- Deployed the analytics and model-evaluation workflow in Streamlit and added automated tests for core transformation and feature-building logic.

If you want to mention an accuracy number on your resume, run `python3 scripts/train_model.py` and use the measured test accuracy from `artifacts/model_metrics.json`. Do not invent a number.
