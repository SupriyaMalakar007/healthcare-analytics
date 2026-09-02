# Healthcare Analytics & Disease Risk Prediction

A full-stack demo combining **PostgreSQL**, **Python/scikit-learn**, and **Streamlit**
to predict disease risk (heart disease, diabetes) and visualize results.

Built and smoke-tested end-to-end against a real local PostgreSQL instance —
schema applied, data loaded, models trained, predictions written and read back,
and the Streamlit app booted and served successfully.

## Architecture

```
healthcare-analytics/
├── db/
│   ├── schema.sql          # Postgres schema: patients, *_records, model_metadata, predictions
│   └── connection.py       # SQLAlchemy engine, reads config from .env
├── data/
│   └── generate_synthetic_data.py   # generates heart.csv / diabetes.csv
├── scripts/
│   └── load_data_to_db.py  # loads CSVs into Postgres tables
├── ml/
│   ├── train_models.py     # trains + compares models per disease, logs metrics to DB
│   └── models/              # saved .joblib pipelines (created after training)
├── app/
│   ├── streamlit_app.py    # main app: Home / Predict / Dashboard / History
│   └── model_utils.py      # shared prediction + DB helper functions
├── requirements.txt
├── .env.example
└── README.md
```

**Data flow:** CSV → Postgres (`*_records` tables) → `train_models.py` reads from
Postgres, trains & evaluates models, saves the winning pipeline to disk, and logs
metrics for every candidate to `model_metadata` → Streamlit reads the active model
for live predictions, and writes every prediction back to `predictions` (linked to
`patients`), which powers the Analytics Dashboard and Patient History pages.

**Why this design:** disease-specific tables/feature lists are isolated per disease
config (see `DISEASE_CONFIGS` in `ml/train_models.py`), so adding a new disease is:
add a `diseases` row + a `<name>_records` table + one config entry — nothing else
in the pipeline or app changes.

## Setup

### 1. Database

Requires PostgreSQL 13+.

```bash
createdb healthcare_analytics
psql -d healthcare_analytics -f db/schema.sql
```

Copy `.env.example` to `.env` and fill in your connection details:

```bash
cp .env.example .env
# edit .env with your DB_USER / DB_PASSWORD / DB_HOST / DB_PORT / DB_NAME
```

> **MySQL users:** the schema uses Postgres-specific types (`SERIAL`, `JSONB`,
> `TIMESTAMP DEFAULT NOW()`). For MySQL, swap `SERIAL` → `AUTO_INCREMENT`,
> `JSONB` → `JSON`, and use `mysql+pymysql://...` in `DATABASE_URL`
> (`pip install pymysql`). The Python code itself is DB-agnostic via SQLAlchemy.

### 2. Python environment

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 3. Generate data and load it into Postgres

```bash
python data/generate_synthetic_data.py
python scripts/load_data_to_db.py
```

This ships with **synthetic-but-realistic** data (feature schemas match the
well-known UCI Heart Disease and Pima Indians Diabetes datasets exactly, labels
generated from a logistic function of real clinical risk factors) so the project
runs immediately with no download/license friction. **To use real data**, drop
CSVs with matching column names into `data/heart.csv` and `data/diabetes.csv`
and re-run `load_data_to_db.py` — no other code changes needed.

### 4. Train models

```bash
python ml/train_models.py
```

Trains Logistic Regression and Random Forest per disease, keeps the best by
ROC-AUC, saves it to `ml/models/`, and logs every candidate's metrics into
`model_metadata`.

### 5. Run the app

```bash
streamlit run app/streamlit_app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`).

## App pages

- **Home** — system status, active model summary
- **Risk Prediction** — pick a disease, fill in a patient form, get a live risk
  score (Low/Moderate/High + probability); saved to the DB automatically
- **Analytics Dashboard** — risk distribution, risk-vs-age trends, model
  performance comparison — all queried live from Postgres
- **Patient History** — searchable log of every prediction ever made

## Extending to a new disease

1. Add a row to `diseases` in `db/schema.sql` (or via SQL)
2. Create a `<disease>_records` table with your feature columns + `target`
3. Load training data into it
4. Add an entry to `DISEASE_CONFIGS` in `ml/train_models.py`
5. Add the disease to `DISEASE_LABELS` in `app/streamlit_app.py` and its input
   fields in `page_prediction()`
6. Run `train_models.py` again

## Notes

- Risk thresholds (Low <33%, Moderate 33–66%, High >66%) are in
  `app/model_utils.py::risk_label` — tune for your use case.
- This is a **demo/portfolio project**, not a medical device — predictions
  are illustrative and should never inform real clinical decisions.
