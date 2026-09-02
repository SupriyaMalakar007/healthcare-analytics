"""
Trains disease-risk classifiers directly from the Postgres tables,
compares a couple of model families per disease, keeps the best one
(by ROC-AUC), and:
  1. saves the winning pipeline (scaler + model) to ml/models/*.joblib
  2. writes metrics for ALL trained candidates into model_metadata
  3. marks the winning model as is_active=True (and deactivates prior ones)

This is a general framework: adding a new disease means adding an
entry to DISEASE_CONFIGS below, plus a matching *_records table.
Nothing else in this file changes.
"""
import os
import sys
import joblib
import pandas as pd
from sqlalchemy import text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.connection import get_engine

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Add a new disease here to extend the framework.
# ------------------------------------------------------------------
DISEASE_CONFIGS = {
    "heart_disease": {
        "table": "heart_records",
        "feature_cols": [
            "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
            "fasting_bs", "resting_ecg", "max_hr", "exercise_angina",
            "oldpeak", "st_slope",
        ],
        "target_col": "target",
    },
    "diabetes": {
        "table": "diabetes_records",
        "feature_cols": [
            "pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "diabetes_pedigree", "age",
        ],
        "target_col": "target",
    },
}

CANDIDATE_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
}


def get_disease_id(engine, disease_code: str) -> int:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT disease_id FROM diseases WHERE disease_code = :code"),
            {"code": disease_code},
        ).fetchone()
    if row is None:
        raise ValueError(f"Disease '{disease_code}' not found in diseases table")
    return row[0]


def train_one_disease(disease_code: str, config: dict, engine):
    print(f"\n=== Training models for: {disease_code} ===")
    df = pd.read_sql(f"SELECT * FROM {config['table']}", engine)
    X = df[config["feature_cols"]]
    y = df[config["target_col"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    disease_id = get_disease_id(engine, disease_code)
    results = []

    for model_name, model in CANDIDATE_MODELS.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        print(f"  {model_name:<20} acc={metrics['accuracy']:.3f}  "
              f"f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")

        results.append({
            "model_name": model_name,
            "pipeline": pipe,
            "metrics": metrics,
        })

    best = max(results, key=lambda r: r["metrics"]["roc_auc"])
    print(f"  -> Best model: {best['model_name']} (ROC-AUC={best['metrics']['roc_auc']:.3f})")

    # Save the winning pipeline (scaler + model bundled together)
    artifact_path = os.path.join(MODELS_DIR, f"{disease_code}_{best['model_name']}.joblib")
    joblib.dump({
        "pipeline": best["pipeline"],
        "feature_cols": config["feature_cols"],
    }, artifact_path)

    # Persist metadata for ALL candidates, deactivate old, activate winner
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE model_metadata SET is_active = FALSE WHERE disease_id = :did"),
            {"did": disease_id},
        )
        for r in results:
            is_winner = r is best
            conn.execute(
                text("""
                    INSERT INTO model_metadata
                        (disease_id, model_name, version, accuracy, precision_score,
                         recall_score, f1_score, roc_auc, is_active, artifact_path)
                    VALUES
                        (:disease_id, :model_name, :version, :accuracy, :precision_score,
                         :recall_score, :f1_score, :roc_auc, :is_active, :artifact_path)
                """),
                {
                    "disease_id": disease_id,
                    "model_name": r["model_name"],
                    "version": "v1",
                    "accuracy": r["metrics"]["accuracy"],
                    "precision_score": r["metrics"]["precision"],
                    "recall_score": r["metrics"]["recall"],
                    "f1_score": r["metrics"]["f1"],
                    "roc_auc": r["metrics"]["roc_auc"],
                    "is_active": is_winner,
                    "artifact_path": artifact_path if is_winner else None,
                },
            )

    return best


if __name__ == "__main__":
    engine = get_engine()
    for code, cfg in DISEASE_CONFIGS.items():
        train_one_disease(code, cfg, engine)
    print("\nAll models trained and registered.")
