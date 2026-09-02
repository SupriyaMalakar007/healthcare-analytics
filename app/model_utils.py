"""
Utilities shared across app pages: loading the active model for a
disease, running predictions, and writing them back to the DB.
"""
import os
import sys
import joblib
import pandas as pd
from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.connection import get_engine

_model_cache = {}


def get_active_model(disease_code: str):
    """Loads (and caches) the currently active model bundle for a disease."""
    if disease_code in _model_cache:
        return _model_cache[disease_code]

    engine = get_engine()
    query = text("""
        SELECT mm.artifact_path, mm.model_name, mm.roc_auc, mm.accuracy, mm.model_id
        FROM model_metadata mm
        JOIN diseases d ON d.disease_id = mm.disease_id
        WHERE d.disease_code = :code AND mm.is_active = TRUE
        ORDER BY mm.trained_at DESC LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {"code": disease_code}).mappings().fetchone()

    if row is None:
        return None

    bundle = joblib.load(row["artifact_path"])
    result = {
        "pipeline": bundle["pipeline"],
        "feature_cols": bundle["feature_cols"],
        "model_name": row["model_name"],
        "roc_auc": row["roc_auc"],
        "accuracy": row["accuracy"],
        "model_id": row["model_id"],
    }
    _model_cache[disease_code] = result
    return result


def risk_label(prob: float) -> str:
    if prob < 0.33:
        return "Low"
    elif prob < 0.66:
        return "Moderate"
    return "High"


def predict(disease_code: str, feature_values: dict):
    """Runs a prediction and returns (probability, label, model_info)."""
    model_info = get_active_model(disease_code)
    if model_info is None:
        raise RuntimeError(
            f"No trained/active model found for '{disease_code}'. "
            "Run `python ml/train_models.py` first."
        )

    row = pd.DataFrame([{col: feature_values[col] for col in model_info["feature_cols"]}])
    proba = float(model_info["pipeline"].predict_proba(row)[0, 1])
    label = risk_label(proba)
    return proba, label, model_info


def save_prediction(patient_id: int, disease_code: str, model_id: int,
                     feature_values: dict, proba: float, label: str):
    import json
    engine = get_engine()
    with engine.begin() as conn:
        disease_id = conn.execute(
            text("SELECT disease_id FROM diseases WHERE disease_code = :code"),
            {"code": disease_code},
        ).scalar()
        conn.execute(
            text("""
                INSERT INTO predictions
                    (patient_id, disease_id, model_id, input_features, risk_probability, risk_label)
                VALUES
                    (:patient_id, :disease_id, :model_id, :input_features, :risk_probability, :risk_label)
            """),
            {
                "patient_id": patient_id,
                "disease_id": disease_id,
                "model_id": model_id,
                "input_features": json.dumps(feature_values),
                "risk_probability": proba,
                "risk_label": label,
            },
        )


def get_or_create_patient(full_name: str, age: int, sex: str) -> int:
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT patient_id FROM patients WHERE full_name = :n AND age = :a AND sex = :s"),
            {"n": full_name, "a": age, "s": sex},
        ).fetchone()
        if row:
            return row[0]
        new_id = conn.execute(
            text("""
                INSERT INTO patients (full_name, age, sex)
                VALUES (:n, :a, :s) RETURNING patient_id
            """),
            {"n": full_name, "a": age, "s": sex},
        ).scalar()
        return new_id
