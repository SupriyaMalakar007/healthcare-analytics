"""
Healthcare Analytics & Disease Risk Prediction — Streamlit app.

Pages:
  - Home                 : overview + system status
  - Risk Prediction      : pick a disease, enter patient data, get a live risk score
  - Analytics Dashboard  : charts built from live Postgres data
  - Patient History      : searchable log of every prediction made
"""
import os
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.connection import get_engine, test_connection
from app.model_utils import predict, save_prediction, get_or_create_patient, get_active_model

st.set_page_config(page_title="Healthcare Risk Analytics", page_icon="🩺", layout="wide")

# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.title("🩺 Healthcare Analytics")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Risk Prediction", "Analytics Dashboard", "Patient History"],
)

DISEASE_LABELS = {"heart_disease": "Heart Disease", "diabetes": "Diabetes"}


# ==================================================================
# HOME
# ==================================================================
def page_home():
    st.title("Healthcare Analytics & Disease Risk Prediction")
    st.markdown(
        "A full-stack demo: **PostgreSQL** stores clinical data, predictions and model "
        "metadata; **scikit-learn** trains per-disease risk models; **Streamlit** "
        "serves prediction and analytics on top."
    )

    col1, col2, col3 = st.columns(3)
    db_ok = test_connection()
    col1.metric("Database", "Connected ✅" if db_ok else "Disconnected ❌")

    engine = get_engine()
    try:
        n_predictions = pd.read_sql("SELECT COUNT(*) AS c FROM predictions", engine).iloc[0]["c"]
        n_patients = pd.read_sql("SELECT COUNT(*) AS c FROM patients", engine).iloc[0]["c"]
    except Exception:
        n_predictions, n_patients = 0, 0
    col2.metric("Predictions Logged", int(n_predictions))
    col3.metric("Patients", int(n_patients))

    st.subheader("Active Models")
    for code, label in DISEASE_LABELS.items():
        info = get_active_model(code)
        if info:
            st.markdown(
                f"- **{label}** → `{info['model_name']}` "
                f"(ROC-AUC: {info['roc_auc']:.3f}, Accuracy: {info['accuracy']:.3f})"
            )
        else:
            st.markdown(f"- **{label}** → ⚠️ no active model. Run `python ml/train_models.py`.")

    st.info(
        "Go to **Risk Prediction** to score a patient, or **Analytics Dashboard** "
        "to explore the underlying data and model performance."
    )


# ==================================================================
# RISK PREDICTION
# ==================================================================
def page_prediction():
    st.title("Patient Risk Prediction")

    disease_code = st.selectbox(
        "Select disease", options=list(DISEASE_LABELS.keys()),
        format_func=lambda c: DISEASE_LABELS[c],
    )

    model_info = get_active_model(disease_code)
    if model_info is None:
        st.error("No trained model available for this disease yet. Run `python ml/train_models.py`.")
        return

    st.caption(f"Using model: **{model_info['model_name']}** "
               f"(ROC-AUC {model_info['roc_auc']:.3f})")

    with st.form("patient_form"):
        st.subheader("Patient Info")
        c1, c2, c3 = st.columns(3)
        full_name = c1.text_input("Full name", value="Jane Doe")
        age = c2.number_input("Age", 1, 120, 45)
        sex_display = c3.selectbox("Sex", ["Female", "Male"])

        st.subheader("Clinical Measurements")
        feature_values = {"age": age}

        if disease_code == "heart_disease":
            feature_values["sex"] = 1 if sex_display == "Male" else 0
            c1, c2, c3 = st.columns(3)
            feature_values["chest_pain_type"] = c1.selectbox(
                "Chest pain type", [0, 1, 2, 3],
                format_func=lambda v: ["Typical angina", "Atypical angina",
                                        "Non-anginal pain", "Asymptomatic"][v])
            feature_values["resting_bp"] = c2.number_input("Resting BP (mm Hg)", 80, 220, 130)
            feature_values["cholesterol"] = c3.number_input("Cholesterol (mg/dl)", 100, 600, 240)

            c1, c2, c3 = st.columns(3)
            feature_values["fasting_bs"] = 1 if c1.selectbox(
                "Fasting blood sugar > 120 mg/dl?", ["No", "Yes"]) == "Yes" else 0
            feature_values["resting_ecg"] = c2.selectbox(
                "Resting ECG", [0, 1, 2],
                format_func=lambda v: ["Normal", "ST-T abnormality", "LV hypertrophy"][v])
            feature_values["max_hr"] = c3.number_input("Max heart rate achieved", 60, 220, 150)

            c1, c2, c3 = st.columns(3)
            feature_values["exercise_angina"] = 1 if c1.selectbox(
                "Exercise-induced angina?", ["No", "Yes"]) == "Yes" else 0
            feature_values["oldpeak"] = c2.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, 0.1)
            feature_values["st_slope"] = c3.selectbox(
                "ST slope", [0, 1, 2], format_func=lambda v: ["Upsloping", "Flat", "Downsloping"][v])

        else:  # diabetes
            c1, c2, c3 = st.columns(3)
            feature_values["pregnancies"] = c1.number_input("Pregnancies", 0, 20, 1)
            feature_values["glucose"] = c2.number_input("Glucose (mg/dl)", 40, 250, 110)
            feature_values["blood_pressure"] = c3.number_input("Blood pressure (mm Hg)", 20, 140, 70)

            c1, c2, c3 = st.columns(3)
            feature_values["skin_thickness"] = c1.number_input("Skin thickness (mm)", 0, 100, 20)
            feature_values["insulin"] = c2.number_input("Insulin (mu U/ml)", 0, 900, 80)
            feature_values["bmi"] = c3.number_input("BMI", 10.0, 70.0, 28.0, 0.1)

            feature_values["diabetes_pedigree"] = st.number_input(
                "Diabetes pedigree function (family history score)", 0.0, 3.0, 0.4, 0.01)

        submitted = st.form_submit_button("Predict Risk", type="primary")

    if submitted:
        proba, label, info = predict(disease_code, feature_values)

        color = {"Low": "green", "Moderate": "orange", "High": "red"}[label]
        st.markdown(f"## Risk: :{color}[{label}]  —  {proba:.1%} probability")
        st.progress(min(max(proba, 0.0), 1.0))

        sex_for_db = "male" if sex_display == "Male" else "female"
        patient_id = get_or_create_patient(full_name, age, sex_for_db)
        save_prediction(patient_id, disease_code, info["model_id"], feature_values, proba, label)
        st.success(f"Saved prediction for **{full_name}** to the database.")


# ==================================================================
# ANALYTICS DASHBOARD
# ==================================================================
def page_dashboard():
    st.title("Analytics Dashboard")
    engine = get_engine()

    preds = pd.read_sql("""
        SELECT p.prediction_id, p.risk_probability, p.risk_label, p.predicted_at,
               d.display_name AS disease, pt.age, pt.sex, mm.model_name
        FROM predictions p
        JOIN diseases d ON d.disease_id = p.disease_id
        JOIN patients pt ON pt.patient_id = p.patient_id
        LEFT JOIN model_metadata mm ON mm.model_id = p.model_id
        ORDER BY p.predicted_at DESC
    """, engine)

    if preds.empty:
        st.info("No predictions logged yet. Go make a few on the **Risk Prediction** page first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Predictions", len(preds))
        c2.metric("High Risk Flagged", int((preds.risk_label == "High").sum()))
        c3.metric("Avg. Risk Probability", f"{preds.risk_probability.mean():.1%}")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(preds, names="risk_label", title="Risk Level Distribution",
                         color="risk_label",
                         color_discrete_map={"Low": "#2ecc71", "Moderate": "#f39c12", "High": "#e74c3c"})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.histogram(preds, x="disease", color="risk_label", barmode="group",
                                 title="Predictions by Disease & Risk Level",
                                 color_discrete_map={"Low": "#2ecc71", "Moderate": "#f39c12", "High": "#e74c3c"})
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.scatter(preds, x="age", y="risk_probability", color="disease",
                           title="Risk Probability vs. Age", trendline="ols",
                           labels={"risk_probability": "Predicted Risk"})
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Model Performance")
    metrics_df = pd.read_sql("""
        SELECT d.display_name AS disease, mm.model_name, mm.accuracy,
               mm.precision_score, mm.recall_score, mm.f1_score, mm.roc_auc, mm.is_active
        FROM model_metadata mm
        JOIN diseases d ON d.disease_id = mm.disease_id
        ORDER BY d.display_name, mm.roc_auc DESC
    """, engine)
    if metrics_df.empty:
        st.warning("No trained models yet. Run `python ml/train_models.py`.")
    else:
        st.dataframe(
            metrics_df.style.format({
                "accuracy": "{:.3f}", "precision_score": "{:.3f}",
                "recall_score": "{:.3f}", "f1_score": "{:.3f}", "roc_auc": "{:.3f}",
            }),
            use_container_width=True,
        )
        fig4 = px.bar(metrics_df, x="model_name", y="roc_auc", color="disease", barmode="group",
                      title="ROC-AUC by Model & Disease")
        st.plotly_chart(fig4, use_container_width=True)


# ==================================================================
# PATIENT HISTORY
# ==================================================================
def page_history():
    st.title("Patient History")
    engine = get_engine()

    search = st.text_input("Search by patient name")

    query = """
        SELECT pt.full_name, pt.age, pt.sex, d.display_name AS disease,
               p.risk_probability, p.risk_label, p.predicted_at
        FROM predictions p
        JOIN patients pt ON pt.patient_id = p.patient_id
        JOIN diseases d ON d.disease_id = p.disease_id
    """
    params = {}
    if search:
        query += " WHERE pt.full_name ILIKE :search"
        params["search"] = f"%{search}%"
    query += " ORDER BY p.predicted_at DESC"

    df = pd.read_sql(query, engine, params=params)

    if df.empty:
        st.info("No matching predictions found.")
    else:
        st.dataframe(
            df.style.format({"risk_probability": "{:.1%}"}),
            use_container_width=True,
        )
        st.caption(f"{len(df)} record(s)")


# ------------------------------------------------------------------
PAGES = {
    "Home": page_home,
    "Risk Prediction": page_prediction,
    "Analytics Dashboard": page_dashboard,
    "Patient History": page_history,
}
PAGES[page]()
