-- ============================================================
-- Healthcare Analytics & Disease Risk Prediction — DB Schema
-- Target: PostgreSQL 13+
-- ============================================================

DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS model_metadata CASCADE;
DROP TABLE IF EXISTS diabetes_records CASCADE;
DROP TABLE IF EXISTS heart_records CASCADE;
DROP TABLE IF EXISTS patients CASCADE;
DROP TABLE IF EXISTS diseases CASCADE;

-- ---------------------------------------------------------------
-- Core reference table: which diseases the platform supports.
-- Adding a new disease = one row here + a *_records table + a
-- training script. The app/DB layer is otherwise disease-agnostic.
-- ---------------------------------------------------------------
CREATE TABLE diseases (
    disease_id      SERIAL PRIMARY KEY,
    disease_code    VARCHAR(50) UNIQUE NOT NULL,   -- 'heart_disease', 'diabetes'
    display_name    VARCHAR(100) NOT NULL,
    description     TEXT
);

-- ---------------------------------------------------------------
-- Patients: minimal demographic record. In a real system this
-- would link out to an EHR; here it anchors predictions/history.
-- ---------------------------------------------------------------
CREATE TABLE patients (
    patient_id      SERIAL PRIMARY KEY,
    full_name       VARCHAR(150) NOT NULL,
    age             INT NOT NULL CHECK (age BETWEEN 0 AND 120),
    sex             VARCHAR(10) NOT NULL CHECK (sex IN ('male', 'female')),
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------
-- Disease-specific feature tables (training/reference data).
-- Kept separate per-disease since feature sets differ; this is
-- the "wide" clinical data used to train each model.
-- ---------------------------------------------------------------
CREATE TABLE heart_records (
    record_id       SERIAL PRIMARY KEY,
    age             INT NOT NULL,
    sex             INT NOT NULL,             -- 1 = male, 0 = female
    chest_pain_type INT NOT NULL,             -- 0-3
    resting_bp      INT NOT NULL,             -- mm Hg
    cholesterol     INT NOT NULL,             -- mg/dl
    fasting_bs      INT NOT NULL,             -- 1 if >120 mg/dl else 0
    resting_ecg     INT NOT NULL,             -- 0-2
    max_hr          INT NOT NULL,
    exercise_angina INT NOT NULL,             -- 1 = yes, 0 = no
    oldpeak         FLOAT NOT NULL,           -- ST depression
    st_slope        INT NOT NULL,             -- 0-2
    target          INT NOT NULL              -- 1 = disease present
);

CREATE TABLE diabetes_records (
    record_id           SERIAL PRIMARY KEY,
    pregnancies         INT NOT NULL,
    glucose             FLOAT NOT NULL,
    blood_pressure       FLOAT NOT NULL,
    skin_thickness       FLOAT NOT NULL,
    insulin             FLOAT NOT NULL,
    bmi                 FLOAT NOT NULL,
    diabetes_pedigree    FLOAT NOT NULL,
    age                 INT NOT NULL,
    target              INT NOT NULL          -- 1 = diabetic
);

-- ---------------------------------------------------------------
-- Model metadata: one row per trained model version, so the app
-- and dashboard can show which model is live and how it performs.
-- ---------------------------------------------------------------
CREATE TABLE model_metadata (
    model_id        SERIAL PRIMARY KEY,
    disease_id      INT NOT NULL REFERENCES diseases(disease_id),
    model_name      VARCHAR(100) NOT NULL,     -- 'RandomForest', 'LogisticRegression'
    version         VARCHAR(20) NOT NULL,
    accuracy        FLOAT,
    precision_score FLOAT,
    recall_score    FLOAT,
    f1_score        FLOAT,
    roc_auc         FLOAT,
    is_active       BOOLEAN NOT NULL DEFAULT FALSE,
    trained_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    artifact_path   VARCHAR(255)               -- path to saved .joblib file
);

-- ---------------------------------------------------------------
-- Predictions: every risk score the app has ever produced, tied
-- to a patient + disease + the model version that produced it.
-- This is what powers "Patient History" and the analytics
-- dashboard's real-world usage charts.
-- ---------------------------------------------------------------
CREATE TABLE predictions (
    prediction_id   SERIAL PRIMARY KEY,
    patient_id      INT NOT NULL REFERENCES patients(patient_id),
    disease_id      INT NOT NULL REFERENCES diseases(disease_id),
    model_id        INT REFERENCES model_metadata(model_id),
    input_features  JSONB NOT NULL,            -- raw feature values submitted
    risk_probability FLOAT NOT NULL,           -- 0.0 - 1.0
    risk_label      VARCHAR(20) NOT NULL,      -- 'Low' / 'Moderate' / 'High'
    predicted_at    TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_predictions_patient ON predictions(patient_id);
CREATE INDEX idx_predictions_disease ON predictions(disease_id);
CREATE INDEX idx_predictions_date ON predictions(predicted_at);

-- Seed the diseases table
INSERT INTO diseases (disease_code, display_name, description) VALUES
    ('heart_disease', 'Heart Disease', 'Cardiovascular disease risk based on clinical measurements'),
    ('diabetes', 'Diabetes', 'Type 2 diabetes risk based on diagnostic measurements');
