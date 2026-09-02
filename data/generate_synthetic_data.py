"""
Generates synthetic-but-clinically-plausible datasets for:
  - Heart disease  (features modeled on the UCI Heart Disease dataset schema)
  - Diabetes       (features modeled on the Pima Indians Diabetes dataset schema)

Why synthetic: this keeps the project self-contained and runnable offline,
with no license/download friction. The feature schemas match the real,
well-known public datasets exactly, so swapping in the real CSVs later
(see README) is a drop-in replacement — no code changes needed downstream.

Labels are generated from a logistic function of clinically-sensible risk
factors plus noise, so the data has genuine, learnable signal (not random).
"""
import numpy as np
import pandas as pd

np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_heart_data(n=1000) -> pd.DataFrame:
    age = np.random.randint(29, 78, n)
    sex = np.random.binomial(1, 0.68, n)  # 1 = male
    chest_pain_type = np.random.choice([0, 1, 2, 3], n, p=[0.47, 0.17, 0.28, 0.08])
    resting_bp = np.random.normal(131, 17, n).clip(94, 200).round().astype(int)
    cholesterol = np.random.normal(246, 52, n).clip(126, 564).round().astype(int)
    fasting_bs = np.random.binomial(1, 0.15, n)
    resting_ecg = np.random.choice([0, 1, 2], n, p=[0.5, 0.48, 0.02])
    max_hr = np.random.normal(150, 23, n).clip(71, 202).round().astype(int)
    exercise_angina = np.random.binomial(1, 0.33, n)
    oldpeak = np.random.exponential(1.0, n).clip(0, 6.2).round(1)
    st_slope = np.random.choice([0, 1, 2], n, p=[0.07, 0.47, 0.46])

    # Risk score built from known cardiovascular risk factors
    risk = (
        -1.6
        + 0.035 * (age - 54)
        + 0.9 * sex
        + 0.55 * (chest_pain_type == 0).astype(int)
        + 0.02 * (resting_bp - 131)
        + 0.008 * (cholesterol - 246)
        + 0.4 * fasting_bs
        + 0.9 * exercise_angina
        + 0.5 * oldpeak
        + 0.7 * (st_slope == 1).astype(int)
        - 0.02 * (max_hr - 150)
        + np.random.normal(0, 1.1, n)
    )
    target = (sigmoid(risk) > 0.5).astype(int)

    return pd.DataFrame({
        "age": age, "sex": sex, "chest_pain_type": chest_pain_type,
        "resting_bp": resting_bp, "cholesterol": cholesterol,
        "fasting_bs": fasting_bs, "resting_ecg": resting_ecg,
        "max_hr": max_hr, "exercise_angina": exercise_angina,
        "oldpeak": oldpeak, "st_slope": st_slope, "target": target,
    })


def generate_diabetes_data(n=1000) -> pd.DataFrame:
    pregnancies = np.random.poisson(3.3, n).clip(0, 17)
    glucose = np.random.normal(120, 32, n).clip(44, 199).round(1)
    blood_pressure = np.random.normal(69, 19, n).clip(24, 122).round(1)
    skin_thickness = np.random.normal(20, 16, n).clip(0, 99).round(1)
    insulin = np.random.exponential(80, n).clip(0, 846).round(1)
    bmi = np.random.normal(32, 7.9, n).clip(18, 67).round(1)
    diabetes_pedigree = np.random.exponential(0.47, n).clip(0.08, 2.42).round(3)
    age = np.random.randint(21, 81, n)

    risk = (
        -2.1
        + 0.04 * (glucose - 120)
        + 0.05 * (bmi - 32)
        + 0.03 * (age - 33)
        + 0.15 * pregnancies
        + 1.1 * diabetes_pedigree
        + 0.01 * (blood_pressure - 69)
        + np.random.normal(0, 1.3, n)
    )
    target = (sigmoid(risk) > 0.5).astype(int)

    return pd.DataFrame({
        "pregnancies": pregnancies, "glucose": glucose,
        "blood_pressure": blood_pressure, "skin_thickness": skin_thickness,
        "insulin": insulin, "bmi": bmi,
        "diabetes_pedigree": diabetes_pedigree, "age": age, "target": target,
    })


if __name__ == "__main__":
    heart_df = generate_heart_data(1000)
    diabetes_df = generate_diabetes_data(1000)

    heart_df.to_csv("data/heart.csv", index=False)
    diabetes_df.to_csv("data/diabetes.csv", index=False)

    print(f"heart.csv    -> {len(heart_df)} rows, positive rate {heart_df.target.mean():.2%}")
    print(f"diabetes.csv -> {len(diabetes_df)} rows, positive rate {diabetes_df.target.mean():.2%}")
