"""
Loads the generated CSVs into the Postgres *_records tables.
Run after generate_synthetic_data.py (or after dropping in real
UCI Heart Disease / Pima Diabetes CSVs with matching column names).
"""
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.connection import get_engine

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_table(csv_name: str, table_name: str):
    path = os.path.join(DATA_DIR, csv_name)
    df = pd.read_csv(path)
    engine = get_engine()
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Loaded {len(df):>5} rows from {csv_name:<20} -> table '{table_name}'")


if __name__ == "__main__":
    load_table("heart.csv", "heart_records")
    load_table("diabetes.csv", "diabetes_records")
    print("Done.")
