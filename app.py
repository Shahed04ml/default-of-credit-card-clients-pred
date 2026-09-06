import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Credit Default Predictor", page_icon="💳", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "credit_default_rf_pipeline.joblib"

TARGET = "default payment next month"
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

NUM_FEATURES = [
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "LIMIT_BAL", "AGE", "UTILIZATION_1", "UTILIZATION_2", "PAY_RATIO_1",
    "TOTAL_DELAY_MONTHS", "BILL_DIFF"
]
CAT_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"] + PAY_COLS

@st.cache_data
def load_data():
    csv_path = DATA_DIR / "default_of_credit_card_clients.csv"
    xls_path = DATA_DIR / "default of credit card clients.xls"

    if csv_path.exists():
        df = pd.read_csv(csv_path, sep=None, engine="python")
    elif xls_path.exists():
        df = pd.read_excel(xls_path)
    else:
        raise FileNotFoundError(
            "Dataset not found. Put default_of_credit_card_clients.csv "
            "or 'default of credit card clients.xls' in the data folder."
        )

    df.columns = df.columns.astype(str).str.strip()
    df = df.drop_duplicates().copy()

    # Convert all columns to numeric where possible.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    return df

def engineer_features(df):
    out = df.copy()
    out["UTILIZATION_1"] = out["BILL_AMT1"] / (out["LIMIT_BAL"].abs() + 1)
    out["UTILIZATION_2"] = out["BILL_AMT2"] / (out["LIMIT_BAL"].abs() + 1)
    out["PAY_RATIO_1"] = out["PAY_AMT1"] / (out["BILL_AMT2"].abs() + 1)
    out["TOTAL_DELAY_MONTHS"] = (out[PAY_COLS] > 0).sum(axis=1)
    out["BILL_DIFF"] = out["BILL_AMT1"] - out["BILL_AMT2"]
    return out

@st.cache_resource
def train_model():
    df = engineer_features(load_data())

    required = list(dict.fromkeys(NUM_FEATURES + CAT_FEATURES + [TARGET]))
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[NUM_FEATURES + CAT_FEATURES].copy()
    y = df[TARGET].astype(int)

    # Simple cleaning for the application.
    for c in NUM_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())

    for c in CAT_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].mode().iloc[0])

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    return pipeline, df, X_test, y_test

def make_input():
    st.sidebar.header("Client information")

    limit_bal = st.sidebar.number_input("Credit limit (LIMIT_BAL)", min_value=0, value=200000, step=10000)
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)

    sex = st.sidebar.selectbox("Sex", [1, 2], format_func=lambda x: "Male (1)" if x == 1 else "Female (2)")
    education = st.sidebar.selectbox("Education", [1, 2, 3, 4], format_func=lambda x: {
        1:"Graduate school (1)", 2:"University (2)", 3:"High school (3)", 4:"Other (4)"
    }[x])
    marriage = st.sidebar.selectbox("Marriage", [1, 2, 3], format_func=lambda x: {
        1:"Married (1)", 2:"Single (2)", 3:"Other (3)"
    }[x])

    st.sidebar.subheader("Repayment status")
    pays = {}
    for c in PAY_COLS:
        pays[c] = st.sidebar.number_input(
            c, min_value=-2, max_value=9, value=0, step=1,
            help="-1 = paid duly; positive values indicate delayed payment months."
        )

    st.sidebar.subheader("Recent bills and payments")
    bills = {}
    payments = {}
    for i in range(1, 7):
        bills[f"BILL_AMT{i}"] = st.sidebar.number_input(f"BILL_AMT{i}", value=50000, step=1000)
    for i in range(1, 7):
        payments[f"PAY_AMT{i}"] = st.sidebar.number_input(f"PAY_AMT{i}", value=5000, step=500)

    row = {
        "LIMIT_BAL": limit_bal, "AGE": age, "SEX": sex,
        "EDUCATION": education, "MARRIAGE": marriage,
        **pays, **bills, **payments
    }
    return pd.DataFrame([row])

st.title("💳 Credit Card Default Prediction")
st.write(
    "A mini real-world ML application that predicts whether a client is likely "
    "to default on the next credit-card payment."
)

try:
    model, df, _, _ = train_model()
except Exception as e:
    st.error(str(e))
    st.stop()

# Dashboard metrics
c1, c2, c3 = st.columns(3)
c1.metric("Dataset rows", f"{len(df):,}")
c2.metric("Features used", "28")
c3.metric("Model", "Random Forest")

st.divider()

input_df = make_input()

if st.button("Predict default risk", type="primary"):
    x = engineer_features(input_df)[NUM_FEATURES + CAT_FEATURES]

    # Match the same cleaning used during training.
    for c in NUM_FEATURES:
        x[c] = pd.to_numeric(x[c], errors="coerce")
        x[c] = x[c].fillna(df[c].median())
    for c in CAT_FEATURES:
        x[c] = pd.to_numeric(x[c], errors="coerce")
        x[c] = x[c].fillna(df[c].mode().iloc[0])

    probability = float(model.predict_proba(x)[0, 1])
    prediction = int(probability >= 0.5)

    st.subheader("Prediction result")
    left, right = st.columns(2)
    with left:
        if prediction == 1:
            st.error("⚠️ Higher predicted default risk")
        else:
            st.success("✅ Lower predicted default risk")
    with right:
        st.metric("Predicted probability of default", f"{probability:.1%}")

    st.progress(min(max(probability, 0.0), 1.0))

    st.info(
        "Interpretation: the probability is the model's estimated likelihood "
        "of class 1 (default). This should support, not replace, human review."
    )

with st.expander("How the model works"):
    st.markdown("""
    1. **Data cleaning:** remove duplicates, standardize column names, and handle missing values.
    2. **Feature engineering:** create credit utilization, payment-to-bill ratio,
       total delayed-payment months, and bill amount change.
    3. **Preprocessing:** scale numeric variables and one-hot encode categorical variables.
    4. **Model training:** Random Forest with 150 trees, maximum depth 10,
       balanced class weights, and stratified train/test split.
    5. **Prediction:** return class 0/1 and the probability of default.
    """)
