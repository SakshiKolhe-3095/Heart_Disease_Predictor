import os
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="Heart Stroke Predictor — Sakshi ❤️", layout="centered")

# ---------- Helpers ----------
@st.cache_resource
def load_artifacts(model_path="KNN_heart.pkl", scaler_path="scaler.pkl", cols_path="columns.pkl"):
    """Load model, scaler and expected columns. Returns (model, scaler_or_None, expected_columns)."""
    missing = []
    for p in (model_path, cols_path):
        if not Path(p).exists():
            missing.append(p)
    if missing:
        raise FileNotFoundError(f"Missing required file(s): {', '.join(missing)}")

    model = joblib.load(model_path)

    scaler = None
    if Path(scaler_path).exists():
        scaler = joblib.load(scaler_path)

    expected_columns = joblib.load(cols_path)
    if not isinstance(expected_columns, (list, tuple)):
        expected_columns = list(expected_columns)

    return model, scaler, expected_columns

def set_categorical_onehot(input_row, expected_cols, value):
    """
    Find a column in expected_cols that endswith '_' + value and set it to 1.
    Returns True if found and set, False otherwise.
    """
    suffix = "_" + str(value)
    for c in expected_cols:
        if c.endswith(suffix):
            input_row[c] = 1
            return True
    return False

def set_if_exists(input_row, colname, value):
    """Set value if colname exists in expected columns (exact match)."""
    if colname in input_row.index:
        input_row[colname] = value
        return True
    return False

def file_timestamp(path):
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")

# ---------- Load artifacts ----------
try:
    model, scaler, EXPECTED_COLUMNS = load_artifacts()
except Exception as e:
    st.title("Heart Stroke Prediction — Sakshi")
    st.error("Could not load model artifacts.")
    st.exception(e)
    st.info("Make sure the files `KNN_heart.pkl` and `columns.pkl` exist in the app directory.")
    st.stop()

# ---------- Sidebar info ----------
with st.sidebar:
    st.header("Model info")
    try:
        st.write(f"Model type: `{type(model).__name__}`")
        if Path("KNN_heart.pkl").exists():
            st.write("Model file last modified:", file_timestamp("KNN_heart.pkl"))
        if Path("scaler.pkl").exists():
            st.write("Scaler found:", Path("scaler.pkl").name)
        st.write("Expected features (count):", len(EXPECTED_COLUMNS))
        if st.checkbox("Show expected columns"):
            st.code(EXPECTED_COLUMNS)
    except Exception:
        pass
    st.markdown("---")
    st.markdown("**Disclaimer:** This is a demo model for educational purposes — not a medical diagnosis tool.")

# ---------- Page ----------
st.title("❤️ Heart Stroke Risk Predictor")
st.markdown("Fill in the fields and click **Predict**. The model predicts LOW or HIGH risk (class output).")

# Put inputs in a form to avoid immediate reruns
with st.form("input_form"):
    st.subheader("Patient details")
    cols1, cols2, cols3 = st.columns(3)
    with cols1:
        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    with cols2:
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 60, 250, 120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 50, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    with cols3:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    predict_btn = st.form_submit_button("Predict")

# ---------- Prepare input row robustly ----------
def build_input_row():
    # initialize a row of zeros with expected columns
    row = pd.Series(0, index=EXPECTED_COLUMNS, dtype=float)

    # numeric fields (set only if column exists)
    numeric_map = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
    }
    for colname, val in numeric_map.items():
        set_if_exists(row, colname, val)

    # categorical handling: prefer exact numeric column (e.g., 'Sex'), otherwise set one-hot like 'Sex_M'
    # Sex
    if not set_if_exists(row, "Sex", 1 if sex == "M" else 0):
        set_categorical_onehot(row, EXPECTED_COLUMNS, sex)

    # Chest pain type
    if not set_if_exists(row, "ChestPainType", chest_pain):
        set_categorical_onehot(row, EXPECTED_COLUMNS, chest_pain)

    # Resting ECG
    if not set_if_exists(row, "RestingECG", resting_ecg):
        set_categorical_onehot(row, EXPECTED_COLUMNS, resting_ecg)

    # Exercise angina
    if not set_if_exists(row, "ExerciseAngina", 1 if exercise_angina == "Y" else 0):
        set_categorical_onehot(row, EXPECTED_COLUMNS, exercise_angina)

    # ST Slope
    if not set_if_exists(row, "ST_Slope", st_slope):
        set_categorical_onehot(row, EXPECTED_COLUMNS, st_slope)

    return row

# ---------- Prediction (no probability) ----------
def predict_from_row(row):
    X = pd.DataFrame([row[EXPECTED_COLUMNS]])  # ensure order
    # apply scaler if available
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            st.warning("Warning: scaler.transform failed. Using raw features. See console for details.")
            st.exception(e)
            X_scaled = X.values
    else:
        X_scaled = X.values

    pred = model.predict(X_scaled)[0]
    return int(pred)

if predict_btn:
    with st.spinner("Predicting..."):
        try:
            input_row = build_input_row()
            pred_class = predict_from_row(input_row)

            # Show result (class only)
            if pred_class == 1:
                st.error("⚠️ Predicted: HIGH risk of heart disease (class = 1)")
            else:
                st.success("✅ Predicted: LOW risk of heart disease (class = 0)")

            # show input summary and allow download
            st.markdown("**Input summary used for prediction**")
            st.dataframe(pd.DataFrame([input_row[EXPECTED_COLUMNS]]).transpose().rename(columns={0: "value"}))

            csv_bytes = pd.DataFrame([input_row[EXPECTED_COLUMNS]]).to_csv().encode("utf-8")
            st.download_button("Download input as CSV", data=csv_bytes, file_name="input_for_prediction.csv")

        except Exception as e:
            st.exception(e)
            st.error("Prediction failed. Check model artifacts and expected column names.")

# ---------- Batch CSV upload (no probability) ----------
st.markdown("---")
st.header("Batch prediction (CSV upload)")
st.markdown("Upload a CSV that contains columns that match the model's expected features (or subset). Missing columns will be filled with 0s.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        # Prepare features: create df_out with expected columns filled with 0, then copy numeric columns from uploaded df
        X = pd.DataFrame(0, index=df.index, columns=EXPECTED_COLUMNS, dtype=float)

        # copy numeric exact matches
        for col in ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]:
            if col in df.columns:
                X[col] = df[col]

        # For categorical columns in df, try to set one-hot columns accordingly
        for c in df.columns:
            if df[c].dtype == object or df[c].dtype.name == "category":
                for idx, val in df[c].iteritems():
                    if pd.isna(val):
                        continue
                    # try to set exact column
                    if c in X.columns:
                        X.at[idx, c] = val
                        continue
                    # otherwise set one-hot if suffix exists
                    set_categorical_onehot(X.loc[idx], EXPECTED_COLUMNS, val)

        # scale if scaler
        if scaler is not None:
            try:
                X_scaled = scaler.transform(X)
            except Exception:
                st.warning("Scaler transform failed for batch input. Using raw features.")
                X_scaled = X.values
        else:
            X_scaled = X.values

        # predict (class only)
        preds = model.predict(X_scaled)

        out = df.copy()
        out["predicted_class"] = preds
        st.write(out.head(50))
        st.download_button("Download predictions as CSV", out.to_csv(index=False).encode("utf-8"), "predictions.csv")
    except Exception as e:
        st.exception(e)
        st.error("Failed to run batch predictions. Check CSV format and column names.")

# ---------- Footer ----------
st.markdown("---")
st.caption("Built by Sakshi — Demo app. Not for diagnostic use.")
