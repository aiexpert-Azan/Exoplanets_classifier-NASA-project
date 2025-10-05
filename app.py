# app.py
import json
import joblib
import pandas as pd
import streamlit as st
import numpy as np

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="Exoplanet Classifier", layout="centered")
st.title("🪐 Exoplanet Classifier — NASA Catalog Model")
st.write("Upload a NASA KOI/TOI/K2 catalog CSV or enter values manually below.")

# -------------------------------
# Load model
# -------------------------------
try:
    artifact = joblib.load("model.joblib")

    if isinstance(artifact, dict):
        pipeline = artifact.get("pipeline")
        saved_features = artifact.get("features", [])
        label_map = artifact.get("label_map", {"0": "FALSE_POSITIVE", "1": "CANDIDATE", "2": "CONFIRMED"})
    else:
        pipeline = artifact
        saved_features = getattr(pipeline, "feature_names_in_", [])
        label_map = {"0": "FALSE_POSITIVE", "1": "CANDIDATE", "2": "CONFIRMED"}

except Exception as e:
    st.error(f"⚠️ Could not load model.joblib: {e}")
    st.stop()


# Try to get feature order directly from pipeline if available
if hasattr(pipeline, "feature_names_in_"):
    model_features = list(pipeline.feature_names_in_)
else:
    model_features = saved_features

model_features = [str(f).strip() for f in model_features]

# -------------------------------
# Sidebar info
# -------------------------------
st.sidebar.header("🧠 Model Info")
st.sidebar.write(f"Model trained on: {artifact.get('table_used', 'unknown')}")
st.sidebar.write("Feature count: " + str(len(model_features)))
for f in model_features:
    st.sidebar.write(f"- {f}")

# -------------------------------
# Upload section
# -------------------------------
st.header("📂 Upload Catalog CSV")
uploaded = st.file_uploader("Upload a NASA Exoplanet CSV file", type=["csv"])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived log, ratio, and SNR features safely."""
    df = df.copy()

    def safe_log(x):
        x = pd.to_numeric(x, errors="coerce")
        return np.log1p(np.clip(x, 0, None))

    # --- Base log features ---
    for raw, new in [("koi_period", "log_period"),
                     ("koi_duration", "log_duration"),
                     ("koi_prad", "log_prad")]:
        df[new] = safe_log(df[raw]) if raw in df.columns else np.nan

    # --- Depth ratios ---
    if "koi_depth" in df.columns and "koi_prad" in df.columns:
        prad = pd.to_numeric(df["koi_prad"], errors="coerce").replace(0, np.nan)
        depth = pd.to_numeric(df["koi_depth"], errors="coerce")
        df["depth_frac"] = depth / prad
        df["depth_over_prad2"] = depth / (prad ** 2)
    else:
        df["depth_frac"] = np.nan
        df["depth_over_prad2"] = np.nan

    # --- Derived planetary radius ratio ---
    df["rp_rs"] = np.sqrt(np.clip(df["depth_frac"], 0, None))
    df["predicted_depth_from_rprs"] = df["rp_rs"] ** 2
    df["depth_residual"] = df["depth_frac"] - df["predicted_depth_from_rprs"]

    # --- Stellar logs ---
    for raw, new in [("koi_srad", "log_srad"), ("koi_smass", "log_smass")]:
        df[new] = safe_log(df[raw]) if raw in df.columns else np.nan

    # --- Stellar temp ---
    if "koi_teff" in df.columns:
        df["teff"] = pd.to_numeric(df["koi_teff"], errors="coerce")
    elif "koi_steff" in df.columns:
        df["teff"] = pd.to_numeric(df["koi_steff"], errors="coerce")
    else:
        df["teff"] = np.nan

    # --- Signal-to-noise ratio ---
    if "koi_model_snr" in df.columns:
        df["snr"] = pd.to_numeric(df["koi_model_snr"], errors="coerce")
    elif {"koi_depth", "koi_depth_err1"}.issubset(df.columns):
        d = pd.to_numeric(df["koi_depth"], errors="coerce")
        e = pd.to_numeric(df["koi_depth_err1"], errors="coerce").replace(0, np.nan)
        df["snr"] = d / e
    else:
        df["snr"] = np.nan

    # remove inf/nan issues
    df = df.replace([np.inf, -np.inf], np.nan)

    # ✅ remove duplicate column names
    df = df.loc[:, ~df.columns.duplicated()]

    return df


# -------------------------------
# File uploaded case
# -------------------------------
if uploaded is not None:
    df = pd.read_csv(uploaded)
    df.columns = df.columns.str.strip()
    df = engineer_features(df)

    # Deduplicate again just in case
    df = df.loc[:, ~df.columns.duplicated()]

    st.subheader("🧩 Debug Info")
    st.write("Model expects:", model_features)
    st.write("Uploaded CSV columns:", list(df.columns))

    missing = [f for f in model_features if f not in df.columns]
    extra = [c for c in df.columns if c not in model_features]

    st.write("❓ Missing:", missing)
    st.write("⚠️ Extra:", extra)

    if missing:
        st.error(f"Missing features: {missing}")
    else:
        st.success("✅ All required features found — aligning and predicting.")

        # align and ensure correct order
        df = df.reindex(columns=model_features, fill_value=np.nan).astype(float)

        try:
            preds = pipeline.predict(df)
            pred_names = [label_map.get(str(int(p)), str(p)) for p in preds]
            out = df.copy()
            out["predicted_disposition"] = pred_names

            # ✅ Remove any duplicate columns once more — final safety net
            out = out.loc[:, ~out.columns.duplicated()]

            st.subheader("Predictions (first 50 rows)")
            st.dataframe(out.head(50))
            st.download_button(
                "⬇️ Download Predictions CSV",
                out.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error("❌ Prediction failed:")
            st.code(str(e))

# -------------------------------
# Manual input
# -------------------------------
st.header("🧮 Manual Input (Single Sample)")

# Deduplicate model feature names for Streamlit keys
unique_features = []
for f in model_features:
    if f not in unique_features:
        unique_features.append(f)

with st.form("manual_form"):
    values = {}
    for i, f in enumerate(unique_features):
        values[f] = st.number_input(
            f"Enter {f}:",
            value=0.0,
            format="%.6f",
            key=f"manual_{f}_{i}"  # unique key using index
        )

    # ✅ Submit button INSIDE form
    submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        row = pd.DataFrame([values])
        row = row.reindex(columns=model_features, fill_value=np.nan).astype(float)

        try:
            pred = pipeline.predict(row)[0]
            st.success(f"Predicted Disposition: {label_map.get(str(int(pred)), str(pred))}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
