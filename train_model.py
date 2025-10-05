# train_model.py
"""
Train a beginner-friendly exoplanet classifier on NASA KOI/TOI/K2 catalog tables.

Usage:
    python train_model.py            # downloads KOI cumulative and trains
    python train_model.py --table toi  # use TOI table instead (TFOPWG disposition)
    python train_model.py --limit 2000 # limit rows for quick runs (useful for testing)
"""
import numpy as np
import pandas as pd

# Conversion constants (computed precisely)
EARTH_RADIUS_TO_SOLAR_RADIUS = 0.0091576829093    # 1 R_earth = 0.00915768 R_sun
EARTH_MASS_TO_SOLAR_MASS = 3.0034096566707067e-06 # 1 M_earth  = 3.0034e-6 M_sun

def _pick_col(df, candidates):
    """Return the first column name from candidates that exists in df, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def compute_engineered_features(df):
    """
    Given a pandas DataFrame from KOI/TOI/K2 tables, add engineered features:
      - log_period, log_duration, log_prad
      - depth_frac (fractional; converts ppm -> fraction)
      - depth_over_prad2
      - snr (best available column or approximation)
      - rp_rs, predicted_depth_from_rprs, depth_residual
      - log_srad, log_smass, teff
    Returns a new DataFrame (copy) with added columns.
    """
    df = df.copy()

    # --- common candidate column names for different catalogs ---
    period_col = _pick_col(df, ['koi_period', 'period', 'pl_orbper', 'orbital_period', 'period_days'])
    duration_col = _pick_col(df, ['koi_duration', 'duration', 'pl_trandur', 'transit_duration'])
    prad_col = _pick_col(df, ['koi_prad', 'pl_rade', 'pl_radj', 'prad', 'planet_radius'])
    depth_col = _pick_col(df, ['koi_depth', 'pl_trandep', 'transit_depth', 'depth'])
    depth_err_col = _pick_col(df, ['koi_depth_err1', 'koi_depth_err', 'pl_trandeperr1', 'pl_trandep_err', 'depth_err'])
    snr_col = _pick_col(df, ['koi_model_snr', 'koi_snr', 'snr', 'pl_tran_snr', 'koi_mes', 'mes'])
    srad_col = _pick_col(df, ['koi_srad', 'st_rad', 'st_radius', 'stellar_radius'])
    smass_col = _pick_col(df, ['koi_smass', 'st_mass', 'stellar_mass'])
    steff_col = _pick_col(df, ['koi_steff', 'st_teff', 'teff', 'stellar_teff'])
    # epsilon to prevent division by zero
    eps = 1e-12

    # --- safe log transforms (avoid log(0) or negatives) ---
    if period_col:
        df['log_period'] = np.where(df[period_col].astype(float) > 0,
                                    np.log10(df[period_col].astype(float)),
                                    np.nan)
    else:
        df['log_period'] = np.nan

    if duration_col:
        df['log_duration'] = np.where(df[duration_col].astype(float) > 0,
                                      np.log10(df[duration_col].astype(float)),
                                      np.nan)
    else:
        df['log_duration'] = np.nan

    if prad_col:
        df['log_prad'] = np.where(df[prad_col].astype(float) > 0,
                                  np.log10(df[prad_col].astype(float)),
                                  np.nan)
    else:
        df['log_prad'] = np.nan

    # --- depth: convert to fractional depth if necessary ---
    if depth_col:
        depth = df[depth_col].astype(float)
        # Heuristic: if depth values are > 1 -> likely in ppm (catalogs frequently store depth in ppm)
        # Convert ppm -> fraction by dividing by 1e6
        df['depth_frac'] = np.where(depth > 1, depth / 1e6, depth)
        # normalized depth per radius^2 (proxy for consistency with expected (Rp/Rs)^2)
        if prad_col:
            df['depth_over_prad2'] = df['depth_frac'] / (df[prad_col].astype(float)**2 + eps)
        else:
            df['depth_over_prad2'] = np.nan
    else:
        df['depth_frac'] = np.nan
        df['depth_over_prad2'] = np.nan

    # --- SNR: pick best available or approximate if possible ---
    if snr_col:
        df['snr'] = df[snr_col].astype(float)
    elif depth_col and depth_err_col:
        # fallback approximation: depth / depth_err
        df['snr'] = df[depth_col].astype(float) / (df[depth_err_col].astype(float) + eps)
    else:
        df['snr'] = np.nan

    # --- Rp/Rs and predicted depth (needs planet radius and stellar radius) ---
    if prad_col and srad_col:
        # convert prad (likely in Earth radii) -> Solar radii using constant above
        df['rp_rs'] = (df[prad_col].astype(float) * EARTH_RADIUS_TO_SOLAR_RADIUS) / (df[srad_col].astype(float) + eps)
        df['predicted_depth_from_rprs'] = df['rp_rs']**2
        if depth_col:
            df['depth_residual'] = df['depth_frac'] - df['predicted_depth_from_rprs']
        else:
            df['depth_residual'] = np.nan
    else:
        df['rp_rs'] = np.nan
        df['predicted_depth_from_rprs'] = np.nan
        df['depth_residual'] = np.nan

    # --- stellar derived features (logs where helpful) ---
    if srad_col:
        df['log_srad'] = np.where(df[srad_col].astype(float) > 0,
                                  np.log10(df[srad_col].astype(float)),
                                  np.nan)
    else:
        df['log_srad'] = np.nan

    if smass_col:
        df['log_smass'] = np.where(df[smass_col].astype(float) > 0,
                                   np.log10(df[smass_col].astype(float)),
                                   np.nan)
    else:
        df['log_smass'] = np.nan

    if steff_col:
        df['teff'] = df[steff_col].astype(float)
    else:
        df['teff'] = np.nan

    # Return expanded DataFrame
    return df

import argparse
import io
import json
import os
from typing import List

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_API = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"

# mapping of table -> (target_column, readable_name)
TABLE_INFO = {
    "cumulative": ("koi_disposition", "KOI (Kepler cumulative)"),
    "toi": ("tfopwg_disp", "TOI (TESS Objects of Interest)"),
    "k2pandc": ("disposition", "K2 Planets & Candidates"),
}


def fetch_table_csv(table: str, params: dict = None, limit: int = None) -> pd.DataFrame:
    """Download a table from the Exoplanet Archive API and return a pandas DataFrame."""
    params = params or {}
    params.update({"table": table, "format": "csv"})
    # If user asked to limit rows for quick experiments, use 'where' to filter something simple.
    # (The API supports SQL-like where clauses; here we won't set one unless limit is provided.)
    r = requests.get(BASE_API, params=params, timeout=60)
    r.raise_for_status()
    # The API returns CSV text
    df = pd.read_csv(io.StringIO(r.text))
    if limit is not None and limit > 0:
        df = df.sample(n=min(limit, len(df)), random_state=42)
    return df


def choose_features(df: pd.DataFrame, candidate_columns: List[str]) -> List[str]:
    """Pick features that exist in the DataFrame from a candidate list."""
    return [c for c in candidate_columns if c in df.columns]


def main(table="cumulative", limit=None):
    if table not in TABLE_INFO:
        raise SystemExit(f"Unknown table '{table}'. Valid: {list(TABLE_INFO.keys())}")

    target_col, pretty = TABLE_INFO[table]
    print(f"[1/6] Downloading table '{table}' -> target column '{target_col}' ({pretty})")
    df = fetch_table_csv(table, limit=limit)
    df = compute_engineered_features(df)
    print(f"Downloaded {len(df)} rows and {len(df.columns)} columns.")

    # Filter rows that actually have a target label
    df = df[df[target_col].notnull()].copy()
    print(f"Rows with label: {len(df)}")

    # Candidate features (common names across KOI/TOI/K2). We'll auto-detect what's present.
    candidate_features = [
        # Kepler/KOI style
        "koi_period", "koi_duration", "koi_prad", "koi_depth", "koi_model_snr", "koi_teq",
        # TOI/K2 / generic names
        "period", "duration", "pl_orbper", "pl_rade", "pl_bmasse",
        "pl_trandep", "pl_trandur", "snr",
        # stellar features (helpful)
        "koi_steff", "st_teff", "st_logg", "st_mass", "st_rad"
        "log_period", "log_duration", "log_prad", "depth_frac", "depth_over_prad2",
        "snr", "rp_rs", "predicted_depth_from_rprs", "depth_residual",
        "log_srad", "log_smass", "teff"
    ]

    features = choose_features(df, candidate_features)
    if len(features) < 2:
        raise SystemExit("Not enough common features found in the table. Try a different table or download the CSV manually.")

    print("Using features:", features)

    # Keep only rows where these features are not all NaN
    df = df[df[features].notnull().any(axis=1)].copy()

    # Target mapping: map disposition strings to integers (FP=0, CANDIDATE=1, CONFIRMED=2)
    raw_labels = df[target_col].astype(str).str.upper()
    # Handle common label variants
    def map_label(s):
        if "FALSE" in s or "FP" in s or "FA" in s or "REFUTED" in s:
            return 0
        if "CANDIDATE" in s or s == "PC" or "CAND" in s:
            return 1
        if "CONFIRMED" in s or "CP" in s or "KNOWN" in s:
            return 2
        # else fallback
        return np.nan

    y = raw_labels.apply(map_label)
    valid = y.notnull()
    df = df.loc[valid]
    y = y.loc[valid].astype(int)
    X = df[features].astype(float)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Build pipeline: median impute -> scale -> RandomForest
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42,
                                       class_weight="balanced_subsample", n_jobs=-1))
    ])

    print("[4/6] Training RandomForest...")
    pipeline.fit(X_train, y_train)

    print("[5/6] Evaluating...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=true, cols=pred):\n", cm)

    # Cross-validation (quick)
    print("5-fold cross-val accuracy (pipeline):")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(cv_scores, "mean", cv_scores.mean())

    # Save the trained pipeline + feature list + target column + label mapping
    artifact = {
        "pipeline": pipeline,
        "features": features,
        "target_col": target_col,
        "table_used": table,
        "label_map": {"0": "FALSE_POSITIVE", "1": "CANDIDATE", "2": "CONFIRMED"}
    }
    joblib.dump(artifact, "model.joblib")
    print("[6/6] Saved artifact to model.joblib")

    metrics = {
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": cm.tolist(),
        "cv_scores": cv_scores.tolist(),
        "cv_mean": float(cv_scores.mean())
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default="cumulative", help="Table to use: cumulative (KOI), toi, k2pandc")
    parser.add_argument("--limit", type=int, default=None, help="Downsample the downloaded rows for quick experiments")
    args = parser.parse_args()
    main(table=args.table, limit=args.limit)
