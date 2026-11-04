from __future__ import annotations
"""Review 3 regression workflow for EV remaining-range prediction.

Run: python review3_ml_analysis.py --data EV_Range_Prediction_Dataset_2000.csv --out .
The default input is the full repository dataset bundled with this package.
"""
import joblib
import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor

TARGET = "remaining_range"
NUMERIC = ["battery_soc", "voltage", "current", "power_consumption", "speed",
           "distance_traveled", "road_gradient", "temperature", "humidity",
           "wind_speed", "vehicle_load", "battery_health", "regen_braking"]
CATEGORICAL = ["traffic_density", "driving_style"]
FEATURES = NUMERIC + CATEGORICAL


def load_data(path: Path) -> pd.DataFrame:
    # Accept either the repository CSV or the tab-separated pasted attachment.
    return pd.read_csv(path, sep=None, engine="python")


def build_preprocessor(scale: bool) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), NUMERIC),
        ("cat", cat, CATEGORICAL),
    ], remainder="drop")


def get_default_data_path() -> str:
    candidate_data = Path(__file__).parent.parent / "data" / "EV_Range_Prediction_Dataset_2000.csv"
    if candidate_data.exists():
        return str(candidate_data)
    return str(Path(__file__).parent / "EV_Range_Prediction_Dataset_2000.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=get_default_data_path())
    ap.add_argument("--out", default=str(Path(__file__).parent))
    args = ap.parse_args()
    source, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    df = load_data(source)
    required = set(FEATURES + [TARGET])
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {sorted(missing_cols)}")
    n_loaded = len(df)
    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # Preserve original row ordering for a safe chronological holdout.
    df["_source_row"] = np.arange(len(df))
    # Numeric coercion reveals malformed values as missing and enables profile checks.
    for c in NUMERIC + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    target_missing = int(df[TARGET].isna().sum())
    df = df.loc[df[TARGET].notna()].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df.to_csv(out / "cleaned_range_data.csv", index=False)

    profile = pd.DataFrame({
        "dtype": df[FEATURES + [TARGET]].dtypes.astype(str),
        "missing_count": df[FEATURES + [TARGET]].isna().sum(),
        "missing_percent": (100 * df[FEATURES + [TARGET]].isna().mean()).round(2),
        "unique_count": df[FEATURES + [TARGET]].nunique(dropna=True),
    })
    profile.to_csv(out / "data_profile.csv")
    df[FEATURES + [TARGET]].describe(include="all").T.to_csv(out / "descriptive_statistics.csv")

    # Normality diagnostics (Shapiro-Wilk); sample at most 5,000 to respect test limits.
    normals = []
    for col in NUMERIC + [TARGET]:
        vals = df[col].dropna().to_numpy()
        if len(vals) < 3:
            normals.append({"feature": col, "n": len(vals), "W": np.nan, "p_value": np.nan,
                            "interpretation": "insufficient observations"})
            continue
        if len(vals) > 5000:
            vals = np.random.default_rng(42).choice(vals, 5000, replace=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w, p = shapiro(vals)
        normals.append({"feature": col, "n": len(vals), "W": w, "p_value": p,
                        "interpretation": "evidence against normality (p<0.05)" if p < .05 else "no evidence against normality (p>=0.05)"})
    pd.DataFrame(normals).to_csv(out / "normality_checks.csv", index=False)

    # IQR flags are diagnostic; retain rows by default because extreme operation may be real.
    outlier_rows = []
    for col in NUMERIC + [TARGET]:
        s = df[col].dropna()
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[col] < lo) | (df[col] > hi)
        outlier_rows.append({"feature": col, "lower_fence": lo, "upper_fence": hi,
                             "flagged_count": int(mask.sum()), "flagged_percent": 100 * mask.mean(),
                             "action": "flagged for review; retained"})
    pd.DataFrame(outlier_rows).to_csv(out / "outlier_report.csv", index=False)

    X, y = df[FEATURES], df[TARGET]
    # If a usable ordered timestamp exists, hold out the latest 20% to reduce temporal leakage.
    use_time = ("timestamp" in df and df["timestamp"].notna().all()
                and df["timestamp"].nunique() == len(df)
                and df["timestamp"].is_monotonic_increasing)
    if use_time:
        order = np.arange(len(df))
        cut = max(1, int(len(order) * .8))
        train_idx, test_idx = order[:cut], order[cut:]
        split = "chronological 80/20"
    else:
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=.2, random_state=42)
        split = "random 80/20, seed 42 (no valid unique timestamp)"
    Xtr, Xte, ytr, yte = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

    models = {
        "Mean baseline": Pipeline([("prep", build_preprocessor(False)), ("model", DummyRegressor(strategy="mean"))]),
        "Linear regression": Pipeline([("prep", build_preprocessor(True)), ("model", LinearRegression())]),
        "Random forest": Pipeline([("prep", build_preprocessor(False)),
                                    ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                                    random_state=42, n_jobs=-1))]),
        "KNN regression": Pipeline([("prep", build_preprocessor(True)),
                                    ("model", KNeighborsRegressor(n_neighbors=7, weights="distance"))]),
    }
    rows, predictions = [], {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        predictions[name] = pred
        rows.append({"model": name, "MAE_km": mean_absolute_error(yte, pred),
                     "RMSE_km": mean_squared_error(yte, pred) ** .5,
                     "MSE_km2": mean_squared_error(yte, pred),
                     "R2": r2_score(yte, pred), "split": split,
                     "train_n": len(train_idx), "test_n": len(test_idx)})
    scores = pd.DataFrame(rows).sort_values("RMSE_km")
    scores.to_csv(out / "model_metrics.csv", index=False)
    best_name = scores.iloc[0]["model"]
    # Save the complete best-performing ML pipeline
    best_model = models[best_name]
    model_path = out / "best_model_pipeline.joblib"
    joblib.dump(best_model, model_path)
    print(f"Best model saved successfully: {model_path}")
    print(f"Selected model: {best_name}")

    pd.DataFrame({"actual_km": yte.to_numpy(), "predicted_km": predictions[best_name]}).to_csv(out / "best_model_predictions.csv", index=False)

    # Feature importance only for random forest; names retain encoded categorical levels.
    rf = models["Random forest"]
    names = rf.named_steps["prep"].get_feature_names_out()
    fi = pd.DataFrame({"feature": names, "importance": rf.named_steps["model"].feature_importances_}).sort_values("importance", ascending=False)
    fi.to_csv(out / "random_forest_feature_importance.csv", index=False)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[TARGET], kde=True, ax=ax, color="#147D92")
    ax.set(title="Remaining range distribution", xlabel="Remaining range (km)")
    fig.tight_layout(); fig.savefig(figures / "range_distribution.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    top = fi.head(12).sort_values("importance")
    ax.barh(top["feature"], top["importance"], color="#2589A6")
    ax.set(title="Random forest: top feature importances", xlabel="Impurity-based importance", ylabel="")
    fig.tight_layout(); fig.savefig(figures / "feature_importance.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(yte, predictions[best_name], alpha=.7, color="#D56B35")
    lims = [min(yte.min(), predictions[best_name].min()), max(yte.max(), predictions[best_name].max())]
    ax.plot(lims, lims, "--", color="#333333")
    ax.set(title=f"Actual vs predicted: {best_name}", xlabel="Actual range (km)", ylabel="Predicted range (km)")
    fig.tight_layout(); fig.savefig(figures / "actual_vs_predicted.png", dpi=180); plt.close(fig)

    # Numeric-only pairwise correlation heatmap.
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[NUMERIC + [TARGET]].corr()
    sns.heatmap(corr, cmap="vlag", center=0, ax=ax, square=True)
    ax.set_title("Numeric feature correlation")
    fig.tight_layout(); fig.savefig(figures / "correlation_heatmap.png", dpi=180); plt.close(fig)

    invalid_ranges = {
        "battery_soc_outside_0_100": int(((df.battery_soc < 0) | (df.battery_soc > 100)).sum()),
        "battery_health_outside_0_100": int(((df.battery_health < 0) | (df.battery_health > 100)).sum()),
        "humidity_outside_0_100": int(((df.humidity < 0) | (df.humidity > 100)).sum()),
        "negative_target": int((df[TARGET] < 0).sum()),
        "negative_distance_traveled": int((df.distance_traveled < 0).sum()),
        "negative_speed": int((df.speed < 0).sum()),
    }
    pd.DataFrame([{"check": k, "count": v, "action": "review against source/domain; do not auto-remove"}
                  for k, v in invalid_ranges.items()]).to_csv(out / "domain_range_checks.csv", index=False)
    summary = {"source": str(source), "rows_loaded": n_loaded, "rows_used": len(df),
               "target_missing_rows_dropped": target_missing, "predictor_missing_cells": int(df[FEATURES].isna().sum().sum()),
               "split": split, "best_by_test_RMSE": best_name,
               "full_project_dataset_confirmed": n_loaded >= 1000,
               "domain_range_checks": invalid_ranges,
               "note": "Treat scores as final only if source is the full project CSV."}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nModel comparison (lower MAE/RMSE is better; higher R2 is better):")
    print(scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
