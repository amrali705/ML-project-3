"""
test_script_m1.py  —  Milestone 1 Regression Test Script
=========================================================
Usage:
    python test_script_m1.py <path_to_test_csv>

What it does:
    1. Loads the test CSV
    2. Applies the exact same preprocessing as the training notebook
       (using only parameters learned from the training set — no leakage)
    3. Loads all 4 saved regression models
    4. Outputs predictions + MSE and R2 scores for each model
       (scores are shown if the test CSV contains a 'popularity' column;
        if not, only predictions are saved)

Outputs per model:
    predictions_lr_m1.csv
    predictions_poly2_m1.csv
    predictions_rf_m1.csv
    predictions_xgb_m1.csv
"""

import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 0. Paths & constants
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
PREPROCESSING_FILE = BASE_DIR / "preprocessing_m1.pkl"
MODELS = {
    "Linear Regression": BASE_DIR / "model_lr_m1.pkl",
    "Polynomial Regression (deg=2)": BASE_DIR / "model_poly2_m1.pkl",
    "Random Forest": BASE_DIR / "model_rf_m1.pkl",
    "XGBoost": BASE_DIR / "model_xgb_m1.pkl",
}
# Linear models use a curated feature subset; tree models use all features
LINEAR_MODELS = {"Linear Regression", "Polynomial Regression (deg=2)"}


# ─────────────────────────────────────────────
# 1. Load preprocessing artifacts
# ─────────────────────────────────────────────


def load_preprocessing(path: Path = PREPROCESSING_FILE) -> dict:
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"Loaded preprocessing bundle from: {path}")
    print(f"  Keys: {list(bundle.keys())}")
    return bundle


# ─────────────────────────────────────────────
# 2. Preprocess the test CSV
#    — mirrors every step in the training notebook
#    — uses ONLY parameters from the bundle (no fitting on test data)
# ─────────────────────────────────────────────


def preprocess(df_raw: pd.DataFrame, bundle: dict):
    """
    Apply training-equivalent preprocessing to a raw test DataFrame.
    Returns:
        X_tree   — full feature matrix for tree models
        X_linear — curated feature matrix for linear models
        y        — raw popularity series (or None if column absent)
    """
    df = df_raw.copy()

    # ── Extract target if present ──
    y = None
    if "popularity" in df.columns:
        y = df["popularity"].clip(upper=bundle["pop_cap"]).reset_index(drop=True)

    # ── Basic type fixes (same as notebook cell 4) ──
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    if "theatrical" in df.columns:
        df["theatrical"] = (
            pd.to_numeric(df["theatrical"], errors="coerce").fillna(0).astype(int)
        )
    if "adult" in df.columns:
        df["adult"] = (
            df["adult"]
            .map({"False": 0, "True": 1, False: 0, True: 1})
            .fillna(0)
            .astype(int)
        )

    df["revenue"] = df["revenue"].clip(lower=0) if "revenue" in df.columns else 0
    df["budget"] = df["budget"].clip(lower=0) if "budget" in df.columns else 0
    if "runtime" in df.columns:
        df["runtime"] = df["runtime"].clip(lower=0).replace(0, np.nan)

    df["release_date"] = df["release_date"].where(
        df["release_date"].between("1888-01-01", "2026-12-31"), other=pd.NaT
    )

    # sentiment
    sentiment_map = bundle["sentiment_map"]
    if "movie_sentiment" in df.columns:
        df["sentiment_encoded"] = df["movie_sentiment"].map(sentiment_map).fillna(0)
    else:
        df["sentiment_encoded"] = 0

    # fill text / list columns
    for c in ["title", "original_title"]:
        df[c] = df[c].fillna("Unknown") if c in df.columns else "Unknown"
    for c in [
        "overview",
        "tagline",
        "backdrop_path",
        "homepage",
        "imdb_id",
        "poster_path",
    ]:
        if c in df.columns:
            df[c] = df[c].fillna("")
    for c in [
        "genres",
        "production_companies",
        "production_countries",
        "spoken_languages",
    ]:
        if c in df.columns:
            df[c] = df[c].fillna("")

    # ── Feature engineering (cell 8) ──
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_quarter"] = df["release_date"].dt.quarter
    df["is_summer"] = df["release_month"].isin([6, 7, 8]).astype(int)
    df["is_holiday"] = df["release_month"].isin([11, 12]).astype(int)
    df["movie_age"] = (2025 - df["release_year"]).clip(lower=0)
    df["is_recent"] = (df["release_year"] >= 2015).astype(int)

    df["has_overview"] = (df["overview"].str.strip() != "").astype(int)
    df["has_tagline"] = (df["tagline"].str.strip() != "").astype(int)
    df["has_homepage"] = (df["homepage"].str.strip() != "").astype(int)
    df["has_poster"] = (df["poster_path"].str.strip() != "").astype(int)
    df["has_backdrop"] = (df["backdrop_path"].str.strip() != "").astype(int)
    df["has_imdb"] = (df["imdb_id"].str.strip() != "").astype(int)

    df["has_budget"] = (df["budget"] > 0).astype(int)
    df["has_revenue"] = (df["revenue"] > 0).astype(int)

    df["log_vote_count"] = (
        np.log1p(df["vote_count"]) if "vote_count" in df.columns else 0
    )
    df["log_budget"] = np.log1p(df["budget"])
    df["log_revenue"] = np.log1p(df["revenue"])

    df["profit"] = (df["revenue"] - df["budget"]).clip(lower=0)
    df["log_profit"] = np.log1p(df["profit"])
    df["has_profit"] = (df["profit"] > 0).astype(int)

    df["genre_count"] = df["genres"].apply(
        lambda x: len([g for g in str(x).split(",") if g.strip()]) if x else 0
    )
    df["company_count"] = df["production_companies"].apply(
        lambda x: len([c for c in str(x).split(",") if c.strip()]) if x else 0
    )
    df["country_count"] = df["production_countries"].apply(
        lambda x: len([c for c in str(x).split(",") if c.strip()]) if x else 0
    )

    df["is_us_production"] = (
        df["production_countries"]
        .str.contains("United States of America", na=False)
        .astype(int)
    )
    df["is_uk_production"] = (
        df["production_countries"].str.contains("United Kingdom", na=False).astype(int)
    )
    df["is_international"] = (df["country_count"] > 1).astype(int)

    df["spoken_lang_count"] = df["spoken_languages"].apply(
        lambda x: len([l for l in str(x).split(",") if l.strip()]) if x else 0
    )
    df["is_english_lang"] = (
        df["spoken_languages"].str.contains("English", na=False).astype(int)
    )
    df["is_multilingual"] = (df["spoken_lang_count"] > 1).astype(int)

    df["overview_len"] = df["overview"].str.split().str.len().fillna(0)

    # ── Encoding (cell 10) — use TRAIN parameters ──
    df["quality_encoded"] = (
        df["quality"].map(bundle["quality_map"]) if "quality" in df.columns else np.nan
    )
    df["status_encoded"] = (
        df["status"].map(bundle["status_map"]) if "status" in df.columns else np.nan
    )
    df["status_encoded"] = df["status_encoded"].fillna(bundle["fill_status"])

    # language grouping — use top_langs learned from train
    top_langs = bundle["top_langs"]
    lang_cols = bundle["lang_columns"]
    if "original_language" in df.columns:
        df["lang_group"] = df["original_language"].apply(
            lambda x: x if x in top_langs else "other"
        )
    else:
        df["lang_group"] = "other"
    lang_dummies = pd.get_dummies(df["lang_group"], prefix="lang", dtype=int)
    lang_dummies = lang_dummies.reindex(columns=lang_cols, fill_value=0)
    df = pd.concat([df, lang_dummies], axis=1)

    # genre one-hot — use genre list from train
    for genre in bundle["all_genres"]:
        col = "genre_" + genre.lower().replace(" ", "_").replace("-", "_").replace(
            "&", "and"
        )
        df[col] = (
            df["genres"].str.contains(genre, na=False).astype(int)
            if "genres" in df.columns
            else 0
        )

    # top-company flags — use company list from train
    for company in bundle["top_companies"]:
        col = (
            "co_"
            + company.lower().replace(" ", "_").replace("-", "_").replace(".", "")[:25]
        )
        df[col] = (
            df["production_companies"]
            .str.contains(company, na=False, regex=False)
            .astype(int)
            if "production_companies" in df.columns
            else 0
        )

    # ── Imputation (cell 12) — fill with 0 / train medians ──
    df["has_emotion_data"] = (
        df["movie_valence"].notna().astype(int) if "movie_valence" in df.columns else 0
    )
    for col in bundle["emotion_cols"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0  # column absent → treat as "no signal"

    medians = bundle["medians"]
    for col, median_val in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)
        else:
            df[col] = median_val  # column missing entirely → use train median

    # IQR capping — use TRAIN bounds
    for col, (lo, hi) in bundle["cap_params"].items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # ── Drop redundant columns (cell 14) ──
    cols_to_drop = [
        "quality",
        "status",
        "original_language",
        "lang_group",
        "budget",
        "revenue",
        "vote_count",
        "profit",
        "release_date",
        "genres",
        "production_companies",
        "production_countries",
        "spoken_languages",
        "title",
        "original_title",
        "overview",
        "tagline",
        "backdrop_path",
        "homepage",
        "imdb_id",
        "poster_path",
        "adult",
        "movie_sentiment",
        "id",
        "popularity",
        "popularityLevel",  # in case classification label is present
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # ── Feature selection (cell 16) — apply TRAIN VarianceThreshold mask ──
    keep_mask = bundle["keep_mask"]
    feature_cols = bundle["feature_cols"]  # final column order for tree models

    # Align columns to what the variance selector saw at train time
    # (keep_mask was built on X_train; we must reconstruct the same column space)
    # Use feature_cols as the reference — it is the post-mask column list
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0  # column missing in test → fill with 0 (safe default)

    X_tree = df[feature_cols].copy().fillna(0).astype(float)
    X_linear = X_tree[[c for c in bundle["linear_core"] if c in X_tree.columns]].copy()

    return X_tree, X_linear, y


# ─────────────────────────────────────────────
# 3. Load a model and predict
# ─────────────────────────────────────────────


def predict(model, X: pd.DataFrame, log_target: bool = True) -> np.ndarray:
    """Predict and invert the log transform if the model was trained on log1p(y)."""
    raw = model.predict(X)
    if log_target:
        return np.expm1(raw).clip(0)
    return raw.clip(0)


# ─────────────────────────────────────────────
# 4. Score helper
# ─────────────────────────────────────────────


def score(name: str, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    accuracy_pct = max(0, r2) * 100
    sep = "=" * 55
    print(sep)
    print(f"  {name}")
    print(sep)
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R2   : {r2:.4f}")
    print(f"  Accuracy (R2%) : {accuracy_pct:.2f}%")
    print()
    return {
        "Model": name,
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Accuracy (R2%)": round(accuracy_pct, 2),
    }


# ─────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────


def main():
    # Determine default CSV path relative to this script's directory
    default_path = BASE_DIR / "test_data_m1.csv"
    if len(sys.argv) < 2:
        csv_path = str(default_path)
        print(f"\nNo path supplied. Loading default test data: {csv_path}")
    else:
        csv_path = sys.argv[1]
        print(f"\nLoading test data: {csv_path}")
    df_raw = pd.read_csv(csv_path, low_memory=False)
    print(f"Shape: {df_raw.shape}\n")

    # Load preprocessing bundle
    bundle = load_preprocessing()

    # Preprocess
    print("\nPreprocessing...")
    X_tree, X_linear, y = preprocess(df_raw, bundle)
    print(f"  X_tree   shape: {X_tree.shape}")
    print(f"  X_linear shape: {X_linear.shape}")
    has_labels = y is not None
    if has_labels:
        print(f"  Target present - will compute MSE and R2\n")
    else:
        print(f"  No 'popularity' column - predictions only\n")

    scores = []

    for model_name, pkl_file in MODELS.items():
        print(f"Loading model: {model_name}  ({pkl_file.name})")
        with open(pkl_file, "rb") as f:
            model = pickle.load(f)

        X = X_linear if model_name in LINEAR_MODELS else X_tree
        preds = predict(model, X, log_target=True)

        # Save predictions to CSV
        out_name = pkl_file.stem.replace("model_", "predictions_") + ".csv"
        out_path = BASE_DIR / out_name
        pd.DataFrame({"predicted_popularity": preds}).to_csv(out_path, index=False)
        print(f"  Predictions saved -> {out_path}")

        if has_labels:
            row = score(model_name, y.values, preds)
            scores.append(row)

    # Summary table
    if scores:
        print("\n" + "=" * 55)
        print("  SUMMARY TABLE")
        print("=" * 55)
        summary = pd.DataFrame(scores).sort_values("R2", ascending=False)
        print(summary.to_string(index=False))
        scores_path = BASE_DIR / "scores_m1.csv"
        summary.to_csv(scores_path, index=False)
        print(f"\nScores table saved -> {scores_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
