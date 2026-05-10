"""
test_script2.py  —  Milestone 2 Classification Test Script
=========================================================
Usage:
    python test_script2.py <path_to_test_csv>

What it does:
    1. Loads the test CSV (default: test_data_m2.csv in the script's directory)
    2. Applies the exact same preprocessing as the training notebook
       (using only parameters learned from the training set — no leakage)
    3. Loads all 4 saved classification models
    4. Outputs predictions + accuracy and classification report for each model
       (scores are shown if the test CSV contains a 'popularityLevel' column;
        if not, only predictions are saved)
"""

import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

# ── Paths & constants ──
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
PREPROCESSING_FILE = BASE_DIR / "preprocessing.pkl"
MODELS = {
    "Logistic Regression": BASE_DIR / "model_lr.pkl",
    "Random Forest": BASE_DIR / "model_rf.pkl",
    "XGBoost": BASE_DIR / "model_xgb.pkl",
    "Decision Tree": BASE_DIR / "model_dt.pkl",
}
LINEAR_MODELS = {"Logistic Regression"}


def load_preprocessing(path: Path = PREPROCESSING_FILE) -> dict:
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"Loaded preprocessing bundle from: {path}")
    return bundle


def preprocess(df_raw: pd.DataFrame, bundle: dict):
    """Apply training‑equivalent preprocessing to a raw test DataFrame.
    Returns:
        X_tree   — full feature matrix for tree models
        X_linear — curated feature matrix for linear models
        y        — encoded target series (or None if column absent)
    """
    df = df_raw.copy()

    # ── Extract target if present ──
    y = None
    if "popularityLevel" in df.columns:
        level_map = bundle.get("level_map", {})
        y = df["popularityLevel"].map(level_map).reset_index(drop=True)

    # Provide safe defaults for columns that may be absent in lightweight test CSVs.
    required_defaults = {
        "release_date": pd.NaT,
        "theatrical": 0,
        "adult": 0,
        "revenue": 0.0,
        "budget": 0.0,
        "runtime": np.nan,
        "movie_sentiment": "",
        "title": "Unknown",
        "original_title": "Unknown",
        "overview": "",
        "tagline": "",
        "backdrop_path": "",
        "homepage": "",
        "imdb_id": "",
        "poster_path": "",
        "genres": "",
        "production_companies": "",
        "production_countries": "",
        "spoken_languages": "",
        "vote_count": 0.0,
        "quality": "Unknown",
        "status": "Unknown",
        "original_language": "other",
        "movie_valence": 0.0,
    }
    for col, default_val in required_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    # ── Basic type fixes ──
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    if "theatrical" in df.columns:
        df["theatrical"] = pd.to_numeric(df["theatrical"], errors="coerce").fillna(0).astype(int)
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

    # Sentiment map (may be missing in classification bundle)
    sentiment_map = bundle.get("sentiment_map", {})
    if "movie_sentiment" in df.columns:
        df["sentiment_encoded"] = df["movie_sentiment"].map(sentiment_map).fillna(0)
    else:
        df["sentiment_encoded"] = 0

    # Fill text / list columns
    text_fill = {"title": "Unknown", "original_title": "Unknown"}
    for c, default in text_fill.items():
        df[c] = df[c].fillna(default) if c in df.columns else default
    list_cols = [
        "overview",
        "tagline",
        "backdrop_path",
        "homepage",
        "imdb_id",
        "poster_path",
    ]
    for c in list_cols:
        if c in df.columns:
            df[c] = df[c].fillna("")
    for c in ["genres", "production_companies", "production_countries", "spoken_languages"]:
        if c in df.columns:
            df[c] = df[c].fillna("")

    # ── Feature engineering ──
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
    df["log_vote_count"] = np.log1p(df["vote_count"]) if "vote_count" in df.columns else 0
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
        df["production_countries"].str.contains("United States of America", na=False).astype(int)
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

    # ── Encoding ──
    quality_map = bundle.get("quality_map", {})
    status_map = bundle.get("status_map", {})
    for d in [df]:
        d["quality_encoded"] = d["quality"].map(quality_map)
        d["status_encoded"] = d["status"].map(status_map)
        d["status_encoded"] = d["status_encoded"].fillna(d["status_encoded"].median())

    # Language grouping (train‑only stats)
    top_langs = bundle.get("top_langs", [])
    df["lang_group"] = df["original_language"].apply(
        lambda x: x if x in top_langs else "other"
    )
    # Language dummies using saved column order
    lang_cols = bundle.get("lang_columns", [])
    lang_dummies = pd.get_dummies(df["lang_group"], prefix="lang", dtype=int)
    lang_dummies = lang_dummies.reindex(columns=lang_cols, fill_value=0)
    df = pd.concat([df, lang_dummies], axis=1)

    # Genre one‑hot (train list)
    all_genres = bundle.get("all_genres", [])
    for genre in all_genres:
        col = "genre_" + genre.lower().replace(" ", "_").replace("-", "_").replace("&", "and")
        df[col] = df["genres"].str.contains(genre, na=False).astype(int)

    # Top production companies (train list)
    top_companies = bundle.get("top_companies", [])
    for company in top_companies:
        col = (
            "co_"
            + company.lower().replace(" ", "_").replace("-", "_").replace(".", "")[:25]
        )
        df[col] = df["production_companies"].str.contains(company, na=False, regex=False).astype(int)

    # ── Imputation & capping ──
    emotion_cols = [
        "movie_intensity_anger",
        "movie_intensity_anticipation",
        "movie_intensity_disgust",
        "movie_intensity_fear",
        "movie_intensity_joy",
        "movie_intensity_sadness",
        "movie_intensity_surprise",
        "movie_intensity_trust",
        "movie_valence",
        "movie_vad_valence",
        "movie_vad_arousal",
        "movie_vad_dominance",
        "movie_scl_shift",
        "movie_scl_coverage",
    ]
    df["has_emotion_data"] = df["movie_valence"].notna().astype(int)
    for col in emotion_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Median imputation (numeric columns)
    medians = bundle.get("medians", {})
    for col, median_val in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)
        else:
            df[col] = median_val

    # IQR capping
    cap_params = bundle.get("cap_params", {})
    for col, (lo, hi) in cap_params.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # ── Drop redundant columns ──
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
        "popularity",  # not present in classification data
        "popularityLevel",
    ]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    # ── Feature selection ──
    keep_mask = bundle.get("keep_mask")
    feature_cols = bundle.get("feature_cols", [])
    if keep_mask is not None and feature_cols:
        X_v = df.reindex(columns=feature_cols, fill_value=0)
    else:
        X_v = df.copy()
    linear_core = bundle.get("linear_core", [])
    X_tree = X_v.copy().fillna(0).astype(float)
    if linear_core:
        X_linear = X_v.reindex(columns=linear_core, fill_value=0).copy()
    else:
        X_linear = X_v.copy()
    X_linear = X_linear.fillna(0).astype(float)

    return X_tree, X_linear, y


def predict(model, X: pd.DataFrame):
    return model.predict(X)


def main():
    default_path = BASE_DIR / "test_data_m2.csv"
    if len(sys.argv) < 2:
        csv_path = str(default_path)
        print(f"\nNo path supplied. Loading default test data: {csv_path}")
    else:
        csv_path = sys.argv[1]
        print(f"\nLoading test data: {csv_path}")
    df_raw = pd.read_csv(csv_path, low_memory=False)
    print(f"Shape: {df_raw.shape}\n")

    bundle = load_preprocessing()
    X_tree, X_linear, y = preprocess(df_raw, bundle)
    print(f"  X_tree shape: {X_tree.shape}\n    X_linear shape: {X_linear.shape}\n")

    scores = []
    for name, pkl_file in MODELS.items():
        print(f"Loading model: {name}  ({pkl_file.name})")
        with open(pkl_file, "rb") as f:
            model = pickle.load(f)
        X = X_linear if name in LINEAR_MODELS else X_tree
        preds = predict(model, X)
        out_name = pkl_file.stem.replace("model_", "predictions_") + ".csv"
        out_path = BASE_DIR / out_name
        pd.DataFrame({"predicted_label": preds}).to_csv(out_path, index=False)
        print(f"  Predictions saved -> {out_path}")
        if y is not None:
            acc = accuracy_score(y, preds)
            print(f"  Accuracy: {acc*100:.2f}%")
            print(classification_report(y, preds, target_names=["Very Low","Low","Medium","High"]))
            scores.append({"Model": name, "Accuracy": round(acc*100, 2)})
    if scores:
        summary = pd.DataFrame(scores).sort_values("Accuracy", ascending=False)
        print("\nValidation Summary")
        print(summary.to_string(index=False))
        scores_path = BASE_DIR / "scores_m2.csv"
        summary.to_csv(scores_path, index=False)
        print(f"\nScores table saved -> {scores_path}")
    print("\nDone.")

if __name__ == "__main__":
    main()
