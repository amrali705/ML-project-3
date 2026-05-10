import io
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score

from milestone_1_linear_regression import test_script_m1 as m1
from milestone_2_classification import test_script2 as m2


warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent

TASKS = {
    "Regression": {
        "description": "Predict continuous movie popularity scores.",
        "default_csv": m1.BASE_DIR / "test_data_m1.csv",
        "target": "popularity",
        "prediction_col": "predicted_popularity",
    },
    "Classification": {
        "description": "Classify movies into popularity levels.",
        "default_csv": m2.BASE_DIR / "test_data_m2.csv",
        "target": "popularityLevel",
        "prediction_col": "predicted_label",
    },
}


st.set_page_config(
    page_title="Movie Popularity Prediction Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --navy: #0f172a;
            --muted: #64748b;
            --blue: #2563eb;
            --blue-soft: #dbeafe;
            --panel: #ffffff;
            --line: #e2e8f0;
            --good: #16a34a;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }

        h1, h2, h3 {
            color: var(--navy);
            letter-spacing: 0;
        }

        [data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid var(--line);
        }

        .hero {
            padding: 1.4rem 1.6rem;
            border: 1px solid var(--line);
            border-radius: 14px;
            background:
                linear-gradient(135deg, rgba(37, 99, 235, 0.08), rgba(14, 165, 233, 0.02)),
                #ffffff;
            margin-bottom: 1.2rem;
        }

        .hero-title {
            font-size: 2.15rem;
            line-height: 1.12;
            font-weight: 800;
            color: var(--navy);
            margin-bottom: 0.35rem;
        }

        .hero-subtitle {
            color: var(--muted);
            font-size: 1rem;
            max-width: 900px;
        }

        .metric-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            min-height: 112px;
            box-shadow: 0 14px 28px rgba(15, 23, 42, 0.04);
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .metric-value {
            color: var(--navy);
            font-size: 1.65rem;
            font-weight: 800;
            margin-top: 0.18rem;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 0.25rem;
        }

        .best-card {
            border-color: rgba(37, 99, 235, 0.35);
            background: linear-gradient(135deg, #eff6ff, #ffffff);
        }

        .section-panel {
            border: 1px solid var(--line);
            border-radius: 14px;
            background: #ffffff;
            padding: 1rem;
        }

        div[data-testid="stDownloadButton"] > button,
        div[data-testid="stButton"] > button {
            border-radius: 10px;
            border: 1px solid rgba(37, 99, 235, 0.25);
            font-weight: 700;
        }

        div[data-testid="stButton"] > button[kind="primary"] {
            background: var(--blue);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_models(model_paths: tuple[tuple[str, str], ...]) -> dict:
    return {name: load_pickle(path) for name, path in model_paths}


@st.cache_data(show_spinner=False)
def load_default_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def model_paths(models: dict) -> tuple[tuple[str, str], ...]:
    return tuple((name, str(path)) for name, path in models.items())


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2f}%"


def metric_card(label: str, value: str, note: str = "", best: bool = False) -> None:
    class_name = "metric-card best-card" if best else "metric-card"
    st.markdown(
        f"""
        <div class="{class_name}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_preview(
    df_raw: pd.DataFrame,
    predictions: np.ndarray,
    prediction_col: str,
    label_map: dict | None = None,
) -> pd.DataFrame:
    keep_cols = [
        col
        for col in ["id", "title", "popularity", "popularityLevel"]
        if col in df_raw.columns
    ]
    out = df_raw[keep_cols].copy() if keep_cols else pd.DataFrame(index=df_raw.index)
    out[prediction_col] = predictions
    if label_map is not None:
        pred_series = pd.Series(predictions, index=out.index)
        out["predicted_level"] = pred_series.map(label_map).fillna(pred_series)
    return out


def run_regression(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    bundle = load_pickle(str(m1.PREPROCESSING_FILE))
    models = load_models(model_paths(m1.MODELS))
    X_tree, X_linear, y = m1.preprocess(df_raw, bundle)

    rows = []
    predictions = {}
    has_labels = y is not None

    for name, model in models.items():
        X = X_linear if name in m1.LINEAR_MODELS else X_tree
        preds = m1.predict(model, X, log_target=True)
        predictions[name] = result_preview(
            df_raw, preds, TASKS["Regression"]["prediction_col"]
        )

        row = {"Model": name}
        if has_labels:
            mse = mean_squared_error(y.values, preds)
            rmse = float(np.sqrt(mse))
            r2 = r2_score(y.values, preds)
            row.update(
                {
                    "MSE": round(mse, 4),
                    "RMSE": round(rmse, 4),
                    "R2": round(r2, 4),
                    "Accuracy (R2%)": round(max(0, r2) * 100, 2),
                }
            )
        rows.append(row)

    results = pd.DataFrame(rows)
    if has_labels:
        results = results.sort_values("R2", ascending=False).reset_index(drop=True)
    return results, predictions


def run_classification(
    df_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    bundle = load_pickle(str(m2.PREPROCESSING_FILE))
    models = load_models(model_paths(m2.MODELS))
    X_tree, X_linear, y = m2.preprocess(df_raw, bundle)
    label_map = bundle.get("level_map_inv", {})

    rows = []
    predictions = {}
    reports = {}
    has_labels = y is not None and not y.isna().any()

    for name, model in models.items():
        X = X_linear if name in m2.LINEAR_MODELS else X_tree
        preds = m2.predict(model, X)
        predictions[name] = result_preview(
            df_raw, preds, TASKS["Classification"]["prediction_col"], label_map
        )

        row = {"Model": name}
        if has_labels:
            acc = accuracy_score(y, preds) * 100
            row["Accuracy"] = round(acc, 2)
            report = classification_report(
                y,
                preds,
                labels=[0, 1, 2, 3],
                target_names=["Very Low", "Low", "Medium", "High"],
                output_dict=True,
                zero_division=0,
            )
            reports[name] = pd.DataFrame(report).transpose().round(3)
        rows.append(row)

    results = pd.DataFrame(rows)
    if has_labels:
        results = results.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return results, predictions, reports


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Movie Popularity Prediction Dashboard</div>
            <div class="hero-subtitle">
                Compare trained regression and classification models, inspect prediction output,
                and download clean CSV results from one dashboard.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, str, pd.DataFrame | None]:
    st.sidebar.title("Controls")
    task = st.sidebar.radio(
        "Prediction task",
        ["Regression", "Classification"],
        captions=[TASKS["Regression"]["description"], TASKS["Classification"]["description"]],
    )
    source = st.sidebar.radio("Data source", ["Use default test data", "Upload CSV"])

    uploaded_df = None
    if source == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            uploaded_df = pd.read_csv(uploaded, low_memory=False)

    st.sidebar.divider()
    st.sidebar.caption("Default files")
    st.sidebar.code(str(TASKS[task]["default_csv"].relative_to(ROOT_DIR)))
    return task, source, uploaded_df


def render_data_cards(df: pd.DataFrame, task: str, source: str) -> None:
    target = TASKS[task]["target"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Rows", f"{len(df):,}", source)
    with c2:
        metric_card("Columns", f"{len(df.columns):,}", "Input feature table")
    with c3:
        metric_card("Target", "Present" if target in df.columns else "Missing", target)
    with c4:
        metric_card("Task", task, TASKS[task]["description"])


def render_best_card(task: str, results: pd.DataFrame) -> None:
    if results.empty:
        return

    if task == "Regression" and "R2" in results.columns:
        best = results.iloc[0]
        value = format_pct(best["Accuracy (R2%)"])
        note = f"Best model by R2: {best['Model']}"
    elif task == "Classification" and "Accuracy" in results.columns:
        best = results.iloc[0]
        value = format_pct(best["Accuracy"])
        note = f"Best model by accuracy: {best['Model']}"
    else:
        best = results.iloc[0]
        value = best["Model"]
        note = "Predictions generated. Add labels to compute metrics."

    metric_card("Best Result", value, note, best=True)


def render_downloads(predictions: dict[str, pd.DataFrame]) -> None:
    cols = st.columns(2)
    for idx, (name, pred_df) in enumerate(predictions.items()):
        with cols[idx % 2]:
            st.download_button(
                label=f"Download {name} predictions",
                data=csv_bytes(pred_df),
                file_name=f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )


def main() -> None:
    inject_css()
    render_header()
    task, source, uploaded_df = render_sidebar()

    if source == "Upload CSV":
        if uploaded_df is None:
            st.info("Upload a CSV from the sidebar to run predictions.")
            return
        df_raw = uploaded_df
    else:
        df_raw = load_default_csv(str(TASKS[task]["default_csv"]))

    run_clicked = st.sidebar.button("Run Predictions", type="primary", use_container_width=True)
    render_data_cards(df_raw, task, source)

    tabs = st.tabs(["Data Preview", "Model Results", "Predictions", "Downloads"])

    with tabs[0]:
        st.subheader("Input Data")
        st.dataframe(df_raw.head(100), use_container_width=True, height=420)

    if not run_clicked and "last_results" not in st.session_state:
        with tabs[1]:
            st.info("Select your task and click Run Predictions.")
        return

    if run_clicked:
        with st.spinner("Running models..."):
            if task == "Regression":
                results, predictions = run_regression(df_raw)
                reports = {}
            else:
                results, predictions, reports = run_classification(df_raw)
            st.session_state["last_results"] = {
                "task": task,
                "results": results,
                "predictions": predictions,
                "reports": reports,
                "target_present": TASKS[task]["target"] in df_raw.columns,
            }

    state = st.session_state["last_results"]
    if state["task"] != task:
        with tabs[1]:
            st.info("Click Run Predictions to refresh results for the selected task.")
        return

    results = state["results"]
    predictions = state["predictions"]
    reports = state["reports"]

    with tabs[1]:
        st.subheader("Model Results")
        if not state["target_present"]:
            st.warning("Target column is missing, so metrics are unavailable. Predictions are still shown.")
        render_best_card(task, results)
        st.write("")
        st.dataframe(results, use_container_width=True, hide_index=True)

        if reports:
            st.subheader("Classification Reports")
            selected_model = st.selectbox("Model report", list(reports.keys()))
            st.dataframe(reports[selected_model], use_container_width=True)

    with tabs[2]:
        st.subheader("Prediction Preview")
        selected_model = st.selectbox("Prediction model", list(predictions.keys()))
        st.dataframe(predictions[selected_model].head(200), use_container_width=True, height=460)

    with tabs[3]:
        st.subheader("Download Predictions")
        render_downloads(predictions)


if __name__ == "__main__":
    main()
