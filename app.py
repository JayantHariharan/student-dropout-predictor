import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils import METRIC_COLS, evaluate, load_artifacts, prepare_data

st.set_page_config(page_title="ML Classifier Evaluator", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="Loading saved models...")
def get_artifacts():
    return load_artifacts("model")


def plot_cm(cm, classes, title):
    fig, ax = plt.subplots(figsize=(4.5 + 0.3 * len(classes), 4 + 0.3 * len(classes)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    thr = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black")
    fig.tight_layout()
    return fig


# ---------- load models ----------
try:
    models, meta = get_artifacts()
except Exception as e:
    st.error(f"Could not load the saved models: {e}")
    st.stop()

classes = meta["classes"]
st.title("📊 Classification Model Evaluator")
st.caption("Upload a TEST csv. The pre-trained models are used as-is - nothing is retrained.")

# ---------- sidebar ----------
with st.sidebar:
    st.header("1. Data")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    use_sample = st.checkbox("Use the bundled test_data.csv instead", value=False,
                             disabled=not os.path.exists("test_data.csv"))
    st.header("2. Model")
    choice = st.selectbox("Select a model", list(models.keys()))
    with st.expander("Expected CSV format"):
        st.write(f"**Target column:** `{meta['target_column']}`")
        st.write(f"**Feature columns ({len(meta['feature_columns'])}):**")
        st.code(", ".join(meta["feature_columns"]))
    if os.path.exists("test_data.csv"):
        with open("test_data.csv", "rb") as f:
            st.download_button("Download sample test_data.csv", f.read(), "test_data.csv", "text/csv")

# ---------- read the file ----------
df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty.")
    except Exception as e:
        st.error(f"Could not read the file as CSV: {e}")
elif use_sample:
    df = pd.read_csv("test_data.csv")

if df is None:
    st.info("👈 Upload a test CSV (or tick the sample option) to begin.")
    st.stop()

try:
    X, y, warns = prepare_data(df, meta)
except ValueError as e:
    st.error(str(e))
    st.stop()
for w in warns:
    st.warning(w)

st.success(f"Loaded {len(X)} rows and {len(meta['feature_columns'])} features.")
with st.expander("Preview uploaded data"):
    st.dataframe(df.head(20), width="stretch")

# ---------- evaluate all models (inference only) ----------
results, cms, reports, preds = {}, {}, {}, {}
for name, pipe in models.items():
    try:
        results[name], cms[name], reports[name], preds[name] = evaluate(pipe, X, y, classes)
    except Exception as e:
        st.error(f"{name} failed on this data: {e}")

if not results:
    st.stop()
if choice not in results:
    st.stop()

tab1, tab2, tab3 = st.tabs([f"Selected: {choice}", "Compare all models", "Predictions"])

with tab1:
    m = results[choice]
    cols = st.columns(6)
    for c, k in zip(cols, METRIC_COLS):
        c.metric(k, "n/a" if np.isnan(m[k]) else f"{m[k]:.4f}")
    if np.isnan(m["AUC"]):
        st.caption("AUC is n/a because the uploaded data does not contain enough classes.")
    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        st.pyplot(plot_cm(cms[choice], classes, choice))
    with right:
        st.subheader("Classification report")
        st.dataframe(reports[choice].round(3), width="stretch")

with tab2:
    table = pd.DataFrame(results).T[METRIC_COLS]
    table.index.name = "ML Model Name"
    st.dataframe(table.style.format("{:.4f}", na_rep="n/a").highlight_max(axis=0, color="#c8e6c9"),
                 width="stretch")
    st.bar_chart(table[["Accuracy", "F1", "MCC"]])
    st.download_button("Download comparison table (CSV)", table.round(4).to_csv().encode(),
                       "model_comparison.csv", "text/csv")

with tab3:
    out = X.copy()
    out["Actual"] = y.values
    out[f"Predicted ({choice})"] = preds[choice]
    st.dataframe(out.head(200), width="stretch")
    st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(),
                       "predictions.csv", "text/csv")
