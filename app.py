import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import streamlit as st

from utils import METRIC_COLS, evaluate, load_artifacts, prepare_data

st.set_page_config(page_title="Student Dropout Risk - Model Dashboard", page_icon="🎓", layout="wide")

APP_CSS = """
<style>
.stApp {background: radial-gradient(1200px 600px at 8% -10%, rgba(139,124,255,.28), transparent 60%),
                    radial-gradient(900px 500px at 100% 0%, rgba(0,200,255,.18), transparent 55%), #0B1020;}
[data-testid="stSidebar"] {background: rgba(255,255,255,.05); backdrop-filter: blur(16px);
                           border-right: 1px solid rgba(255,255,255,.10);}
[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2.2rem; max-width: 1400px;}
[data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12);
    border-radius: 16px; backdrop-filter: blur(12px); box-shadow: 0 8px 28px rgba(0,0,0,.25);}
[data-testid="stMetric"] {padding: .8rem 1rem;}
[data-testid="stMetricValue"] {font-size: clamp(1.25rem, 2.1vw, 2rem);}
[data-testid="stMetricLabel"] p {opacity: .8; letter-spacing: .04em; text-transform: uppercase; font-size: .75rem;}
.hero h1 {font-size: clamp(1.5rem, 3.2vw, 2.4rem); margin: 0; line-height: 1.2;}
.hero p {opacity: .75; margin: .35rem 0 1rem;}
.chip {display: inline-block; padding: .15rem .65rem; margin: 0 .35rem .35rem 0; border-radius: 999px;
       background: rgba(139,124,255,.18); border: 1px solid rgba(139,124,255,.4); font-size: .8rem;}
@media (max-width: 640px) {
  .st-key-metrics [data-testid="stHorizontalBlock"] {flex-wrap: wrap; gap: .6rem;}
  .st-key-metrics [data-testid="stColumn"], .st-key-metrics [data-testid="column"] {
      flex: 1 1 calc(50% - .6rem) !important; min-width: calc(50% - .6rem) !important;}
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

INK = "#E8ECF8"
CM_CMAP = LinearSegmentedColormap.from_list("glass", ["#1B2243", "#4B3FC0", "#7C6BFF"])
BAR_COLORS = {"Accuracy": "#6EA8FE", "F1": "#8B7CFF", "MCC": "#FF8FAB"}
SHORT = {"Support Vector Machine": "Support Vector\nMachine", "Logistic Regression": "Logistic\nRegression",
         "Decision Tree": "Decision\nTree", "Naive Bayes": "Naive\nBayes"}


@st.cache_resource(show_spinner="Loading saved models...")
def get_artifacts():
    return load_artifacts("model")


def plot_cm(cm, classes, title):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.imshow(cm, cmap=CM_CMAP)
    ax.set_xticks(range(len(classes)), classes, color=INK)
    ax.set_yticks(range(len(classes)), classes, color=INK)
    ax.set_xlabel("Predicted", color=INK)
    ax.set_ylabel("Actual", color=INK)
    ax.set_title(title, color=INK, fontsize=12, pad=10)
    thr = cm.max() / 2 if cm.max() else 0
    row_tot = cm.sum(axis=1, keepdims=True)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = 100 * cm[i, j] / row_tot[i, 0] if row_tot[i, 0] else 0
            ax.text(j, i, f"{int(cm[i, j])}\n{pct:.0f}%", ha="center", va="center", fontsize=10,
                    color="white")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


def plot_comparison(table):
    """Grouped bars: one group per model, one bar per metric (all on the same 0-1 scale)."""
    metrics = ["Accuracy", "F1", "MCC"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    x = np.arange(len(table))
    w = 0.26
    for k, m in enumerate(metrics):
        bars = ax.bar(x + (k - 1) * w, table[m].values, w, label=m, color=BAR_COLORS[m])
        for b, v in zip(bars, table[m].values):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center",
                        fontsize=9, color=INK)
    ax.set_xticks(x, [SHORT.get(n, n) for n in table.index], color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (0 - 1, higher is better)", color=INK)
    ax.tick_params(colors=INK)
    ax.grid(axis="y", color="white", alpha=.12)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#ffffff40")
    ax.legend(frameon=False, labelcolor=INK, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    return fig


def best_per_column(s):
    """Highlight the best value of each metric column (readable on the dark theme)."""
    best = s.max()
    return ["background-color: rgba(46,204,113,.35); font-weight: 700" if v == best else "" for v in s]


# ---------- load models ----------
try:
    models, meta = get_artifacts()
except Exception as e:
    st.error(f"Could not load the saved models: {e}")
    st.stop()

classes = meta["classes"]
st.markdown(
    """<div class="hero">
    <h1>🎓 Student Dropout &amp; Academic Success - Model Dashboard</h1>
    <p>Compare five pre-trained classifiers on student records and see how well each one predicts
    whether a student will <b>Graduate</b>, <b>Drop out</b> or stay <b>Enrolled</b>.
    Models are loaded as-is - nothing is retrained on your upload.</p>
    <span class="chip">UCI dataset - 4,424 students</span><span class="chip">36 features</span>
    <span class="chip">3 classes</span><span class="chip">5 models</span></div>""",
    unsafe_allow_html=True)

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
    with st.container(key="metrics"):
        cols = st.columns(6)
        for c, k in zip(cols, METRIC_COLS):
            c.metric(k, "n/a" if np.isnan(m[k]) else f"{m[k]:.4f}")
    if np.isnan(m["AUC"]):
        st.caption("AUC is n/a because the uploaded data does not contain enough classes.")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("Confusion matrix")
            st.caption("Rows = actual class, columns = predicted class. Cell shows count and % of that actual class.")
            st.pyplot(plot_cm(cms[choice], classes, choice))
    with right:
        with st.container(border=True):
            st.subheader("Classification report")
            st.caption("Per-class precision, recall and F1, plus overall averages.")
            st.dataframe(reports[choice].round(3), width="stretch")

with tab2:
    table = pd.DataFrame(results).T[METRIC_COLS]
    table.index.name = "ML Model Name"
    with st.container(border=True):
        st.subheader("All models on the same test data")
        st.caption("Green = best score in that column.")
        st.dataframe(table.style.format("{:.4f}", na_rep="n/a").apply(best_per_column, axis=0),
                     width="stretch")
    with st.container(border=True):
        st.subheader("Accuracy vs F1 vs MCC")
        st.caption("Each group is one model; taller bars are better. MCC is the strictest of the three.")
        st.pyplot(plot_comparison(table))
    st.download_button("Download comparison table (CSV)", table.round(4).to_csv().encode(),
                       "model_comparison.csv", "text/csv")

with tab3:
    pred_col = f"Predicted ({choice})"
    out = pd.DataFrame({"Actual": y.values, pred_col: preds[choice]})
    out["Correct?"] = np.where(out["Actual"] == out[pred_col], "Yes", "No")
    out = pd.concat([out, X], axis=1)
    acc = (out["Correct?"] == "Yes").mean()
    with st.container(border=True):
        st.subheader("Row-level predictions")
        st.caption(f"{choice}: {int((out['Correct?'] == 'Yes').sum())} of {len(out)} rows correct ({acc:.1%}). "
                   "Showing the first 200 rows; Actual and Predicted come first.")
        st.dataframe(out.head(200), width="stretch")
    st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(),
                       "predictions.csv", "text/csv")
