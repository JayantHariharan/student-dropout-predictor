"""Train all 5 classifiers on ONE dataset, save them in model/, and write test_data.csv.
Also rewrites the auto-generated sections of README.md (dataset stats, metrics table,
observations) so those numbers never go stale after a retrain.

Usage:
    python train_models.py --data data/raw.csv --target Target
"""
import argparse
import json
import os
import re
import shutil

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils import MODEL_FILES, METRIC_COLS, compute_metrics, normalise_target

SEED = 42
GENERATED_DATA_FILES = ["data/train_data.csv", "data/test_data.csv", "test_data.csv", "metrics_table.csv"]


def reset_outputs(model_dir):
    """Wipe previously generated models/splits so a retrain never leaves stale files
    behind (e.g. a model file from a dataset/column set that no longer matches metadata.json).
    data/raw.csv is never touched - it is the permanent input, not an output."""
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    for f in GENERATED_DATA_FILES:
        if os.path.exists(f):
            os.remove(f)


def build_preprocessor(num_cols, cat_cols):
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=SEED),
    }


def _replace_between(text, start_marker, end_marker, new_body):
    pattern = re.compile(re.escape(start_marker) + r".*?" + re.escape(end_marker), re.DOTALL)
    replacement = f"{start_marker}\n{new_body}\n{end_marker}"
    if pattern.search(text) is None:
        return text, False
    return pattern.sub(replacement, text), True


def _dataset_section(df, target, feats, classes, source_name, source_url):
    counts = df[target].value_counts(normalize=True)
    balance = ", ".join(f"{c} ({counts.get(c, 0) * 100:.1f}%)" for c in classes)
    missing = int(df[feats].isna().sum().sum())
    return (
        f"- Source: {source_name} - [{source_url}]({source_url})\n"
        f"- Rows: {len(df)} | Features: {len(feats)} | Target: `{target}` | Classes: {balance}\n"
        f"- Missing values in features at training time: {missing}\n"
        f"- Preprocessing: median imputation + standardisation (numeric); most-frequent imputation + one-hot (categorical)\n"
        f"- Split: 80% train / 20% test, stratified, random_state={SEED} (files in `data/`)"
    )


def _metrics_table_section(res):
    header = "| ML Model Name | " + " | ".join(METRIC_COLS) + " |"
    sep = "|---" * (len(METRIC_COLS) + 1) + "|"
    lines = [header, sep]
    for name, row in res.iterrows():
        vals = " | ".join(f"{row[c]:.4f}" for c in METRIC_COLS)
        lines.append(f"| {name} | {vals} |")
    return "\n".join(lines)


def _observations_section(res):
    ranked = res["MCC"].sort_values(ascending=False)
    winner, loser = ranked.index[0], ranked.index[-1]
    lines = ["| ML Model Name | Observation about model performance |", "|---|---|"]
    for name in res.index:
        rank = list(ranked.index).index(name) + 1
        row = res.loc[name]
        if name == winner:
            text = (f"Best performer overall - rank 1 of {len(res)} by MCC ({row['MCC']:.4f}), "
                    f"also leads on Accuracy ({row['Accuracy']:.4f}) and F1 ({row['F1']:.4f}).")
        elif name == loser:
            text = f"Weakest of the {len(res)} models here - lowest MCC ({row['MCC']:.4f}) and F1 ({row['F1']:.4f})."
        else:
            text = f"Mid-table - rank {rank} of {len(res)} by MCC ({row['MCC']:.4f}), Accuracy {row['Accuracy']:.4f}."
        lines.append(f"| {name} | {text} |")
    lines.append(f"| **Overall winner** | **{winner}** - highest MCC ({ranked.iloc[0]:.4f}), "
                 f"which (unlike raw accuracy) is not inflated by class imbalance. |")
    return "\n".join(lines)


def update_readme(readme_path, df, target, feats, classes, res, source_name, source_url):
    if not os.path.exists(readme_path):
        print(f"Note: {readme_path} not found, skipping README auto-update.")
        return
    with open(readme_path, encoding="utf-8") as f:
        text = f.read()
    text, ok1 = _replace_between(text, "<!-- DATASET_INFO:START -->", "<!-- DATASET_INFO:END -->",
                                 _dataset_section(df, target, feats, classes, source_name, source_url))
    text, ok2 = _replace_between(text, "<!-- METRICS_TABLE:START -->", "<!-- METRICS_TABLE:END -->",
                                 _metrics_table_section(res))
    text, ok3 = _replace_between(text, "<!-- OBSERVATIONS:START -->", "<!-- OBSERVATIONS:END -->",
                                 _observations_section(res))
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"README.md auto-updated (dataset={ok1}, metrics={ok2}, observations={ok3}).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--out", default="model")
    ap.add_argument("--drop", nargs="*", default=[], help="ID-like columns to remove, e.g. --drop employee_id")
    ap.add_argument("--max_rows", type=int, default=0,
                    help="If >0, take a stratified sample of this size (keeps SVM/KNN fast and repo small)")
    ap.add_argument("--readme", default="README.md", help="README.md to auto-update with dataset/metrics sections")
    ap.add_argument("--source_name", default="UCI Machine Learning Repository - Predict Students' Dropout and Academic Success")
    ap.add_argument("--source_url", default="https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success")
    a = ap.parse_args()

    reset_outputs(a.out)

    df = pd.read_csv(a.data)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop(columns=[c for c in a.drop if c in df.columns])
    df = df.dropna(subset=[a.target]).drop_duplicates().reset_index(drop=True)
    df[a.target] = normalise_target(df[a.target])
    if a.max_rows and len(df) > a.max_rows:
        df, _ = train_test_split(df, train_size=a.max_rows, stratify=df[a.target], random_state=SEED)
        df = df.reset_index(drop=True)
    feats = [c for c in df.columns if c != a.target]
    print(f"Rows (after cleaning/sampling): {len(df)} | Features: {len(feats)} | Classes: {sorted(df[a.target].unique())}")
    assert len(df) >= 500 and len(feats) >= 12, "Assignment needs >=500 rows and >=12 features"

    num_cols = df[feats].select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in feats if c not in num_cols]
    classes = sorted(df[a.target].unique())

    train, test = train_test_split(df, test_size=a.test_size, stratify=df[a.target], random_state=SEED)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    train.to_csv("data/train_data.csv", index=False)
    test.to_csv("data/test_data.csv", index=False)
    test.to_csv("test_data.csv", index=False)  # copy required at repo root; upload this to the app

    rows = {}
    for name, clf in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor(num_cols, cat_cols)), ("clf", clf)])
        pipe.fit(train[feats], train[a.target])
        joblib.dump(pipe, os.path.join(a.out, MODEL_FILES[name]))
        pred = pipe.predict(test[feats])
        proba = pipe.predict_proba(test[feats])[:, [list(pipe.classes_).index(c) for c in classes]]
        rows[name] = compute_metrics(test[a.target].values, pred, proba, classes)
        print(f"{name:25s}", {k: round(v, 4) for k, v in rows[name].items()})

    meta = {"target_column": a.target, "feature_columns": feats, "numeric_features": num_cols,
            "categorical_features": cat_cols, "classes": classes}
    with open(os.path.join(a.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    res = pd.DataFrame(rows).T[METRIC_COLS].round(4)
    res.index.name = "ML Model Name"
    res.to_csv("metrics_table.csv")
    print("\n" + res.to_string())

    update_readme(a.readme, df, a.target, feats, classes, res, a.source_name, a.source_url)


if __name__ == "__main__":
    main()
