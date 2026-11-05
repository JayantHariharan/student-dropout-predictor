"""Train all 5 classifiers on ONE dataset, save them in model/, and write test_data.csv.

Usage:
    python train_models.py --data data/dataset.csv --target Target
"""
import argparse
import json
import os

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--out", default="model")
    ap.add_argument("--drop", nargs="*", default=[], help="ID-like columns to remove, e.g. --drop employee_id")
    ap.add_argument("--max_rows", type=int, default=0,
                    help="If >0, take a stratified sample of this size (keeps SVM/KNN fast and repo small)")
    a = ap.parse_args()

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


if __name__ == "__main__":
    main()
