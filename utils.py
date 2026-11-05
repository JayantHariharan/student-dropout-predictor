"""Shared helpers: loading saved models, validating uploaded CSVs, computing metrics.
Kept free of Streamlit so it can be unit-tested on its own."""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
METRIC_COLS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def normalise_target(series: pd.Series) -> pd.Series:
    """Turn the target into clean strings so '1', 1 and 1.0 all mean the same class."""
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        s = s.where(s.isna() | (s % 1 != 0), s.astype("float").round(0))
        s = s.map(lambda v: v if pd.isna(v) else (str(int(v)) if float(v).is_integer() else str(v)))
    else:
        s = s.map(lambda v: v if pd.isna(v) else str(v).strip())
    return s


def load_artifacts(model_dir="model"):
    meta_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"'{meta_path}' not found. Run train_models.py first and commit the model/ folder.")
    with open(meta_path) as f:
        meta = json.load(f)
    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(model_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing saved model file: {path}")
        models[name] = joblib.load(path)
    return models, meta


def prepare_data(df: pd.DataFrame, meta: dict):
    """Validate an uploaded DataFrame. Returns (X, y, warnings). Raises ValueError with a
    user-friendly message for anything that cannot be handled."""
    warnings = []
    if df is None or df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("The uploaded file is empty.")
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    target, feats = meta["target_column"], meta["feature_columns"]
    missing = [c for c in feats + [target] if c not in df.columns]
    if missing:
        raise ValueError("The CSV is missing required column(s): " + ", ".join(missing)
                         + ". It must contain all feature columns and the target column.")

    # drop rows with missing target
    n0 = len(df)
    df = df[df[target].notna()]
    if len(df) < n0:
        warnings.append(f"Dropped {n0 - len(df)} row(s) with an empty target value.")
    if df.empty:
        raise ValueError("No rows with a valid target value were found.")

    y = normalise_target(df[target])
    unknown = sorted(set(y) - set(meta["classes"]))
    if unknown:
        raise ValueError(f"Target column has label(s) the models were never trained on: {unknown}. "
                         f"Expected only: {meta['classes']}.")

    X = df[feats].copy()
    for c in meta["numeric_features"]:
        converted = pd.to_numeric(X[c], errors="coerce")
        bad = int((converted.isna() & X[c].notna()).sum())
        if bad:
            warnings.append(f"Column '{c}': {bad} non-numeric value(s) treated as missing.")
        X[c] = converted
    for c in meta["categorical_features"]:
        X[c] = X[c].map(lambda v: v if pd.isna(v) else str(v).strip())
    X = X.replace([np.inf, -np.inf], np.nan)
    extra = [c for c in df.columns if c not in feats + [target]]
    if extra:
        warnings.append("Ignored extra column(s): " + ", ".join(extra[:10]))
    if y.nunique() < 2:
        warnings.append("Only one class present in the uploaded data - AUC and MCC cannot be computed reliably.")
    return X.reset_index(drop=True), y.reset_index(drop=True), warnings


def compute_metrics(y_true, y_pred, y_proba, classes):
    """Return the six assignment metrics. Binary -> 'binary' averaging with the 2nd class
    as positive; multi-class -> weighted averaging and one-vs-rest AUC."""
    binary = len(classes) == 2
    avg = "binary" if binary else "weighted"
    kw = dict(average=avg, zero_division=0)
    if binary:
        kw["pos_label"] = classes[1]
    try:
        if binary:
            auc = roc_auc_score(y_true == classes[1], y_proba[:, 1])
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted", labels=classes)
    except ValueError:
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, **kw),
        "Recall": recall_score(y_true, y_pred, **kw),
        "F1": f1_score(y_true, y_pred, **kw),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def evaluate(pipeline, X, y, classes):
    """Predict with an ALREADY TRAINED pipeline (no fitting) and compute everything."""
    y_pred = pipeline.predict(X)
    proba = pipeline.predict_proba(X)
    # make sure probability columns follow the order of `classes`
    order = [list(pipeline.classes_).index(c) for c in classes]
    proba = proba[:, order]
    metrics = compute_metrics(y.values, y_pred, proba, classes)
    cm = confusion_matrix(y, y_pred, labels=classes)
    report = classification_report(y, y_pred, labels=classes, zero_division=0, output_dict=True)
    return metrics, cm, pd.DataFrame(report).T, y_pred
