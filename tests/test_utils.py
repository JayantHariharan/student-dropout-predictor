"""Run:  python -m pytest -q   (or  python tests/test_utils.py)"""
import io, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_artifacts, prepare_data, evaluate

models, meta = load_artifacts("model")
T, F, C = meta["target_column"], meta["feature_columns"], meta["classes"]
base = pd.read_csv("test_data.csv")

def expect_error(df, text):
    try:
        prepare_data(df, meta)
    except ValueError as e:
        assert text.lower() in str(e).lower(), str(e)
        return
    raise AssertionError("expected ValueError containing: " + text)

def test_valid_all_models():
    X, y, w = prepare_data(base, meta)
    for name, p in models.items():
        m, cm, rep, pred = evaluate(p, X, y, C)
        assert 0 <= m["Accuracy"] <= 1 and cm.shape == (len(C), len(C)) and len(pred) == len(X)

def test_empty_and_header_only():
    expect_error(pd.DataFrame(), "empty")
    expect_error(base.iloc[0:0], "empty")

def test_missing_column():
    expect_error(base.drop(columns=[F[0]]), F[0])
    expect_error(base.drop(columns=[T]), T)

def test_unknown_label():
    d = base.copy(); d[T] = "Alien"; expect_error(d, "never trained")

def test_all_target_nan():
    d = base.copy(); d[T] = np.nan; expect_error(d, "valid target")

def test_dirty_values_still_work():
    d = base.copy()
    num = meta["numeric_features"][0]
    d[num] = d[num].astype(object); d.loc[d.index[:5], num] = "abc"       # text in numeric col
    d.loc[d.index[5:10], meta["numeric_features"][1]] = np.nan             # NaNs
    if meta["categorical_features"]:
        d.loc[d.index[:5], meta["categorical_features"][0]] = "NEW_UNSEEN"  # unseen category
    d["extra"] = 1
    X, y, w = prepare_data(d, meta)
    assert any("non-numeric" in x for x in w)
    for p in models.values():
        evaluate(p, X, y, C)

def test_single_row_and_single_class():
    X, y, w = prepare_data(base.iloc[:1], meta)
    for p in models.values():
        m, *_ = evaluate(p, X, y, C)       # AUC -> nan, must not crash
        assert np.isnan(m["AUC"])
    one = base[base[T].astype(str) == C[0]]
    X, y, w = prepare_data(one, meta)
    for p in models.values():
        evaluate(p, X, y, C)

def test_shuffled_columns_and_whitespace_headers():
    d = base[list(reversed(base.columns))].copy()
    d.columns = [f" {c} " for c in d.columns]
    X, y, w = prepare_data(d, meta)
    evaluate(models["Logistic Regression"], X, y, C)

def test_csv_roundtrip_numeric_target():
    buf = io.StringIO(); base.to_csv(buf, index=False); buf.seek(0)
    X, y, w = prepare_data(pd.read_csv(buf), meta)
    assert set(y) <= set(C)

if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("PASS", k)
