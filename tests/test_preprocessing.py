import numpy as np
import pandas as pd
import pytest


pytest.importorskip("sklearn")


def test_standardize_columns():
    from gainify_stock_predictor.preprocessing import standardize_columns

    df = pd.DataFrame({
        "date": ["2024-01-01"],
        "open": [100],
        "high": [110],
        "low": [95],
        "close": [105],
        "volume": [1000],
    })

    out = standardize_columns(df)

    assert "Date" in out.columns
    assert "Open" in out.columns
    assert "High" in out.columns
    assert "Low" in out.columns
    assert "Close" in out.columns
    assert "Volume" in out.columns


def test_clean_numeric_cols():
    from gainify_stock_predictor.preprocessing import clean_numeric_cols

    df = pd.DataFrame({
        "Change %": ["1.5%", "-2.0%", "3,000"],
    })

    out = clean_numeric_cols(df, ["Change %"])

    assert pd.api.types.is_numeric_dtype(out["Change %"])


def test_make_sequences_masked():
    from gainify_stock_predictor.preprocessing import make_sequences_masked

    X = np.arange(50).reshape(10, 5).astype(float)
    y_dict = {
        "r1": np.arange(10).reshape(-1, 1).astype(float),
        "dir": np.ones((10, 1)).astype(float),
    }

    mask = np.array([True] * 10)

    out, idxs = make_sequences_masked(X, y_dict, L=3, mask=mask)

    assert "X" in out
    assert "r1" in out
    assert "dir" in out
    assert out["X"].shape[1] == 3
    assert len(idxs) == out["X"].shape[0]


def test_apply_label_smoothing():
    from gainify_stock_predictor.preprocessing import apply_label_smoothing

    y = np.array([[0.0], [1.0]])
    out = apply_label_smoothing(y, eps=0.01)

    assert out[0, 0] > 0.0
    assert out[1, 0] < 1.0
