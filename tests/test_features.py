import numpy as np
import pandas as pd


def test_feature_module_imports():
    import gainify_stock_predictor.features as features

    assert features.rsi is not None
    assert features.nw_kernel_smooth is not None
    assert features._ema is not None
    assert features._rolling_z is not None
    assert features._rolling_slope is not None
    assert features._true_range is not None
    assert features._adx_dmi is not None
    assert features.add_raw_advanced_features is not None
    assert features.add_raw_statistics_and_calculus is not None
    assert features.add_probability_ecosystem_features is not None
    assert features.add_price_volume_structure_dots is not None
    assert features.add_family_ecosystem_features is not None
    assert features.add_indicator_internal_dots is not None
    assert features.add_cross_family_dot_connections is not None
    assert features.add_final_market_evidence_scores is not None
    assert features.build_features_from_df is not None
    assert features.run_feature_pipeline is not None


def test_rsi_runs():
    from gainify_stock_predictor.features import rsi

    s = pd.Series([1, 2, 3, 2, 4, 5, 6, 7, 6, 8, 9, 10, 11, 12, 13, 14])
    out = rsi(s, period=14)

    assert len(out) == len(s)


def test_macd_runs():
    from gainify_stock_predictor.features import _macd

    s = pd.Series(np.linspace(100, 120, 80))
    macd, sig, hist = _macd(s)

    assert len(macd) == len(s)
    assert len(sig) == len(s)
    assert len(hist) == len(s)


def test_true_range_runs():
    from gainify_stock_predictor.features import _true_range

    df = pd.DataFrame({
        "High": [110, 112, 115],
        "Low": [100, 101, 104],
        "Close": [105, 108, 111],
    })

    out = _true_range(df)

    assert len(out) == len(df)


def test_statistical_helpers_run():
    from gainify_stock_predictor.features import (
        _rolling_z,
        _robust_z,
        _rolling_entropy,
        _rolling_autocorr,
        _hurst_approx,
        _variance_ratio,
    )

    s = pd.Series(np.random.normal(0, 1, 120))

    assert len(_rolling_z(s, 20)) == len(s)
    assert len(_robust_z(s, 20)) == len(s)
    assert len(_rolling_entropy(s, 20)) == len(s)
    assert len(_rolling_autocorr(s, 1, 20)) == len(s)
    assert len(_hurst_approx(s, 60)) == len(s)
    assert len(_variance_ratio(s, 20, 5)) == len(s)


def test_calculus_helpers_run():
    from gainify_stock_predictor.features import (
        _rolling_slope,
        _rolling_linear_r2,
        _rolling_quadratic_curvature,
    )

    s = pd.Series(np.linspace(100, 120, 80))

    assert len(_rolling_slope(s, 20)) == len(s)
    assert len(_rolling_linear_r2(s, 20)) == len(s)
    assert len(_rolling_quadratic_curvature(s, 20)) == len(s)


def test_relative_strength_helper():
    from gainify_stock_predictor.features import add_relative_strength_features

    df = pd.DataFrame({
        "Change %": [0.1, 0.2, -0.1, 0.3, 0.4, -0.2],
        "^CNXIT": [100, 101, 102, 101, 103, 104],
    })

    out = add_relative_strength_features(df, sec_idx_col="^CNXIT")

    assert "RelativeRet5d" in out.columns
    assert len(out) == len(df)


def test_no_legacy_call_in_main_feature_files():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    feature_dir = root / "src" / "gainify_stock_predictor" / "features"

    files_to_check = [
        "advanced_indicators.py",
        "statistical_features.py",
        "calculus_features.py",
        "probability_features.py",
        "candlestick_features.py",
        "regime_features.py",
        "feature_pipeline.py",
    ]

    for file_name in files_to_check:
        text = (feature_dir / file_name).read_text(encoding="utf-8")
        assert "call_legacy_function" not in text
