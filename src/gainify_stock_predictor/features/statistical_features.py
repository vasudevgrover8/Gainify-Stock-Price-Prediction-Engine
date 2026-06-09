"""
Statistical and combined statistics-calculus feature helpers.

Function bodies physically extracted from legacy/yearly.py.
Original logic is preserved.
"""

import numpy as np
import pandas as pd

from gainify_stock_predictor.features.calculus_features import (
    _rolling_slope,
    _rolling_linear_r2,
    _rolling_quadratic_curvature,
    _consecutive_condition_count,
)


EPS = 1e-9


def _safe_div(a, b):
    return a / (b + EPS)


def _clip_series(s, lo=-1e6, hi=1e6):
    return pd.Series(s).replace([np.inf, -np.inf], np.nan).clip(lo, hi)


def _rolling_z(s, window=20):
    s = pd.Series(s).astype(float)
    return (s - s.rolling(window).mean()) / (s.rolling(window).std() + EPS)


def _robust_z(s, window=20):
    s = pd.Series(s).astype(float)
    med = s.rolling(window).median()
    mad = (s - med).abs().rolling(window).median()
    return 0.6745 * (s - med) / (mad + EPS)


def _rolling_percentile(s, window=60):
    s = pd.Series(s).astype(float)
    return s.rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 1 else np.nan,
        raw=False
    )


def _rolling_entropy(s, window=20, bins=10):
    s = pd.Series(s).astype(float)

    def _ent(x):
        x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) < 5:
            return np.nan
        hist, _ = np.histogram(x, bins=bins)
        p = hist / (hist.sum() + EPS)
        p = p[p > 0]
        return -np.sum(p * np.log(p + EPS))

    return s.rolling(window).apply(_ent, raw=False)


def _rolling_autocorr(s, lag=1, window=20):
    s = pd.Series(s).astype(float)

    def _acf(x):
        x = pd.Series(x).dropna()
        if len(x) <= lag + 3:
            return np.nan
        return x.autocorr(lag=lag)

    return s.rolling(window).apply(_acf, raw=False)


def _hurst_approx(s, window=60):
    s = pd.Series(s).astype(float)

    def _hurst(x):
        x = pd.Series(x).dropna().values
        if len(x) < 30:
            return np.nan
        y = x - np.mean(x)
        z = np.cumsum(y)
        r = np.max(z) - np.min(z)
        sd = np.std(x) + EPS
        return np.log((r / sd) + EPS) / np.log(len(x) + EPS)

    return s.rolling(window).apply(_hurst, raw=False)


def _variance_ratio(s, window=20, lag=5):
    s = pd.Series(s).astype(float)

    def _vr(x):
        x = pd.Series(x).dropna()
        if len(x) < lag + 5:
            return np.nan
        var_1 = x.diff().var()
        var_q = x.diff(lag).var()
        return var_q / (lag * var_1 + EPS)

    return s.rolling(window + lag).apply(_vr, raw=False)


def add_raw_statistics_and_calculus(df):
    df = df.copy()

    # Statistics
    df["Robust_Return_Z20"] = _robust_z(df["LogRet"], 20)
    df["Rolling_Median_Return_20"] = df["LogRet"].rolling(20).median()
    df["Rolling_MAD_Return_20"] = (df["LogRet"] - df["Rolling_Median_Return_20"]).abs().rolling(20).median()
    df["Return_IQR_60"] = df["LogRet"].rolling(60).quantile(0.75) - df["LogRet"].rolling(60).quantile(0.25)
    df["Rolling_Skew_20"] = df["LogRet"].rolling(20).skew()
    df["Rolling_Kurtosis_20"] = df["LogRet"].rolling(20).kurt()
    df["Rolling_Sharpe_20"] = df["LogRet"].rolling(20).mean() / (df["LogRet"].rolling(20).std() + EPS)
    df["Rolling_TStat_Return_20"] = df["LogRet"].rolling(20).mean() / (
        df["LogRet"].rolling(20).std() / np.sqrt(20) + EPS
    )

    df["Entropy_Return_20"] = _rolling_entropy(df["LogRet"], 20)
    df["Autocorr_Return_1_20"] = _rolling_autocorr(df["LogRet"], 1, 20)
    df["Autocorr_Return_5_60"] = _rolling_autocorr(df["LogRet"], 5, 60)
    df["Hurst_Exponent_60"] = _hurst_approx(df["LogRet"], 60)
    df["Variance_Ratio_20"] = _variance_ratio(df["LogRet"], 20, 5)

    df["RSI_Percentile_60"] = _rolling_percentile(df["RSI14"], 60)
    df["ATR_Percentile_60"] = _rolling_percentile(df["ATR_Pct"], 60)
    df["Volume_Percentile_60"] = _rolling_percentile(df["Volume"], 60)
    df["Range_Percentile_60"] = _rolling_percentile(df["Range"], 60)
    df["Volatility_Percentile_60"] = _rolling_percentile(df["Vol20"], 60)

    if "^NSEI" in df.columns:
        nifty_ret = pd.to_numeric(df["^NSEI"], errors="coerce").pct_change()
        df["Rolling_Cov_Stock_Nifty_60"] = df["LogRet"].rolling(60).cov(nifty_ret)
    else:
        df["Rolling_Cov_Stock_Nifty_60"] = 0.0

    df["Beta_Stability_60"] = df["Rolling_Beta_60"].rolling(60).std()

    # Calculus / dynamics
    df["Price_Slope_5"] = _rolling_slope(df["Close"], 5)
    df["Price_Slope_20"] = _rolling_slope(df["Close"], 20)
    df["EMA20_Slope"] = df["EMA20"].pct_change(5)
    df["EMA50_Slope"] = df["EMA50"].pct_change(5)
    df["EMA20_Acceleration"] = df["EMA20_Slope"].diff(5)

    df["Rolling_Linear_Trend_R2_20"] = _rolling_linear_r2(df["Close"], 20)
    df["Rolling_Quadratic_Curvature_20"] = _rolling_quadratic_curvature(df["Close"], 20)
    df["Trend_Convexity_20"] = df["Rolling_Quadratic_Curvature_20"]
    df["Price_Inflection_Flag"] = (
        np.sign(df["Rolling_Quadratic_Curvature_20"]) != np.sign(df["Rolling_Quadratic_Curvature_20"].shift(1))
    ).astype(float)

    df["RSI_Velocity"] = df["RSI14"].diff()
    df["RSI_Acceleration"] = df["RSI_Velocity"].diff()
    df["MACD_Hist_Velocity"] = df["MACD_Hist"].diff()
    df["MACD_Hist_Acceleration"] = df["MACD_Hist_Velocity"].diff()

    df["RSI_Turning_Point"] = (np.sign(df["RSI_Velocity"]) != np.sign(df["RSI_Velocity"].shift(1))).astype(float)
    df["MACD_Hist_Turning_Point"] = (
        np.sign(df["MACD_Hist_Velocity"]) != np.sign(df["MACD_Hist_Velocity"].shift(1))
    ).astype(float)

    df["OBV_Velocity"] = df["OBV"].diff()
    df["CMF_Velocity"] = df["CMF20"].diff()
    df["ATR_Velocity"] = df["ATR_Pct"].diff()
    df["Volatility_Slope_20"] = _rolling_slope(df["Vol20"], 20)
    df["Volatility_Acceleration_20"] = df["Volatility_Slope_20"].diff()
    df["Volume_Acceleration"] = df["Volume"].diff().diff()

    df["Drawdown_Velocity"] = df["Max_Drawdown_20"].diff()
    df["Drawdown_Acceleration"] = df["Drawdown_Velocity"].diff()

    return df


def add_statistical_features(df):
    """
    Compatibility alias.
    Calls the original full add_raw_statistics_and_calculus().
    """
    return add_raw_statistics_and_calculus(df)
