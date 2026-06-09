"""
Calculus feature helper functions.

Physically moved from legacy/yearly.py.
Core logic is preserved.
"""

import numpy as np
import pandas as pd


EPS = 1e-9


def _rolling_slope(s, window=20):
    s = pd.Series(s).astype(float)

    def _slope(x):
        x = pd.Series(x).dropna().values

        if len(x) < 5:
            return np.nan

        t = np.arange(len(x))
        coef = np.polyfit(t, x, 1)[0]

        return coef

    return s.rolling(window).apply(_slope, raw=False)


def _rolling_linear_r2(s, window=20):
    s = pd.Series(s).astype(float)

    def _r2(x):
        x = pd.Series(x).dropna().values

        if len(x) < 5:
            return np.nan

        t = np.arange(len(x))
        coef = np.polyfit(t, x, 1)
        pred = coef[0] * t + coef[1]

        ss_res = np.sum((x - pred) ** 2)
        ss_tot = np.sum((x - np.mean(x)) ** 2) + EPS

        return 1 - ss_res / ss_tot

    return s.rolling(window).apply(_r2, raw=False)


def _rolling_quadratic_curvature(s, window=20):
    s = pd.Series(s).astype(float)

    def _curv(x):
        x = pd.Series(x).dropna().values

        if len(x) < 8:
            return np.nan

        t = np.arange(len(x))
        coef = np.polyfit(t, x, 2)

        return 2 * coef[0]

    return s.rolling(window).apply(_curv, raw=False)


def _consecutive_condition_count(cond):
    cond = pd.Series(cond).fillna(False).astype(bool)

    out = np.zeros(len(cond), dtype=float)
    run = 0

    for i, v in enumerate(cond.values):
        run = run + 1 if v else 0
        out[i] = run

    return pd.Series(out, index=cond.index)


def add_calculus_features(df):
    """
    Adds the calculus feature subset from the original add_raw_statistics_and_calculus().

    This is separated physically for GitHub modularity.
    """
    df = df.copy()

    df["Price_Slope_5"] = _rolling_slope(df["Close"], 5)
    df["Price_Slope_20"] = _rolling_slope(df["Close"], 20)

    if "EMA20" in df.columns:
        df["EMA20_Slope"] = df["EMA20"].pct_change(3)
        df["EMA20_Acceleration"] = df["EMA20_Slope"].diff()

    if "EMA50" in df.columns:
        df["EMA50_Slope"] = df["EMA50"].pct_change(3)

    df["Rolling_Linear_Trend_R2_20"] = _rolling_linear_r2(df["Close"], 20)
    df["Rolling_Quadratic_Curvature_20"] = _rolling_quadratic_curvature(df["Close"], 20)

    df["Trend_Convexity_20"] = df["Price_Slope_5"] - df["Price_Slope_20"]

    df["Price_Inflection_Flag"] = (
        np.sign(df["Price_Slope_5"]) != np.sign(df["Price_Slope_5"].shift(1))
    ).astype(float)

    if "RSI14" in df.columns:
        df["RSI_Velocity"] = df["RSI14"].diff()
        df["RSI_Acceleration"] = df["RSI_Velocity"].diff()

    if "MACD_Hist" in df.columns:
        df["MACD_Hist_Slope"] = df["MACD_Hist"].diff()

    if "Vol20" in df.columns:
        df["Volatility_Slope_20"] = _rolling_slope(df["Vol20"], 20)

    return df
