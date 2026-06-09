"""
Technical indicator helper functions.

Physically moved from legacy/yearly.py.
Core logic is preserved.
"""

import numpy as np
import pandas as pd


EPS = 1e-9


def rsi(series, period=14):
    d = series.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = -d.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def nw_kernel_smooth(series, h=5, window=20):
    arr = series.values
    out = np.full_like(arr, np.nan, dtype=float)

    for i in range(len(arr)):
        start = max(0, i - window)
        idx = np.arange(start, i + 1)
        w = np.exp(-0.5 * ((i - idx) / h) ** 2)
        out[i] = np.sum(w * arr[start:i + 1]) / (np.sum(w) + 1e-9)

    return out


def _ema(s, span):
    return pd.Series(s).astype(float).ewm(span=span, adjust=False).mean()


def _wma(s, window):
    s = pd.Series(s).astype(float)
    weights = np.arange(1, window + 1)
    return s.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def _hma(s, window=20):
    half = max(2, int(window / 2))
    sqrt_w = max(2, int(np.sqrt(window)))
    return _wma(2 * _wma(s, half) - _wma(s, window), sqrt_w)


def _kama(close, er_window=10, fast=2, slow=30):
    close = pd.Series(close).astype(float)
    change = close.diff(er_window).abs()
    volatility = close.diff().abs().rolling(er_window).sum()
    er = change / (volatility + EPS)

    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = np.full(len(close), np.nan)

    if len(close) == 0:
        return pd.Series(kama, index=close.index)

    kama[0] = close.iloc[0]

    for i in range(1, len(close)):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i - 1]
        else:
            kama[i] = kama[i - 1] + sc.iloc[i] * (close.iloc[i] - kama[i - 1])

    return pd.Series(kama, index=close.index)


def _macd(close, fast=12, slow=26, signal=9):
    macd = _ema(close, fast) - _ema(close, slow)
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist


def _tsi(close, long=25, short=13):
    mom = pd.Series(close).diff()
    abs_mom = mom.abs()
    tsi = 100 * _ema(_ema(mom, long), short) / (_ema(_ema(abs_mom, long), short) + EPS)
    return tsi


def _stochastic(df, period=14, smooth=3):
    low_min = df["Low"].rolling(period).min()
    high_max = df["High"].rolling(period).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min + EPS)
    d = k.rolling(smooth).mean()
    return k, d


def _ultimate_oscillator(df, p1=7, p2=14, p3=28):
    prev_close = df["Close"].shift(1)

    bp = df["Close"] - pd.concat([df["Low"], prev_close], axis=1).min(axis=1)
    tr = (
        pd.concat([df["High"], prev_close], axis=1).max(axis=1)
        - pd.concat([df["Low"], prev_close], axis=1).min(axis=1)
    )

    avg1 = bp.rolling(p1).sum() / (tr.rolling(p1).sum() + EPS)
    avg2 = bp.rolling(p2).sum() / (tr.rolling(p2).sum() + EPS)
    avg3 = bp.rolling(p3).sum() / (tr.rolling(p3).sum() + EPS)

    return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7


def _connors_rsi(close, rsi_period=3, streak_rsi_period=2, rank_period=100):
    close = pd.Series(close).astype(float)

    rsi_close = rsi(close, rsi_period)

    up = close > close.shift(1)
    down = close < close.shift(1)

    streak = np.zeros(len(close))

    for i in range(1, len(close)):
        if up.iloc[i]:
            streak[i] = max(1, streak[i - 1] + 1)
        elif down.iloc[i]:
            streak[i] = min(-1, streak[i - 1] - 1)
        else:
            streak[i] = 0

    streak_rsi = rsi(pd.Series(streak, index=close.index), streak_rsi_period)
    roc1 = close.pct_change()

    percent_rank = roc1.rolling(rank_period).apply(
        lambda x: 100 * pd.Series(x).rank(pct=True).iloc[-1],
        raw=False,
    )

    return (rsi_close + streak_rsi + percent_rank) / 3


def _fisher_transform(df, period=10):
    high_max = df["High"].rolling(period).max()
    low_min = df["Low"].rolling(period).min()

    x = 2 * ((df["Close"] - low_min) / (high_max - low_min + EPS) - 0.5)
    x = x.clip(-0.999, 0.999)

    fisher = 0.5 * np.log((1 + x) / (1 - x + EPS))

    return fisher
