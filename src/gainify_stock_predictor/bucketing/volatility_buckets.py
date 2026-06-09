"""
Volatility bucket helpers.

Moved from yearly.py.
"""

import numpy as np

from configs.bucket_config import IPO_RECENT_DAYS


def calculate_annualized_volatility(df, window=252):
    if "LogRet" not in df.columns:
        df["LogRet"] = np.log(df["Close"] / df["Close"].shift(1))

    log_rets = df["LogRet"].dropna()

    if len(log_rets) < 20:
        return 0.3

    lookback = min(window, len(log_rets))
    recent_rets = log_rets.iloc[-lookback:]

    daily_vol = recent_rets.std()
    annual_vol = daily_vol * np.sqrt(252)

    return np.clip(annual_vol, 0.05, 1.5)


def get_volatility_level(annual_vol):
    if annual_vol < 0.20:
        return "VERY_LOW"
    elif annual_vol < 0.30:
        return "LOW"
    elif annual_vol < 0.40:
        return "MEDIUM"
    elif annual_vol < 0.60:
        return "HIGH"
    else:
        return "VERY_HIGH"


def is_ipo_recent(df, threshold_days=IPO_RECENT_DAYS):
    try:
        dates = df["Date"]
        dates = dates.dropna()

        if len(dates) < 2:
            return True

        import pandas as pd

        dates = pd.to_datetime(dates, errors="coerce").dropna()

        listing_date = dates.min()
        today = pd.Timestamp.today().normalize()

        return (today - listing_date).days < threshold_days

    except Exception:
        return False
