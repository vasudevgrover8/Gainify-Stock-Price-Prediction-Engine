"""
Relative strength feature wrappers.

Original source:
legacy/yearly.py

Your full relative strength calculations are currently inside build_features_from_df().
This file provides a safe helper for separate modular access later.
"""

import numpy as np
import pandas as pd


def add_relative_strength_features(df, sec_idx_col=None):
    """
    Lightweight modular helper.

    NOTE:
    The complete original relative-strength feature logic is still preserved
    inside legacy/yearly.py -> build_features_from_df().

    This helper only adds RelativeRet5d if called independently.
    """
    df = df.copy()

    if "Change %" not in df.columns:
        df["RelativeRet5d"] = 0.0
        return df

    if sec_idx_col and sec_idx_col in df.columns:
        sec_ret = pd.to_numeric(df[sec_idx_col], errors="coerce").pct_change()
        df["RelativeRet5d"] = df["Change %"].rolling(5).sum() - sec_ret.rolling(5).sum()
    else:
        df["RelativeRet5d"] = 0.0

    df["RelativeRet5d"] = df["RelativeRet5d"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df
