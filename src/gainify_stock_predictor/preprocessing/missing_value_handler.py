"""
Missing value and numeric cleaning helpers.

Function moved from yearly.py without changing behavior.
"""

import pandas as pd


def clean_numeric_cols(df, cols):
    """
    Clean numeric columns by removing commas and percentage signs,
    then converting to numeric.
    """
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df