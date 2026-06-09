"""
CSV loading and date-window utilities.

This file contains safe preprocessing helpers adapted from yearly.py.
No model, feature, training, or checkpoint logic is changed.
"""

import glob
import logging
import os

import pandas as pd

from configs.paths_config import DATA_DIR
from configs.training_config import (
    MONTHLY_LOOKBACK_DAYS,
    WEEKLY_LOOKBACK_DAYS,
    DAILY_LOOKBACK_DAYS,
)


log = logging.getLogger(__name__)


def parse_date_column(df):
    """
    Return a sorted Series of parsed dates from any recognisable date column.
    """
    for col in ["Date", "date", "DATE", "Timestamp", "timestamp"]:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce").dropna()
            return parsed.sort_values()

    return pd.Series(dtype="datetime64[ns]")


def detect_latest_dataset_date(data_dir=DATA_DIR):
    """
    Scan all CSV files in data_dir and find the globally latest date present.
    This becomes the run_cutoff_date for whichever stage is currently executing.
    """
    latest = None
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    if not csv_files:
        log.warning(f"No CSV files found in {data_dir} for cutoff-date detection.")
        return pd.Timestamp.today().normalize()

    for path in csv_files:
        try:
            df_peek = pd.read_csv(
                path,
                usecols=lambda c: c.strip().lower() in ["date", "timestamp"],
                low_memory=False,
                nrows=None,
            )

            df_peek.columns = [c.strip() for c in df_peek.columns]
            dates = parse_date_column(df_peek)

            if len(dates) > 0:
                file_latest = dates.max()

                if latest is None or file_latest > latest:
                    latest = file_latest

        except Exception:
            pass

    if latest is None:
        latest = pd.Timestamp.today().normalize()

    log.info(f"Auto-detected dataset latest date: {latest.date()}")
    return latest


def filter_df_to_cutoff(df, cutoff_date):
    """
    Return df rows with Date <= cutoff_date.
    Preserves all columns.
    """
    if "Date" not in df.columns:
        return df

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    cutoff_ts = pd.Timestamp(cutoff_date)

    return df[df["Date"] <= cutoff_ts].reset_index(drop=True)


def apply_stage_window(df, stage, cutoff_date):
    """
    Filter df to the rolling lookback window for the given stage.

    - yearly_pretrain: full history up to cutoff_date
    - monthly_finetune: MONTHLY_LOOKBACK_DAYS before cutoff_date
    - weekly_finetune: WEEKLY_LOOKBACK_DAYS before cutoff_date
    - daily_finetune: DAILY_LOOKBACK_DAYS before cutoff_date
    """
    df = filter_df_to_cutoff(df, cutoff_date)
    cutoff_ts = pd.Timestamp(cutoff_date)

    if stage == "yearly_pretrain":
        return df

    lookback_map = {
        "monthly_finetune": MONTHLY_LOOKBACK_DAYS,
        "weekly_finetune": WEEKLY_LOOKBACK_DAYS,
        "daily_finetune": DAILY_LOOKBACK_DAYS,
    }

    days = lookback_map.get(stage, DAILY_LOOKBACK_DAYS)
    window_start = cutoff_ts - pd.Timedelta(days=days)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df[df["Date"] >= window_start].reset_index(drop=True)

    return df


def load_csv_by_symbol(symbol, data_dir=DATA_DIR):
    """
    Load a stock CSV by symbol from the configured data directory.

    This is a safe helper for modular scripts.
    It does not change your yearly.py training logic.
    """
    candidates = [
        os.path.join(data_dir, f"{symbol}.csv"),
        os.path.join(data_dir, f"{symbol.replace('.NS', '')}.csv"),
        os.path.join(data_dir, f"{symbol.replace('.BO', '')}.csv"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return pd.read_csv(path, low_memory=False)

    raise FileNotFoundError(f"No CSV file found for symbol: {symbol} in {data_dir}")