"""
Prediction CSV writer.

Physically extracted from legacy/yearly.py.
Original output format is preserved.
"""

import os
import logging

import pandas as pd

from configs.paths_config import OUTPUT_DIR


log = logging.getLogger(__name__)

def save_master_predictions_csv(all_predictions, cutoff_date):
    """
    Save a master CSV of predictions for all processed stocks.
    File: Outputs/Master_Predictions/master_predictions_<cutoff_date>.csv
    Each row = one stock prediction with all available fields.
    """
    if not all_predictions:
        log.info("[CSV] No predictions to save.")
        return
    out_dir = os.path.join(OUTPUT_DIR, "Master_Predictions")
    os.makedirs(out_dir, exist_ok=True)
    cutoff_str = str(cutoff_date.date() if hasattr(cutoff_date, "date") else cutoff_date)
    fpath = os.path.join(out_dir, f"master_predictions_{cutoff_str}.csv")
    df_out = pd.DataFrame(all_predictions)
    df_out.to_csv(fpath, index=False)
    log.info(f"[CSV] Master predictions saved: {fpath}  ({len(df_out)} rows)")
    return fpath
