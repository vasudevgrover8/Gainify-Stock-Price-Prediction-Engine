"""
Ranked volatility report builder.

Physically extracted from legacy/yearly.py.
Original report logic is preserved.
"""

import os
import logging

import pandas as pd

from configs.paths_config import OUTPUT_DIR


log = logging.getLogger(__name__)

def build_ranked_volatility_reports(all_predictions, cutoff_date, top_n=20):
    """
    For each volatility class, create top-N and bottom-N CSVs ranked by predicted_return.
    Also saves a combined ranked_summary CSV.
    Files: Outputs/Ranked_Reports/<cutoff_date>/top20_<VOL>_<date>.csv etc.
    """
    if not all_predictions:
        return
    cutoff_str = str(cutoff_date.date() if hasattr(cutoff_date, "date") else cutoff_date)
    report_dir = os.path.join(OUTPUT_DIR, "Ranked_Reports", cutoff_str)
    os.makedirs(report_dir, exist_ok=True)

    df_all = pd.DataFrame(all_predictions)
    if "predicted_return" not in df_all.columns or "volatility_class" not in df_all.columns:
        log.info("[Ranked] Missing required columns; skipping ranked reports.")
        return

    df_all["predicted_return"] = pd.to_numeric(df_all["predicted_return"], errors="coerce")
    df_all = df_all.dropna(subset=["predicted_return"])

    rank_cols = ["rank", "symbol", "sector", "volatility_class", "bucket_name",
                 "last_close", "predicted_price", "predicted_return",
                 "predicted_direction", "confidence"]
    rank_cols = [c for c in rank_cols if c in df_all.columns]

    summary_rows = []
    for vol_class in df_all["volatility_class"].unique():
        sub = df_all[df_all["volatility_class"] == vol_class].copy()
        sub = sub.sort_values("predicted_return", ascending=False).reset_index(drop=True)
        sub["rank"] = sub.index + 1

        top = sub.head(top_n)[rank_cols]
        top.to_csv(os.path.join(report_dir, f"top{top_n}_{vol_class}_{cutoff_str}.csv"), index=False)

        bottom = sub.tail(top_n).sort_values("predicted_return").reset_index(drop=True)
        bottom["rank"] = bottom.index + 1
        bottom = bottom[rank_cols]
        bottom.to_csv(os.path.join(report_dir, f"bottom{top_n}_{vol_class}_{cutoff_str}.csv"), index=False)
        log.info(f"[Ranked] {vol_class}: top/bottom{top_n} saved.")

        summary_rows.append(sub)

    if summary_rows:
        df_summary = pd.concat(summary_rows, ignore_index=True)
        df_summary = df_summary.sort_values("predicted_return", ascending=False).reset_index(drop=True)
        df_summary["overall_rank"] = df_summary.index + 1
        df_summary.to_csv(os.path.join(report_dir, f"ranked_summary_{cutoff_str}.csv"), index=False)
        log.info(f"[Ranked] Overall summary saved: ranked_summary_{cutoff_str}.csv")
