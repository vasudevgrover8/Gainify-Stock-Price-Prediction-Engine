"""
Modular stage orchestrator.

This replaces the old yearly.py main() runtime path.

It does NOT call legacy/yearly.py.

It coordinates:
- CSV loading
- feature building
- bucket assignment
- bucket feature union
- yearly/monthly/weekly/daily/predict stage execution
- prediction CSV writing
- ranked report writing
"""

import glob
import logging
import os
from pathlib import Path

import pandas as pd

from configs.paths_config import DATA_DIR
from configs.model_config import MIN_HISTORY_BARS
from configs.training_config import AUTO_DETECT_CUTOFF_DATE
from configs.bucket_config import BUCKETS

from gainify_stock_predictor.preprocessing.csv_loader import (
    detect_latest_dataset_date,
    filter_df_to_cutoff,
)

from gainify_stock_predictor.features.feature_pipeline import build_features_from_df

from gainify_stock_predictor.bucketing.sector_mapper import map_sector_from_metadata
from gainify_stock_predictor.bucketing.volatility_buckets import (
    calculate_annualized_volatility,
    get_volatility_level,
)

from gainify_stock_predictor.training.trainer_utils import merge_small_buckets

from gainify_stock_predictor.training.yearly_pretrain import run_yearly_pretrain
from gainify_stock_predictor.training.monthly_finetune import run_monthly_finetune
from gainify_stock_predictor.training.weekly_finetune import run_weekly_finetune
from gainify_stock_predictor.training.daily_finetune import run_daily_finetune

from gainify_stock_predictor.prediction.predictor import run_predict_only

from gainify_stock_predictor.reporting.prediction_writer import save_master_predictions_csv
from gainify_stock_predictor.reporting.ranked_reports import build_ranked_volatility_reports


log = logging.getLogger(__name__)


VALID_STAGES = {
    "yearly_pretrain",
    "monthly_finetune",
    "weekly_finetune",
    "daily_finetune",
    "predict_only",
    "full_pipeline",
}


def load_symbol_csvs(data_dir=DATA_DIR):
    """
    Load all stock CSV files from DATA_DIR.
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    print(f'[INFO] Scanning data directory: {data_dir}', flush=True)
    print(f'[INFO] CSV files found: {len(csv_files)}', flush=True)

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in DATA_DIR: {data_dir}")

    sym_dfs = {}

    for i, path in enumerate(csv_files, start=1):
        print(f'[LOAD] {i}/{len(csv_files)}: {Path(path).name}', flush=True)
        try:
            symbol = Path(path).stem
            df_raw = pd.read_csv(path, low_memory=False)

            if df_raw is None or df_raw.empty:
                log.warning(f"[Skip] Empty CSV: {path}")
                continue

            sym_dfs[symbol] = df_raw

        except Exception as e:
            log.warning(f"[Skip] Failed to read {path}: {e}")

    return sym_dfs


def assign_bucket_from_df(sym, df_raw_cut):
    """
    Modular equivalent of the old yearly.py bucket assignment.
    """
    try:
        annual_vol = calculate_annualized_volatility(df_raw_cut)
        vol_level = get_volatility_level(annual_vol)

        sector_str = None
        industry_str = None

        if "sector" in df_raw_cut.columns:
            sector_str = df_raw_cut["sector"].iloc[-1]

        if "industry" in df_raw_cut.columns:
            industry_str = df_raw_cut["industry"].iloc[-1]

        sector = map_sector_from_metadata(sector_str, industry_str)

        valid_pairs = set(BUCKETS.values())

        if (vol_level, sector) in valid_pairs:
            return vol_level, sector

        if sector == "GENERIC":
            return vol_level, "GENERIC"

        return "UNKNOWN", "UNKNOWN"

    except Exception as e:
        log.warning(f"[Bucket] Failed for {sym}: {e}")
        return "UNKNOWN", "UNKNOWN"


def build_buckets_for_stage(data_dir=DATA_DIR, cutoff_date=None):
    """
    Load CSVs, build features, assign buckets, merge small buckets,
    and build bucket-wise feature unions.
    """
    if cutoff_date is None:
        if AUTO_DETECT_CUTOFF_DATE:
            cutoff_date = detect_latest_dataset_date(data_dir)
        else:
            cutoff_date = pd.Timestamp.today().normalize()

    sym_dfs = load_symbol_csvs(data_dir)

    buckets = {}
    total_stocks = 0

    for idx, (sym, df_raw) in enumerate(sym_dfs.items(), start=1):
        print(f'[FEATURES] {idx}/{len(sym_dfs)} Building features for {sym}', flush=True)
        df_raw_cut = filter_df_to_cutoff(df_raw, cutoff_date)

        if len(df_raw_cut) < MIN_HISTORY_BARS:
            log.info(f"[Skip] {sym}: only {len(df_raw_cut)} bars after cutoff filter")
            continue

        try:
            df, fcols, bcols = build_features_from_df(df_raw_cut)
        except Exception as e:
            log.warning(f"[Skip] {sym}: feature build failed - {e}")
            continue

        vol_level, sector = assign_bucket_from_df(sym, df_raw_cut)

        buckets.setdefault((vol_level, sector), []).append((sym, df, fcols, bcols))
        total_stocks += 1

    log.info(f"Total stocks loaded: {total_stocks}")

    buckets = merge_small_buckets(buckets, min_size=20)

    for (vol_level, sector), items in sorted(buckets.items()):
        log.info(f"Bucket ({vol_level}, {sector}): {len(items)} stocks")

    bucket_feature_union = {}

    for key, items in buckets.items():
        union = []

        for sym, df, fcols, bcols in items:
            for c in fcols:
                if c not in union:
                    union.append(c)

        bucket_feature_union[key] = union

    return buckets, bucket_feature_union, cutoff_date


def run_stage(stage, data_dir=DATA_DIR, cutoff_date=None):
    """
    Run one modular stage directly.

    This does not call legacy/yearly.py.
    """
    if stage not in VALID_STAGES:
        raise ValueError(f"Invalid stage: {stage}. Valid stages: {sorted(VALID_STAGES)}")

    buckets, bucket_feature_union, run_cutoff_date = build_buckets_for_stage(
        data_dir=data_dir,
        cutoff_date=cutoff_date,
    )

    all_predictions = []

    if stage == "yearly_pretrain":
        run_yearly_pretrain(buckets, bucket_feature_union, run_cutoff_date)

    elif stage == "monthly_finetune":
        run_monthly_finetune(buckets, bucket_feature_union, run_cutoff_date, all_predictions)

    elif stage == "weekly_finetune":
        run_weekly_finetune(buckets, bucket_feature_union, run_cutoff_date, all_predictions)

    elif stage == "daily_finetune":
        run_daily_finetune(buckets, bucket_feature_union, run_cutoff_date, all_predictions)

    elif stage == "predict_only":
        run_predict_only(buckets, bucket_feature_union, run_cutoff_date, all_predictions)

    elif stage == "full_pipeline":
        log.info("[Full Pipeline] Running all stages sequentially.")
        run_yearly_pretrain(buckets, bucket_feature_union, run_cutoff_date)
        run_monthly_finetune(buckets, bucket_feature_union, run_cutoff_date, all_predictions)
        run_weekly_finetune(buckets, bucket_feature_union, run_cutoff_date, all_predictions)
        run_daily_finetune(buckets, bucket_feature_union, run_cutoff_date, all_predictions)

    if all_predictions:
        save_master_predictions_csv(all_predictions, run_cutoff_date)
        build_ranked_volatility_reports(all_predictions, run_cutoff_date, top_n=20)

    log.info(f"[Done] Modular stage '{stage}' completed. cutoff={run_cutoff_date.date()}")

    return all_predictions


def run_yearly():
    return run_stage("yearly_pretrain")


def run_monthly():
    return run_stage("monthly_finetune")


def run_weekly():
    return run_stage("weekly_finetune")


def run_daily():
    return run_stage("daily_finetune")


def run_predict():
    return run_stage("predict_only")
