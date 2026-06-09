"""
Training utility functions.

Physically extracted from legacy/yearly.py.
Original logic is preserved.
"""

import os
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from configs.model_config import SEQ_LEN, LABEL_SMOOTH_EPS
from configs.training_config import EMBARGO_STEPS
from configs.training_config import (
    MONTHLY_LOOKBACK_DAYS,
    WEEKLY_LOOKBACK_DAYS,
    DAILY_LOOKBACK_DAYS,
)

from gainify_stock_predictor.preprocessing.sequence_builder import (
    cumulative_logret_forward,
    make_sequences_masked,
    apply_label_smoothing,
)

from gainify_stock_predictor.features.feature_pipeline import build_features_from_df


log = logging.getLogger(__name__)

def merge_small_buckets(buckets, min_size=20):
    merged       = {}
    generic_pool = []
    for (vol_level, sector), items in buckets.items():
        if len(items) < min_size:
            generic_pool.extend(items)
        else:
            merged[(vol_level, sector)] = items
    if generic_pool:
        vol_groups = {}
        for item in generic_pool:
            sym, df, fcols, bcols = item
            annual_vol = calculate_annualized_volatility(df)
            vl         = get_volatility_level(annual_vol)
            vol_groups.setdefault(vl, []).append(item)
        for vl, items_list in vol_groups.items():
            key = (vl, "GENERIC")
            if key in merged:
                merged[key].extend(items_list)
            else:
                merged[key] = items_list
    return merged


def compute_beta(df, nifty_col="^NSEI", window=252):
    if nifty_col not in df.columns:
        return np.nan
    stock_ret  = df["Close"].pct_change().dropna()
    market_ret = df[nifty_col].pct_change().dropna()
    common_idx = stock_ret.index.intersection(market_ret.index)
    if len(common_idx) < 60:
        return np.nan
    sr  = stock_ret.loc[common_idx]
    mr  = market_ret.loc[common_idx]
    cov = np.cov(sr, mr)[0, 1]
    var = np.var(mr) + 1e-12
    return cov / var


def compute_momentum_score(df, lookback=126):
    if len(df) < lookback + 1:
        return 0.0
    return float(df["Close"].iloc[-1] / df["Close"].iloc[-(lookback+1)] - 1)


def compute_mean_reversion_score(df, lookback=20):
    if len(df) < lookback + 1:
        return 0.0
    ma = df["Close"].rolling(lookback).mean().iloc[-1]
    sd = df["Close"].rolling(lookback).std().iloc[-1] + 1e-9
    return float((df["Close"].iloc[-1] - ma) / sd)


def make_dir_labels_1d(y1_fwd, close, high, low, atr_period=14):
    close      = close.astype(float)
    high       = high.astype(float)
    low        = low.astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    atr     = tr.rolling(atr_period).mean().bfill()
    atr_pct = (atr / (close + 1e-9)).values.reshape(-1, 1)
    atr_pct = np.clip(atr_pct, 1e-6, None)
    norm_ret = y1_fwd / atr_pct
    thresh   = 0.25
    dir_all  = np.full_like(norm_ret, 0.5, dtype=float)
    dir_all[norm_ret >  thresh] = 1.0
    dir_all[norm_ret < -thresh] = 0.0
    return dir_all


def build_and_save_sequences_for_stock(df, feature_cols_union, symbol, npz_dir,
                                       seq_len=SEQ_LEN, stage="yearly_pretrain",
                                       cutoff_date=None):
    for c in feature_cols_union:
        if c not in df.columns:
            df[c] = 0.0

    X_all   = df[feature_cols_union].values.astype(np.float32)
    X_all   = np.nan_to_num(X_all, nan=0.0, posinf=1e6, neginf=-1e6)
    y1_raw  = df[["LogRet"]].values.astype(np.float32)
    y1_fwd  = cumulative_logret_forward(y1_raw, horizon=1).astype(np.float32)
    dir_all = make_dir_labels_1d(y1_fwd, df["Close"], df["High"], df["Low"])

    mask_valid = ~np.isnan(y1_fwd.ravel())
    if not mask_valid.any() or mask_valid.sum() < 20:
        log.info(f"  [Skip] {symbol}: insufficient valid samples ({mask_valid.sum()})")
        return None

    last_valid = np.where(mask_valid)[0].max()
    X_all      = X_all[:last_valid+1]
    y1_all     = y1_fwd[:last_valid+1]
    dir_all    = dir_all[:last_valid+1]
    mask_valid = mask_valid[:last_valid+1]
    B_all = df[["BreakReliab", "RevAfterHigh", "RevAfterLow"]].values.astype(np.float32)[:last_valid+1]

    n       = len(X_all)
    min_val = max(20, min(60, n // 5))
    cut     = n - min_val
    cut_lo  = max(0, cut - EMBARGO_STEPS)
    cut_hi  = min(n, cut + EMBARGO_STEPS)

    m_tr  = np.zeros(n, dtype=bool); m_tr[:cut_lo]  = True; m_tr  &= mask_valid
    m_val = np.zeros(n, dtype=bool); m_val[cut_hi:] = True; m_val &= mask_valid

    if not m_tr.any():
        log.info(f"  [Skip seq] {symbol}: no train samples")
        return None

    x_scaler = MinMaxScaler().fit(X_all[m_tr])
    b_scaler = MinMaxScaler().fit(B_all[m_tr])
    X_scaled = x_scaler.transform(X_all).astype(np.float32)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    B_scaled = b_scaler.transform(B_all).astype(np.float32)
    B_scaled = np.nan_to_num(B_scaled, nan=0.0)

    X_scaled = np.clip(X_scaled, -500.0, 500.0).astype(np.float32)
    B_scaled = np.clip(B_scaled, -500.0, 500.0).astype(np.float32)

    pack_tr, _  = make_sequences_masked(X_scaled, {"y1": y1_all, "dir": dir_all.reshape(-1,1), "B": B_scaled}, seq_len, m_tr)
    pack_val, _ = make_sequences_masked(X_scaled, {"y1": y1_all, "dir": dir_all.reshape(-1,1), "B": B_scaled}, seq_len, m_val)

    if "X" not in pack_tr or pack_tr["X"].shape[0] < 5:
        log.info(f"  [Skip seq] {symbol}: too few sequences ({pack_tr['X'].shape[0]})")
        return None

    y1_mu   = pack_tr["y1"].mean(); y1_sd = pack_tr["y1"].std() + 1e-9
    int_tr  = np.tanh((pack_tr["y1"]  - y1_mu) / (0.8 * y1_sd)).astype(np.float32)
    int_val = (np.tanh((pack_val["y1"] - y1_mu) / (0.8 * y1_sd)).astype(np.float32)
               if pack_val["X"].shape[0] > 0 else np.zeros((0,1), dtype=np.float32))

    dir_tr  = apply_label_smoothing(pack_tr["dir"]).astype(np.float32)
    dir_val = (apply_label_smoothing(pack_val["dir"]).astype(np.float32)
               if pack_val["X"].shape[0] > 0 else np.zeros((0,1), dtype=np.float32))

    # Cache key includes stage and cutoff_date to prevent cross-stage collisions
    cutoff_str = str(cutoff_date.date() if hasattr(cutoff_date, "date") else cutoff_date) if cutoff_date else "nodate"
    cache_tag  = f"{symbol}_{stage}_{cutoff_str}_sl{seq_len}"

    os.makedirs(npz_dir, exist_ok=True)
    train_fp = os.path.join(npz_dir, f"{cache_tag}_train.npz")
    np.savez_compressed(train_fp,
        X=pack_tr["X"].astype(np.float32), B=pack_tr["B"].astype(np.float32),
        y1=pack_tr["y1"].astype(np.float32), dir=dir_tr, inten=int_tr)

    val_fp = None
    if pack_val["X"].shape[0] > 0:
        val_fp = os.path.join(npz_dir, f"{cache_tag}_val.npz")
        np.savez_compressed(val_fp,
            X=pack_val["X"].astype(np.float32), B=pack_val["B"].astype(np.float32),
            y1=pack_val["y1"].astype(np.float32), dir=dir_val, inten=int_val)

    return train_fp, val_fp


def prepare_single_stock_arrays(df, feature_cols, seq_len):
    X_all   = df[feature_cols].values.astype(float)
    y1_raw  = df[["LogRet"]].values.astype(float)
    y1_fwd  = cumulative_logret_forward(y1_raw, horizon=1)
    y1_all  = y1_fwd
    dir_all = make_dir_labels_1d(y1_fwd, df["Close"], df["High"], df["Low"])

    mask_valid = ~np.isnan(y1_all.ravel())
    if not mask_valid.any() or mask_valid.sum() < 30:
        return None

    last_valid = np.where(mask_valid)[0].max()
    X_all      = X_all[:last_valid+1]
    y1_all     = y1_all[:last_valid+1]
    dir_all    = dir_all[:last_valid+1]
    mask_valid = mask_valid[:last_valid+1]
    B_all = df[["BreakReliab", "RevAfterHigh", "RevAfterLow"]].values.astype(float)[:last_valid+1]

    n      = len(X_all)
    cut    = int(0.8 * n)
    cut_lo = max(0, cut - EMBARGO_STEPS)
    cut_hi = min(n, cut + EMBARGO_STEPS)

    m_tr = np.zeros(n, dtype=bool); m_tr[:cut_lo] = True; m_tr &= mask_valid
    m_te = np.zeros(n, dtype=bool); m_te[cut_hi:] = True; m_te &= mask_valid

    if not m_tr.any() or not m_te.any():
        return None

    x_scaler  = MinMaxScaler().fit(X_all[m_tr])
    b_scaler  = MinMaxScaler().fit(B_all[m_tr])
    y1_scaler = StandardScaler().fit(y1_all[m_tr])

    X  = np.clip(x_scaler.transform(X_all),  -500.0, 500.0).astype(np.float32)
    B  = np.clip(b_scaler.transform(B_all),  -500.0, 500.0).astype(np.float32)
    y1 = y1_scaler.transform(y1_all)

    pack_tr, idxs_tr = make_sequences_masked(X, {"y1": y1, "dir": dir_all.reshape(-1,1), "B": B}, seq_len, m_tr)
    pack_te, idxs_te = make_sequences_masked(X, {"y1": y1, "dir": dir_all.reshape(-1,1), "B": B}, seq_len, m_te)

    if pack_tr["X"].shape[0] == 0 or pack_te["X"].shape[0] == 0:
        return None

    y1_mu, y1_sd = pack_tr["y1"].mean(), pack_tr["y1"].std() + 1e-9
    int_tr = np.tanh((pack_tr["y1"] - y1_mu) / (0.8 * y1_sd)).astype(np.float32)
    int_te = np.tanh((pack_te["y1"] - y1_mu) / (0.8 * y1_sd)).astype(np.float32)
    dir_tr = pack_tr["dir"].astype(np.float32)
    dir_te = apply_label_smoothing(pack_te["dir"])

    return {
        "X_tr_seq": pack_tr["X"], "X_te_seq": pack_te["X"],
        "y1_tr_seq": pack_tr["y1"], "y1_te_seq": pack_te["y1"],
        "dir_tr_seq": dir_tr, "dir_te_seq": dir_te,
        "int_tr_seq": int_tr, "int_te_seq": int_te,
        "B_tr_seq": pack_tr["B"], "B_te_seq": pack_te["B"],
        "x_scaler": x_scaler, "b_scaler": b_scaler, "y1_scaler": y1_scaler,
        "anchor_idx": cut + seq_len - 1,
        "orig_idxs_tr": np.array(idxs_tr),
        "orig_idxs_te": np.array(idxs_te)
    }


def prepare_daily_arrays(df, feature_cols, seq_len, daily_window=60):
    min_rows = seq_len + daily_window
    if len(df) < min_rows:
        log.info(f"  [Daily FT] Not enough rows (need {min_rows}, have {len(df)}) — skipping.")
        return None
    df_recent = df.tail(max(min_rows, 200)).reset_index(drop=True)
    pack = prepare_single_stock_arrays(df_recent, feature_cols, seq_len)
    if pack is None:
        log.info(f"  [Daily FT] prepare_single_stock_arrays returned None — skipping.")
    return pack
