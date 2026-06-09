"""
Prediction helpers.

Physically extracted from legacy/yearly.py.
Original forecast logic is preserved.
"""

import os
import pickle
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from configs.model_config import SEQ_LEN, DROPOUT_ENC_FINETUNE
from configs.bucket_config import VOLATILITY_LEVELS

from gainify_stock_predictor.models.main_model import build_multitask_model
from gainify_stock_predictor.models.layers import GatingLayer, PositionalEncoding
from gainify_stock_predictor.checkpoints.checkpoint_manager import load_latest_successful_checkpoint


log = logging.getLogger(__name__)

def forecast_1d(df_in, model, feature_cols, b_scaler, x_scaler, y1_scaler, vol_idx, seq_len):
    if len(df_in) < seq_len:
        return None
    last_close = float(df_in["Close"].iloc[-1])
    last_date  = pd.to_datetime(df_in["Date"].iloc[-1]) if "Date" in df_in.columns else pd.Timestamp.today()

    feats = df_in[feature_cols].values
    X_seq = np.clip(x_scaler.transform(feats[-seq_len:]), -500.0, 500.0).reshape(
        1, seq_len, len(feature_cols)).astype(np.float32)
    B_seq = np.clip(b_scaler.transform(
        df_in[["BreakReliab", "RevAfterHigh", "RevAfterLow"]].iloc[-1].values.reshape(1, -1)
    ), -500.0, 500.0).astype(np.float32)
    vol_arr = np.array([[vol_idx]], dtype=np.int32)

    preds   = model.predict({"price_seq": X_seq, "break_feats": B_seq, "vol_level": vol_arr}, verbose=0)
    r1_pred = preds[0]; dirp = preds[1]; inten = preds[2]

    next_ret = float(y1_scaler.inverse_transform(r1_pred)[0, 0])
    next_ret = np.clip(next_ret, -0.20, 0.20)
    if np.isnan(next_ret) or np.isinf(next_ret):
        next_ret = 0.0

    return {
        "date":        (last_date + BDay(1)).date(),
        "pred_close":  last_close * np.exp(next_ret),
        "pred_logret": next_ret,
        "dir_prob":    float(dirp[0, 0]),
        "intensity":   float(inten[0, 0]),
        "last_close":  last_close
    }


def run_predict_only(buckets, bucket_feature_union, cutoff_date, all_predictions):
    """
    predict_only: load the best available checkpoint for each stock (daily > weekly > monthly > yearly)
    and run forecast without any training. Writes predictions to master CSV.
    """
    log.info(f"\n{'='*60}")
    log.info(f"[PREDICT ONLY] cutoff={cutoff_date.date()}")
    log.info(f"{'='*60}")
    cutoff_str      = str(cutoff_date.date())
    vol_levels_list = list(VOLATILITY_LEVELS.keys()) + ["UNKNOWN", "GENERIC"]
    vol_to_idx      = {v: i for i, v in enumerate(vol_levels_list)}

    stage_priority = ["daily_finetune", "weekly_finetune", "monthly_finetune", "yearly_pretrain"]

    for (vol_level, sector), items in buckets.items():
        feature_cols_union = bucket_feature_union[(vol_level, sector)]
        bucket_tag = f"{vol_level}_{sector}"
        vol_idx    = vol_to_idx.get(vol_level, vol_to_idx["UNKNOWN"])

        for (sym, df_full, fcols, bcols) in items:
            df = filter_df_to_cutoff(df_full, cutoff_date)
            for c in feature_cols_union:
                if c not in df.columns:
                    df[c] = 0.0

            model       = None
            meta_pkl    = None
            stage_used  = "none"
            ckpt_path_used = ""

            for stage in stage_priority:
                ckpt_path, stage_meta = load_latest_successful_checkpoint(stage, bucket_tag, symbol=sym)
                if ckpt_path is None:
                    continue
                meta_path = os.path.join(os.path.dirname(ckpt_path), "meta.pkl")
                if not os.path.isfile(meta_path):
                    continue
                try:
                    with open(meta_path, "rb") as f:
                        meta_pkl = pickle.load(f)
                    n_features = len(meta_pkl["feature_cols"])
                    candidate  = build_multitask_model(SEQ_LEN, n_features, dropout_enc=DROPOUT_ENC_FINETUNE)
                    if os.path.isdir(ckpt_path):
                        candidate = tf.keras.models.load_model(
                            ckpt_path,
                            custom_objects={"GatingLayer": GatingLayer,
                                            "PositionalEncoding": PositionalEncoding}
                        )
                    else:
                        candidate.load_weights(ckpt_path)
                    model = candidate
                    stage_used = stage
                    ckpt_path_used = str(ckpt_path)
                    break
                except Exception as e:
                    log.warning(f"  [Predict] {sym} {stage} load failed: {e}")
                    continue

            if model is None or meta_pkl is None:
                log.info(f"  [Skip predict] {sym}: no usable checkpoint found.")
                continue

            fc = forecast_1d(df, model, meta_pkl["feature_cols"],
                             meta_pkl["b_scaler"], meta_pkl["x_scaler"], meta_pkl["y1_scaler"],
                             meta_pkl["vol_idx"], SEQ_LEN)
            if fc is None:
                continue

            direction = "UP" if fc["dir_prob"] > 0.55 else ("DOWN" if fc["dir_prob"] < 0.45 else "NEUTRAL")
            all_predictions.append({
                "prediction_date":       str(fc["date"]),
                "cutoff_date":           cutoff_str,
                "symbol":                sym,
                "sector":                sector,
                "volatility_class":      vol_level,
                "bucket_name":           bucket_tag,
                "last_close":            round(fc["last_close"], 2),
                "predicted_price":       round(fc["pred_close"], 2),
                "predicted_return":      round(fc["pred_logret"] * 100, 4),
                "predicted_direction":   direction,
                "dir_prob":              round(fc["dir_prob"], 4),
                "intensity":             round(fc["intensity"], 4),
                "model_stage_used":      stage_used,
                "parent_checkpoint_path": ckpt_path_used,
                "data_window_start":     "",
                "data_window_end":       str(cutoff_date.date()),
                "rows_used":             len(df),
            })
