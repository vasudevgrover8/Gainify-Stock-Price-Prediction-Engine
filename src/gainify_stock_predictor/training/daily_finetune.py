"""
Daily fine-tuning stage.

Physically extracted from legacy/yearly.py.
Original daily fine-tune logic is preserved.
"""

import os
import pickle
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from configs.model_config import SEQ_LEN, DROPOUT_ENC_FINETUNE
from configs.training_config import EPOCHS_DAILY_FT, LR_DAILY_FT, AUTO_RESOLVE_PARENT_CHECKPOINT
from configs.bucket_config import VOLATILITY_LEVELS

from gainify_stock_predictor.models.main_model import build_multitask_model
from gainify_stock_predictor.models.layers import GatingLayer, PositionalEncoding
from gainify_stock_predictor.training.losses import LOSSES, LOSS_W_DAILY
from gainify_stock_predictor.training.trainer_utils import prepare_daily_arrays
from gainify_stock_predictor.checkpoints.checkpoint_manager import get_stage_output_dir, resolve_parent_checkpoint
from gainify_stock_predictor.checkpoints.metadata_manager import save_stage_metadata
from gainify_stock_predictor.prediction.predictor import forecast_1d


log = logging.getLogger(__name__)

def run_daily_finetune(buckets, bucket_feature_union, cutoff_date, all_predictions):
    """
    Stage 4: Daily fine-tuning per stock.
    Uses DAILY_LOOKBACK_DAYS of recent data up to cutoff_date.
    Loads from weekly finetuned checkpoint (parent). Falls back through hierarchy.
    Saves to Models/Daily_Finetuned/<cutoff_date>/<bucket>/<symbol>/

    Running this stage does NOT trigger weekly/monthly/yearly reruns.
    Appending new rows to the dataset will shift the cutoff when this
    stage is next explicitly run.
    """
    log.info(f"\n{'='*60}")
    log.info(f"[DAILY FINETUNE] cutoff={cutoff_date.date()}  window={DAILY_LOOKBACK_DAYS}d")
    log.info(f"{'='*60}")
    cutoff_str      = str(cutoff_date.date())
    vol_levels_list = list(VOLATILITY_LEVELS.keys()) + ["UNKNOWN", "GENERIC"]
    vol_to_idx      = {v: i for i, v in enumerate(vol_levels_list)}

    for (vol_level, sector), items in buckets.items():
        feature_cols_union = bucket_feature_union[(vol_level, sector)]
        bucket_tag = f"{vol_level}_{sector}"
        vol_idx    = vol_to_idx.get(vol_level, vol_to_idx["UNKNOWN"])

        for (sym, df_full, fcols, bcols) in items:
            out_dir = os.path.join(STAGE_DIRS["daily_finetune"], cutoff_str, bucket_tag, sym)
            os.makedirs(out_dir, exist_ok=True)

            df = apply_stage_window(df_full, "daily_finetune", cutoff_date)
            for c in feature_cols_union:
                if c not in df.columns:
                    df[c] = 0.0

            if len(df) < MIN_HISTORY_BARS:
                log.info(f"  [Skip daily FT] {sym}: only {len(df)} rows after window filter.")
                continue

            log.info(f"\n[Daily FT] {sym}  (vol={vol_level}) rows={len(df)}")
            pack_daily = prepare_daily_arrays(df, feature_cols_union, SEQ_LEN, daily_window=60)
            if pack_daily is None or pack_daily["X_tr_seq"].shape[0] < 5:
                log.info(f"  [Skip daily FT] {sym}: insufficient sequences")
                continue

            n_features = pack_daily["X_tr_seq"].shape[2]
            model      = build_multitask_model(SEQ_LEN, n_features, dropout_enc=DROPOUT_ENC_FINETUNE)

            daily_ckpt = os.path.join(out_dir, "best.weights.h5")
            parent_path_str = ""
            if os.path.exists(daily_ckpt):
                try:
                    model.load_weights(daily_ckpt)
                    log.info(f"  [Resume] Loaded daily weights for {sym}")
                except Exception:
                    pass
            elif AUTO_RESOLVE_PARENT_CHECKPOINT:
                # Load from latest weekly checkpoint (symbol-level preferred)
                parent_path, _ = resolve_parent_checkpoint(
                    "daily_finetune", bucket_tag, symbol=sym)
                if parent_path:
                    parent_path_str = str(parent_path)
                    try:
                        if os.path.isdir(parent_path):
                            parent_model = tf.keras.models.load_model(
                                parent_path,
                                custom_objects={"GatingLayer": GatingLayer,
                                                "PositionalEncoding": PositionalEncoding}
                            )
                            model.set_weights(parent_model.get_weights())
                        elif os.path.isfile(parent_path):
                            model.load_weights(parent_path)
                        log.info(f"  [Parent] Loaded weekly checkpoint: {parent_path}")
                    except Exception as e:
                        log.warning(f"  [Warn] Could not load parent: {e}")

            model.compile(optimizer=tf.keras.optimizers.Adam(LR_DAILY_FT, clipnorm=1.0),
                          loss=LOSSES, loss_weights=LOSS_W_DAILY)

            vol_tr_d = np.full((pack_daily["X_tr_seq"].shape[0], 1), vol_idx, dtype=np.int32)
            vol_te_d = np.full((pack_daily["X_te_seq"].shape[0], 1), vol_idx, dtype=np.int32)
            has_daily_val = pack_daily["X_te_seq"].shape[0] > 0

            model.fit(
                {"price_seq": pack_daily["X_tr_seq"], "break_feats": pack_daily["B_tr_seq"], "vol_level": vol_tr_d},
                {"r1": pack_daily["y1_tr_seq"], "dir": pack_daily["dir_tr_seq"], "int": pack_daily["int_tr_seq"]},
                validation_data=(
                    {"price_seq": pack_daily["X_te_seq"], "break_feats": pack_daily["B_te_seq"], "vol_level": vol_te_d},
                    {"r1": pack_daily["y1_te_seq"], "dir": pack_daily["dir_te_seq"], "int": pack_daily["int_te_seq"]}
                ) if has_daily_val else None,
                epochs=EPOCHS_DAILY_FT,
                callbacks=[
                    EarlyStopping(monitor="val_loss" if has_daily_val else "loss",
                                  patience=3, restore_best_weights=True),
                    ModelCheckpoint(daily_ckpt, monitor="val_loss" if has_daily_val else "loss",
                                    save_best_only=True, save_weights_only=True, verbose=0)
                ],
                verbose=0
            )

            if os.path.exists(daily_ckpt):
                model.load_weights(daily_ckpt)

            model_path = os.path.join(out_dir, "model")
            model.save(model_path)

            window_start = df["Date"].min() if "Date" in df.columns else None
            window_end   = df["Date"].max() if "Date" in df.columns else cutoff_date
            save_stage_metadata(out_dir, "daily_finetune", cutoff_date,
                                window_start, window_end, parent_path_str, bucket_tag, sym)

            meta = {
                "feature_cols": feature_cols_union, "break_cols": bcols,
                "x_scaler": pack_daily["x_scaler"], "b_scaler": pack_daily["b_scaler"],
                "y1_scaler": pack_daily["y1_scaler"], "seq_len": SEQ_LEN, "vol_idx": vol_idx
            }
            with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
                pickle.dump(meta, f)

            # Forecast + collect for master CSV
            fc = forecast_1d(df, model, feature_cols_union,
                             pack_daily["b_scaler"], pack_daily["x_scaler"], pack_daily["y1_scaler"],
                             vol_idx, SEQ_LEN)
            if fc is not None:
                direction = "UP" if fc["dir_prob"] > 0.55 else ("DOWN" if fc["dir_prob"] < 0.45 else "NEUTRAL")
                all_predictions.append({
                    "prediction_date":       str(fc["date"]),
                    "cutoff_date":           str(cutoff_date.date()),
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
                    "model_stage_used":      "daily_finetune",
                    "parent_checkpoint_path": parent_path_str,
                    "data_window_start":     str(window_start) if window_start is not None else "",
                    "data_window_end":       str(window_end)   if window_end   is not None else "",
                    "rows_used":             len(df),
                })
