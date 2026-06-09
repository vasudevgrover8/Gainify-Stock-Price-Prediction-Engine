"""
Monthly fine-tuning stage.

Physically extracted from legacy/yearly.py.
Original monthly fine-tune logic is preserved.
"""

import os
import pickle
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from configs.model_config import SEQ_LEN, DROPOUT_ENC_FINETUNE
from configs.training_config import EPOCHS_MONTHLY_FT, LR_MONTHLY_FT, AUTO_RESOLVE_PARENT_CHECKPOINT
from configs.bucket_config import VOLATILITY_LEVELS

from gainify_stock_predictor.models.main_model import build_multitask_model
from gainify_stock_predictor.training.losses import LOSSES, LOSS_W_FT_M
from gainify_stock_predictor.training.trainer_utils import prepare_single_stock_arrays
from gainify_stock_predictor.checkpoints.checkpoint_manager import get_stage_output_dir, resolve_parent_checkpoint
from gainify_stock_predictor.checkpoints.metadata_manager import save_stage_metadata
from gainify_stock_predictor.prediction.predictor import forecast_1d


log = logging.getLogger(__name__)

def run_monthly_finetune(buckets, bucket_feature_union, cutoff_date, all_predictions):
    """
    Stage 2: Monthly fine-tuning per stock.
    Uses MONTHLY_LOOKBACK_DAYS of recent data up to cutoff_date.
    Loads from yearly pretrained encoder (parent checkpoint).
    Saves to Models/Monthly_Finetuned/<cutoff_date>/<bucket>/<symbol>/
    """
    log.info(f"\n{'='*60}")
    log.info(f"[MONTHLY FINETUNE] cutoff={cutoff_date.date()}  window={MONTHLY_LOOKBACK_DAYS}d")
    log.info(f"{'='*60}")
    cutoff_str      = str(cutoff_date.date())
    vol_levels_list = list(VOLATILITY_LEVELS.keys()) + ["UNKNOWN", "GENERIC"]
    vol_to_idx      = {v: i for i, v in enumerate(vol_levels_list)}

    for (vol_level, sector), items in buckets.items():
        feature_cols_union = bucket_feature_union[(vol_level, sector)]
        bucket_tag = f"{vol_level}_{sector}"
        vol_idx    = vol_to_idx.get(vol_level, vol_to_idx["UNKNOWN"])

        for (sym, df_full, fcols, bcols) in items:
            out_dir = os.path.join(STAGE_DIRS["monthly_finetune"], cutoff_str, bucket_tag, sym)
            os.makedirs(out_dir, exist_ok=True)

            # Apply monthly window
            df = apply_stage_window(df_full, "monthly_finetune", cutoff_date)
            for c in feature_cols_union:
                if c not in df.columns:
                    df[c] = 0.0

            if len(df) < MIN_HISTORY_BARS:
                log.info(f"  [Skip monthly FT] {sym}: only {len(df)} rows after window filter.")
                continue

            log.info(f"\n[Monthly FT] {sym}  (vol={vol_level}, sector={sector}) rows={len(df)}")
            pack = prepare_single_stock_arrays(df, feature_cols_union, SEQ_LEN)
            if pack is None:
                log.info(f"  [Skip monthly FT] {sym}: insufficient data for train/val split")
                continue

            n_features = pack["X_tr_seq"].shape[2]
            model      = build_multitask_model(SEQ_LEN, n_features, dropout_enc=DROPOUT_ENC_FINETUNE)

            monthly_ckpt = os.path.join(out_dir, "best.weights.h5")
            if os.path.exists(monthly_ckpt):
                try:
                    model.load_weights(monthly_ckpt)
                    log.info(f"  [Resume] Loaded monthly weights for {sym}")
                except Exception:
                    pass
            elif AUTO_RESOLVE_PARENT_CHECKPOINT:
                # Load from yearly pretrained encoder
                parent_path, parent_meta = resolve_parent_checkpoint(
                    "monthly_finetune", bucket_tag, symbol=None)
                if parent_path and os.path.isfile(parent_path):
                    try:
                        model.get_layer("advanced_encoder").load_weights(parent_path)
                        log.info(f"  [Parent] Loaded yearly encoder: {parent_path}")
                    except Exception as e:
                        log.warning(f"  [Warn] Could not load parent encoder: {e}")

            vol_tr = np.full((pack["X_tr_seq"].shape[0], 1), vol_idx, dtype=np.int32)
            vol_te = np.full((pack["X_te_seq"].shape[0], 1), vol_idx, dtype=np.int32)

            model.compile(optimizer=tf.keras.optimizers.Adam(LR_MONTHLY_FT, clipnorm=1.0),
                          loss=LOSSES, loss_weights=LOSS_W_FT_M)

            model.fit(
                {"price_seq": pack["X_tr_seq"], "break_feats": pack["B_tr_seq"], "vol_level": vol_tr},
                {"r1": pack["y1_tr_seq"], "dir": pack["dir_tr_seq"], "int": pack["int_tr_seq"]},
                validation_data=(
                    {"price_seq": pack["X_te_seq"], "break_feats": pack["B_te_seq"], "vol_level": vol_te},
                    {"r1": pack["y1_te_seq"], "dir": pack["dir_te_seq"], "int": pack["int_te_seq"]}
                ),
                epochs=EPOCHS_MONTHLY_FT,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
                    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
                    ModelCheckpoint(monthly_ckpt, monitor="val_loss", save_best_only=True,
                                    save_weights_only=True, verbose=0)
                ],
                verbose=1
            )

            if os.path.exists(monthly_ckpt):
                model.load_weights(monthly_ckpt)

            model_path = os.path.join(out_dir, "model")
            model.save(model_path)

            parent_path_str = str(resolve_parent_checkpoint("monthly_finetune", bucket_tag)[0])
            window_start = df["Date"].min() if "Date" in df.columns else None
            window_end   = df["Date"].max() if "Date" in df.columns else cutoff_date
            save_stage_metadata(out_dir, "monthly_finetune", cutoff_date,
                                window_start, window_end, parent_path_str, bucket_tag, sym)

            meta = {
                "feature_cols": feature_cols_union, "break_cols": bcols,
                "x_scaler": pack["x_scaler"], "b_scaler": pack["b_scaler"],
                "y1_scaler": pack["y1_scaler"], "seq_len": SEQ_LEN, "vol_idx": vol_idx
            }
            with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
                pickle.dump(meta, f)
            log.info(f"  [Monthly FT] {sym}: saved -> {model_path}")

            # Forecast for CSV output
            fc = forecast_1d(df, model, feature_cols_union,
                             pack["b_scaler"], pack["x_scaler"], pack["y1_scaler"],
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
                    "model_stage_used":      "monthly_finetune",
                    "parent_checkpoint_path": parent_path_str,
                    "data_window_start":     str(window_start) if window_start is not None else "",
                    "data_window_end":       str(window_end)   if window_end   is not None else "",
                    "rows_used":             len(df),
                })
