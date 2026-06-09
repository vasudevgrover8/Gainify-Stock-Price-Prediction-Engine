"""
Weekly fine-tuning stage.

Physically extracted from legacy/yearly.py.
Original weekly fine-tune logic is preserved.
"""

import os
import pickle
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from configs.model_config import SEQ_LEN, DROPOUT_ENC_FINETUNE
from configs.training_config import EPOCHS_WEEKLY_FT, LR_WEEKLY_FT, AUTO_RESOLVE_PARENT_CHECKPOINT
from configs.bucket_config import VOLATILITY_LEVELS

from gainify_stock_predictor.models.main_model import build_multitask_model
from gainify_stock_predictor.training.losses import LOSSES, LOSS_W_FT
from gainify_stock_predictor.training.trainer_utils import prepare_single_stock_arrays
from gainify_stock_predictor.checkpoints.checkpoint_manager import get_stage_output_dir, resolve_parent_checkpoint
from gainify_stock_predictor.checkpoints.metadata_manager import save_stage_metadata
from gainify_stock_predictor.prediction.predictor import forecast_1d


log = logging.getLogger(__name__)

def run_weekly_finetune(buckets, bucket_feature_union, cutoff_date, all_predictions):
    """
    Stage 3: Weekly fine-tuning per stock.
    Uses WEEKLY_LOOKBACK_DAYS of recent data up to cutoff_date.
    Loads from monthly finetuned checkpoint (parent). Falls back to yearly if monthly missing.
    Saves to Models/Weekly_Finetuned/<cutoff_date>/<bucket>/<symbol>/
    """
    log.info(f"\n{'='*60}")
    log.info(f"[WEEKLY FINETUNE] cutoff={cutoff_date.date()}  window={WEEKLY_LOOKBACK_DAYS}d")
    log.info(f"{'='*60}")
    cutoff_str      = str(cutoff_date.date())
    vol_levels_list = list(VOLATILITY_LEVELS.keys()) + ["UNKNOWN", "GENERIC"]
    vol_to_idx      = {v: i for i, v in enumerate(vol_levels_list)}

    for (vol_level, sector), items in buckets.items():
        feature_cols_union = bucket_feature_union[(vol_level, sector)]
        bucket_tag = f"{vol_level}_{sector}"
        vol_idx    = vol_to_idx.get(vol_level, vol_to_idx["UNKNOWN"])

        for (sym, df_full, fcols, bcols) in items:
            out_dir = os.path.join(STAGE_DIRS["weekly_finetune"], cutoff_str, bucket_tag, sym)
            os.makedirs(out_dir, exist_ok=True)

            df = apply_stage_window(df_full, "weekly_finetune", cutoff_date)
            for c in feature_cols_union:
                if c not in df.columns:
                    df[c] = 0.0

            if len(df) < MIN_HISTORY_BARS:
                log.info(f"  [Skip weekly FT] {sym}: only {len(df)} rows after window filter.")
                continue

            log.info(f"\n[Weekly FT] {sym}  (vol={vol_level}, sector={sector}) rows={len(df)}")
            pack = prepare_single_stock_arrays(df, feature_cols_union, SEQ_LEN)
            if pack is None:
                log.info(f"  [Skip weekly FT] {sym}: insufficient data")
                continue

            n_features = pack["X_tr_seq"].shape[2]
            model      = build_multitask_model(SEQ_LEN, n_features, dropout_enc=DROPOUT_ENC_FINETUNE)

            weekly_ckpt = os.path.join(out_dir, "best.weights.h5")
            parent_path_str = ""
            if os.path.exists(weekly_ckpt):
                try:
                    model.load_weights(weekly_ckpt)
                    log.info(f"  [Resume] Loaded weekly weights for {sym}")
                except Exception:
                    pass
            elif AUTO_RESOLVE_PARENT_CHECKPOINT:
                # Load from monthly (symbol-level first, then bucket-level encoder)
                parent_path, parent_meta = resolve_parent_checkpoint(
                    "weekly_finetune", bucket_tag, symbol=sym)
                if parent_path:
                    parent_path_str = str(parent_path)
                    try:
                        if os.path.isdir(parent_path):
                            parent_model = tf.keras.models.load_model(
                                parent_path,
                                custom_objects={"GatingLayer": GatingLayer,
                                                "PositionalEncoding": PositionalEncoding}
                            )
                            model.get_layer("advanced_encoder").set_weights(
                                parent_model.get_layer("advanced_encoder").get_weights())
                        elif os.path.isfile(parent_path):
                            model.get_layer("advanced_encoder").load_weights(parent_path)
                        log.info(f"  [Parent] Loaded monthly checkpoint: {parent_path}")
                    except Exception as e:
                        log.warning(f"  [Warn] Could not load parent: {e}")

            vol_tr = np.full((pack["X_tr_seq"].shape[0], 1), vol_idx, dtype=np.int32)
            vol_te = np.full((pack["X_te_seq"].shape[0], 1), vol_idx, dtype=np.int32)

            model.compile(optimizer=tf.keras.optimizers.Adam(LR_WEEKLY_FT, clipnorm=1.0),
                          loss=LOSSES, loss_weights=LOSS_W_FT)

            model.fit(
                {"price_seq": pack["X_tr_seq"], "break_feats": pack["B_tr_seq"], "vol_level": vol_tr},
                {"r1": pack["y1_tr_seq"], "dir": pack["dir_tr_seq"], "int": pack["int_tr_seq"]},
                validation_data=(
                    {"price_seq": pack["X_te_seq"], "break_feats": pack["B_te_seq"], "vol_level": vol_te},
                    {"r1": pack["y1_te_seq"], "dir": pack["dir_te_seq"], "int": pack["int_te_seq"]}
                ),
                epochs=EPOCHS_WEEKLY_FT,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
                    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
                    ModelCheckpoint(weekly_ckpt, monitor="val_loss", save_best_only=True,
                                    save_weights_only=True, verbose=0)
                ],
                verbose=1
            )

            if os.path.exists(weekly_ckpt):
                model.load_weights(weekly_ckpt)

            model_path = os.path.join(out_dir, "model")
            model.save(model_path)

            window_start = df["Date"].min() if "Date" in df.columns else None
            window_end   = df["Date"].max() if "Date" in df.columns else cutoff_date
            save_stage_metadata(out_dir, "weekly_finetune", cutoff_date,
                                window_start, window_end, parent_path_str, bucket_tag, sym)

            meta = {
                "feature_cols": feature_cols_union, "break_cols": bcols,
                "x_scaler": pack["x_scaler"], "b_scaler": pack["b_scaler"],
                "y1_scaler": pack["y1_scaler"], "seq_len": SEQ_LEN, "vol_idx": vol_idx
            }
            with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
                pickle.dump(meta, f)

            # Evaluate (metrics only)
            if pack["X_te_seq"].shape[0] >= 10:
                evaluate_holdout_close(df, pack, model, vol_idx, sym)

            # Tree models
            X_tab, y_tab = build_tabular_dataset_1d(df, feature_cols_union)
            lgbm_model, xgb_model = None, None
            if len(X_tab) > 100:
                lgbm_model = train_lgbm_1d(X_tab, y_tab)
                xgb_model  = train_xgb_1d(X_tab, y_tab)
                tree_dir   = os.path.join(out_dir, "tree_models")
                os.makedirs(tree_dir, exist_ok=True)
                joblib.dump(lgbm_model, os.path.join(tree_dir, "lgbm_1d.pkl"))
                xgb_model.save_model(os.path.join(tree_dir, "xgb_1d.json"))
                log.info(f"  [Tree] LGBM + XGBoost saved for {sym}")

            # Forecast
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
                    "model_stage_used":      "weekly_finetune",
                    "parent_checkpoint_path": parent_path_str,
                    "data_window_start":     str(window_start) if window_start is not None else "",
                    "data_window_end":       str(window_end)   if window_end   is not None else "",
                    "rows_used":             len(df),
                })
