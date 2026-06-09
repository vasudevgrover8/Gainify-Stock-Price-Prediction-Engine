"""
Yearly pretraining stage.

Physically extracted from legacy/yearly.py.
Original yearly pretrain logic is preserved.
"""

import os
import pickle
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from configs.model_config import SEQ_LEN
from configs.training_config import EPOCHS_PT, BATCH, LR_PT
from configs.bucket_config import VOLATILITY_LEVELS

from gainify_stock_predictor.models.main_model import build_pretrain_model
from gainify_stock_predictor.training.losses import LOSSES, LOSS_W_PRE
from gainify_stock_predictor.training.trainer_utils import build_and_save_sequences_for_stock
from gainify_stock_predictor.checkpoints.metadata_manager import save_stage_metadata


log = logging.getLogger(__name__)

def pretrain_bucket(items, feature_cols_union, vol_level, sector, out_dir,
                    cutoff_date=None):
    os.makedirs(out_dir, exist_ok=True)
    npz_dir = os.path.join(out_dir, "npz_cache")
    os.makedirs(npz_dir, exist_ok=True)

    vol_levels_list = list(VOLATILITY_LEVELS.keys()) + ["UNKNOWN", "GENERIC"]
    vol_to_idx      = {v: i for i, v in enumerate(vol_levels_list)}
    vol_idx         = vol_to_idx.get(vol_level, vol_to_idx["UNKNOWN"])

    train_files, val_files = [], []
    for (sym, df, _fcols, _bcols) in items:
        res = build_and_save_sequences_for_stock(
            df, feature_cols_union, sym, npz_dir,
            seq_len=SEQ_LEN, stage="yearly_pretrain", cutoff_date=cutoff_date
        )
        if res is None:
            continue
        tr_fp, val_fp = res
        if tr_fp:  train_files.append(tr_fp)
        if val_fp: val_files.append(val_fp)

    if not train_files:
        log.info(f"  [Skip] No train sequences for bucket {vol_level}/{sector}")
        return None

    n_features = len(feature_cols_union)

    def make_ds(file_list, shuffle=False):
        def gen():
            for fp in file_list:
                data = np.load(fp)
                X, B = data["X"].astype(np.float32), data["B"].astype(np.float32)
                y1, d, inten = (data["y1"].astype(np.float32),
                                data["dir"].astype(np.float32),
                                data["inten"].astype(np.float32))
                for i in range(X.shape[0]):
                    yield (
                        {"price_seq":   X[i],
                         "break_feats": B[i],
                         "vol_level":   np.array([vol_idx], dtype=np.int32)},
                        {"r1": y1[i].reshape(1,), "dir": d[i].reshape(1,), "int": inten[i].reshape(1,)}
                    )
        ds = tf.data.Dataset.from_generator(gen, output_signature=(
            {"price_seq":   tf.TensorSpec((SEQ_LEN, n_features), tf.float32),
             "break_feats": tf.TensorSpec((3,), tf.float32),
             "vol_level":   tf.TensorSpec((1,), tf.int32)},
            {"r1":  tf.TensorSpec((1,), tf.float32),
             "dir": tf.TensorSpec((1,), tf.float32),
             "int": tf.TensorSpec((1,), tf.float32)}
        ))
        if shuffle:
            ds = ds.shuffle(20000)
        return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(train_files, shuffle=True)
    val_ds   = make_ds(val_files,   shuffle=False) if val_files else None

    model     = build_pretrain_model(SEQ_LEN, n_features, dropout_enc=DROPOUT_ENC_PRETRAIN)
    ckpt_path = os.path.join(out_dir, "pretrain_best.weights.h5")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_PT, clipnorm=1.0),
        loss=LOSSES, loss_weights=LOSS_W_PRE
    )

    callbacks = [
        EarlyStopping(monitor="val_loss" if val_ds else "loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss" if val_ds else "loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
        ModelCheckpoint(ckpt_path, monitor="val_loss" if val_ds else "loss",
                        save_best_only=True, save_weights_only=True, verbose=0)
    ]

    model.fit(train_ds, epochs=EPOCHS_PT, validation_data=val_ds, callbacks=callbacks, verbose=1)

    enc      = model.get_layer("advanced_encoder")
    enc_path = os.path.join(out_dir, "encoder.weights.h5")
    enc.save_weights(enc_path)

    bucket_tag = f"{vol_level}_{sector}"
    window_start = df["Date"].min() if "Date" in items[0][1].columns else None
    window_end   = cutoff_date
    save_stage_metadata(out_dir, "yearly_pretrain", cutoff_date,
                        window_start, window_end, None, bucket_tag)
    return enc_path


def run_yearly_pretrain(buckets, bucket_feature_union, cutoff_date):
    """
    Stage 1: Yearly pretraining per bucket.
    Uses full history up to cutoff_date.
    Saves encoder weights + metadata to Models/Yearly_Pretrained/<cutoff_date>/<bucket>/
    """
    log.info(f"\n{'='*60}")
    log.info(f"[YEARLY PRETRAIN] cutoff={cutoff_date.date()}")
    log.info(f"{'='*60}")
    cutoff_str = str(cutoff_date.date())
    bucket_encoder_path = {}

    for (vol_level, sector), items in buckets.items():
        feature_cols_union = bucket_feature_union[(vol_level, sector)]
        bucket_tag = f"{vol_level}_{sector}"
        out_dir    = os.path.join(STAGE_DIRS["yearly_pretrain"], cutoff_str, bucket_tag)
        os.makedirs(out_dir, exist_ok=True)

        enc_path = os.path.join(out_dir, "encoder.weights.h5")
        if os.path.exists(enc_path) and os.path.exists(os.path.join(out_dir, "stage_meta.json")):
            log.info(f"[Skip pretrain] ({vol_level}/{sector}) - checkpoint exists for {cutoff_str}")
            bucket_encoder_path[(vol_level, sector)] = enc_path
            continue

        log.info(f"\n[Pretrain] ({vol_level}/{sector}) | {len(items)} stocks | {len(feature_cols_union)} features")
        enc_path = pretrain_bucket(items, feature_cols_union, vol_level, sector, out_dir,
                                   cutoff_date=cutoff_date)
        if enc_path:
            bucket_encoder_path[(vol_level, sector)] = enc_path

    return bucket_encoder_path
