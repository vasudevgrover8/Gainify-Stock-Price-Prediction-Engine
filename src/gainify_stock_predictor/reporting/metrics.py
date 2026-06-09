"""
Evaluation metric helpers.

Physically extracted from legacy/yearly.py.
Original evaluation logic is preserved.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_holdout_close(df, pack, model, vol_idx, symbol):
    if pack["X_te_seq"].shape[0] < 10:
        log.info(f"  [Abort Eval] {symbol}: n_val={pack['X_te_seq'].shape[0]} too small.")
        return {}
    preds = model.predict(
        {"price_seq":   pack["X_te_seq"],
         "break_feats": pack["B_te_seq"],
         "vol_level":   np.ones((pack["X_te_seq"].shape[0], 1), dtype=np.int32) * vol_idx},
        verbose=0
    )
    r1_pred     = preds[0]
    r1_pred_inv = pack["y1_scaler"].inverse_transform(r1_pred).ravel()
    y_true_inv  = pack["y1_scaler"].inverse_transform(pack["y1_te_seq"]).ravel()

    pred_close, true_close = [], []
    for k in range(len(r1_pred_inv)):
        actual_day_idx = pack["orig_idxs_te"][k]
        prev_day_idx   = actual_day_idx - 1
        if prev_day_idx < 0 or actual_day_idx >= len(df):
            continue
        base   = df.iloc[prev_day_idx]["Close"]
        actual = df.iloc[actual_day_idx]["Close"]
        pred_close.append(base * np.exp(r1_pred_inv[k]))
        true_close.append(actual)

    if not true_close:
        return {}

    mae  = mean_absolute_error(true_close, pred_close)
    rmse = np.sqrt(mean_squared_error(true_close, pred_close))
    r2   = r2_score(true_close, pred_close)

    dir_prob     = preds[1].ravel()
    neutral_band = 0.0015
    dir_label    = np.full_like(y_true_inv, -1, dtype=int)
    dir_label[y_true_inv >  neutral_band] = 1
    dir_label[y_true_inv < -neutral_band] = 0
    pred_dir = np.full_like(dir_prob, -1, dtype=int)
    pred_dir[dir_prob > 0.55] = 1
    pred_dir[dir_prob < 0.45] = 0
    mask = (dir_label != -1) & (pred_dir != -1)
    hit  = (pred_dir[mask] == dir_label[mask]).mean() * 100 if mask.sum() > 0 else 0.0

    log.info(f"\n{'='*60}")
    log.info(f"HOLDOUT METRICS: {symbol}")
    log.info(f"  MAE: Rs.{mae:.2f}  |  RMSE: Rs.{rmse:.2f}  |  R2: {r2:.4f}")
    log.info(f"  1-Day Direction Hit-Rate: {hit:.2f}%")
    log.info(f"{'='*60}\n")

    # Only produce charts if ENABLE_VISUALS is explicitly True
    if ENABLE_VISUALS:
        x_range = range(len(true_close))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        ax1.plot(x_range, true_close, linewidth=2.5, label='Actual', marker='o', markersize=3, alpha=0.8)
        ax1.plot(x_range, pred_close, linewidth=2.5, label='Predicted', marker='s', markersize=3, alpha=0.8, linestyle='--')
        ax1.set_title(f'{symbol} - Actual vs Predicted (Holdout)', fontsize=15, fontweight='bold')
        ax1.set_ylabel('Price (Rs.)', fontsize=12); ax1.legend()
        ax1.grid(True, alpha=0.3)
        accuracy_per_sample = (pred_dir == dir_label).astype(float) * 100
        ax2.bar(range(len(accuracy_per_sample)), accuracy_per_sample, alpha=0.7, edgecolor='black')
        ax2.axhline(50, color='gray', linestyle='--', linewidth=2)
        ax2.axhline(hit, color='purple', linestyle='-', linewidth=2, label=f'Avg {hit:.1f}%')
        ax2.set_title('Direction Accuracy per Sample', fontsize=13)
        ax2.set_ylim([0, 105]); ax2.legend()
        plt.tight_layout(); plt.show()

    return {"mae": mae, "rmse": rmse, "r2": r2, "dir_hit_rate": hit}
