"""
Signal and verdict builders.

Physically extracted from legacy/yearly.py.
Original logic is preserved.
"""

import numpy as np
import pandas as pd

def calculate_verdict(dir_probs, intensity_values, price_changes):
    avg_dir    = np.mean(dir_probs)
    avg_int    = np.mean(np.abs(intensity_values))
    avg_change = np.mean(price_changes)
    if avg_dir > 0.65 and avg_change > 1.0:
        return 'BULLISH', min(95, 60 + avg_int * 30), 'Strong upward momentum with high conviction'
    elif avg_dir < 0.35 and avg_change < -1.0:
        return 'BEARISH', min(95, 60 + avg_int * 30), 'Strong downward pressure with high conviction'
    elif avg_dir > 0.55 and avg_change > 0.3:
        return 'BULLISH', min(75, 50 + avg_int * 20), 'Moderate upward trend with positive signals'
    elif avg_dir < 0.45 and avg_change < -0.3:
        return 'BEARISH', min(75, 50 + avg_int * 20), 'Moderate downward trend with negative signals'
    else:
        return 'NEUTRAL', 40 + 20 * min(1.0, avg_int), 'Mixed signals with no clear directional bias'


def ensemble_next_day_signal(df, feature_cols, meta, dl_model,
                             lgbm=None, xgb_model=None,
                             ret_threshold=0.003, prob_band=(0.48, 0.52)):
    seq_len   = meta["seq_len"]
    x_scaler  = meta["x_scaler"]
    b_scaler  = meta["b_scaler"]
    y1_scaler = meta["y1_scaler"]
    vol_idx   = meta["vol_idx"]

    if len(df) < seq_len:
        return None

    feats   = df[feature_cols].values
    X_seq   = np.clip(x_scaler.transform(feats[-seq_len:]), -500.0, 500.0).reshape(
        1, seq_len, len(feature_cols)).astype(np.float32)
    B_seq   = np.clip(b_scaler.transform(
        df[["BreakReliab", "RevAfterHigh", "RevAfterLow"]].iloc[-1].values.reshape(1, -1)
    ), -500.0, 500.0).astype(np.float32)
    vol_arr = np.array([[vol_idx]], dtype=np.int32)

    preds    = dl_model.predict({"price_seq": X_seq, "break_feats": B_seq, "vol_level": vol_arr}, verbose=0)
    dir_prob = float(preds[1][0, 0])
    inten    = float(preds[2][0, 0])
    dl_ret   = float(y1_scaler.inverse_transform(preds[0])[0, 0])

    row      = df[feature_cols].iloc[[-1]].values
    lgbm_ret = float(lgbm.predict(row)[0]) if lgbm is not None else None
    xgb_ret  = None
    if xgb_model is not None:
        xgb_ret = float(xgb_model.predict(xgb.DMatrix(row))[0])

    rets    = [dl_ret];    weights = [0.6]
    if lgbm_ret is not None: rets.append(lgbm_ret); weights.append(0.25)
    if xgb_ret  is not None: rets.append(xgb_ret);  weights.append(0.15)
    weights   = np.array(weights) / np.sum(weights)
    final_ret = float(np.dot(weights, np.array(rets)))

    last_close   = float(df["Close"].iloc[-1])
    target_price = last_close * np.exp(final_ret)

    low, high = prob_band
    reasons   = []
    if low < dir_prob < high:          reasons.append("DL dir prob near 0.5")
    if abs(final_ret) < ret_threshold: reasons.append("Ensemble expected move too small")
    signs = [np.sign(dl_ret)]
    if lgbm_ret is not None: signs.append(np.sign(lgbm_ret))
    if xgb_ret  is not None: signs.append(np.sign(xgb_ret))
    if len(set(signs)) > 1:            reasons.append("Model disagreement (DL vs trees)")

    action     = "NO_TRADE" if reasons else ("LONG" if final_ret > 0 else "SHORT")
    confidence = (max(0.0, 1.0 - len(reasons) * 0.25) if reasons
                  else min(1.0, 0.6 + abs(inten) * 0.3))

    return {
        "action": action, "target_price": target_price,
        "expected_ret_pct": final_ret * 100,
        "dl_ret": dl_ret * 100,
        "lgbm_ret": None if lgbm_ret is None else lgbm_ret * 100,
        "xgb_ret":  None if xgb_ret  is None else xgb_ret  * 100,
        "dir_prob": dir_prob, "intensity": inten,
        "confidence": confidence, "no_trade_reasons": reasons
    }
