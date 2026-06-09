"""
Sequence-building helpers.

These functions are copied/adapted from yearly.py without changing behavior.
"""

import numpy as np
import pandas as pd

from configs.model_config import LABEL_SMOOTH_EPS


def cumulative_logret_forward(logret_series, horizon=1):
    """
    Compute forward cumulative log-return for a given horizon.
    """
    s = pd.Series(logret_series.reshape(-1))
    cum = s.rolling(window=horizon).sum().shift(-(horizon - 1))
    return cum.values.reshape(-1, 1)


def make_sequences_masked(X, y_dict, L, mask):
    """
    Build masked sequences.

    This preserves your original yearly.py behavior:
    - only creates sequences where mask[i] is True
    - keeps target dictionary structure
    """
    idxs = [i for i in range(L, len(X)) if mask[i]]

    if len(idxs) == 0:
        out = {"X": np.zeros((0, L, X.shape[1]), dtype=X.dtype)}

        for k, arr in y_dict.items():
            out[k] = np.zeros((0, arr.shape[1]), dtype=arr.dtype)

        return out, []

    Xs = np.stack([X[i - L:i] for i in idxs], axis=0)

    out = {"X": Xs}

    for k, arr in y_dict.items():
        out[k] = np.stack([arr[i] for i in idxs], axis=0)

    return out, idxs


def apply_label_smoothing(y, eps=LABEL_SMOOTH_EPS):
    """
    Apply binary label smoothing.
    """
    return (1 - eps) * y + eps * 0.5