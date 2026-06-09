"""
Loss functions and loss weights.

Physically extracted from legacy/yearly.py.
Original loss logic and weights are preserved.

Important:
Functions must be defined before LOSSES dictionary.
"""

import tensorflow as tf



def spike_weighted_mse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    w = 1.5 + 3.5 * tf.pow(tf.abs(y_true), 1.5)
    return tf.reduce_mean(w * tf.square(y_true - y_pred))


def smooth_huber(y_true, y_pred, delta=0.05):
    y_true  = tf.cast(y_true, tf.float32)
    y_pred  = tf.cast(y_pred, tf.float32)
    err     = y_true - y_pred
    abs_err = tf.abs(err)
    loss    = tf.where(abs_err < delta, 0.5 * tf.square(err) / delta, abs_err - 0.5 * delta)
    return tf.reduce_mean(loss)


def focal_bce_soft(y_true, y_pred, alpha=0.5, gamma=2.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    eps    = 1e-7
    y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
    pt     = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
    w      = alpha * tf.pow(1.0 - pt, gamma)
    return -tf.reduce_mean(w * tf.math.log(pt))


LOSSES       = {"r1": spike_weighted_mse, "dir": focal_bce_soft, "int": smooth_huber}


LOSS_W_PRE   = {"r1": 0.20, "dir": 0.70, "int": 0.10}


LOSS_W_FT_M  = {"r1": 0.22, "dir": 0.65, "int": 0.13}


LOSS_W_FT    = {"r1": 0.25, "dir": 0.60, "int": 0.15}


LOSS_W_DAILY = {"r1": 0.30, "dir": 0.55, "int": 0.15}
