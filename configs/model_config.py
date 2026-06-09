"""
Model configuration for Gainify Stock Predictor.

Moved from yearly.py without changing values.
"""

# ---------------------------------------------------------------
# Model / sequence configuration
# ---------------------------------------------------------------
SEQ_LEN = 90

L2REG = 1e-6

DROPOUT_ENC_PRETRAIN = 0.35
DROPOUT_ENC_FINETUNE = 0.20
DROPOUT_HEAD = 0.25

FREEZE_ENCODER_LAYERS = False

LABEL_SMOOTH_EPS = 0.01

MIN_HISTORY_BARS = 60