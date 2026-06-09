"""
Training configuration for Gainify Stock Predictor.

Moved from yearly.py without changing values.
"""

# ---------------------------------------------------------------
# Stage control
# ---------------------------------------------------------------
RUN_STAGE = "daily_finetune"

AUTO_DETECT_CUTOFF_DATE = True
AUTO_RESOLVE_PARENT_CHECKPOINT = True
ENABLE_VISUALS = False


# ---------------------------------------------------------------
# Stage-specific rolling lookback windows
# ---------------------------------------------------------------
MONTHLY_LOOKBACK_DAYS = 365
WEEKLY_LOOKBACK_DAYS = 180
DAILY_LOOKBACK_DAYS = 90


# ---------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------
SEED = 42


# ---------------------------------------------------------------
# Training knobs
# ---------------------------------------------------------------
EPOCHS_PT = 35
EPOCHS_MONTHLY_FT = 25
EPOCHS_WEEKLY_FT = 20
EPOCHS_DAILY_FT = 5

BATCH = 32

LR_PT = 8e-4
LR_MONTHLY_FT = 3e-4
LR_WEEKLY_FT = 2e-4
LR_DAILY_FT = 5e-5

EMBARGO_STEPS = 20