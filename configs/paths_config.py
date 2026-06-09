"""
Path configuration for Gainify Stock Predictor.

Source of truth:
- Constants moved from yearly.py
- No training/model logic changed
"""

import os
from pathlib import Path


# ---------------------------------------------------------------
# Project root
# ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------
# Original working directory
# ---------------------------------------------------------------
BASE_DIR = r"C:\Stock Price Predictor"

DATA_DIR = os.path.join(BASE_DIR, "Stock_Data", "historical")
MODEL_DIR = os.path.join(BASE_DIR, "Models")
OUTPUT_DIR = os.path.join(BASE_DIR, "Outputs")


# ---------------------------------------------------------------
# GitHub project-local paths
# ---------------------------------------------------------------
REPO_DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = REPO_DATA_DIR / "raw"
PROCESSED_DATA_DIR = REPO_DATA_DIR / "processed"
SAMPLE_DATA_DIR = REPO_DATA_DIR / "sample"

REPO_MODEL_DIR = PROJECT_ROOT / "models"
REPO_OUTPUT_DIR = PROJECT_ROOT / "outputs"

PREDICTIONS_DIR = REPO_OUTPUT_DIR / "predictions"
REPORTS_DIR = REPO_OUTPUT_DIR / "reports"
LOGS_DIR = REPO_OUTPUT_DIR / "logs"


# ---------------------------------------------------------------
# Stage checkpoint directories
# ---------------------------------------------------------------
STAGE_DIRS = {
    "yearly_pretrain": os.path.join(MODEL_DIR, "Yearly_Pretrained"),
    "monthly_finetune": os.path.join(MODEL_DIR, "Monthly_Finetuned"),
    "weekly_finetune": os.path.join(MODEL_DIR, "Weekly_Finetuned"),
    "daily_finetune": os.path.join(MODEL_DIR, "Daily_Finetuned"),
}

STAGE_PARENT = {
    "monthly_finetune": "yearly_pretrain",
    "weekly_finetune": "monthly_finetune",
    "daily_finetune": "weekly_finetune",
}


def ensure_project_dirs():
    """
    Creates project-local folders only.
    This does not alter your original C:\\Stock Price Predictor logic.
    """
    dirs = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAMPLE_DATA_DIR,
        REPO_MODEL_DIR / "yearly_pretrained",
        REPO_MODEL_DIR / "monthly_finetuned",
        REPO_MODEL_DIR / "weekly_finetuned",
        REPO_MODEL_DIR / "daily_finetuned",
        PREDICTIONS_DIR,
        REPORTS_DIR,
        LOGS_DIR,
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)