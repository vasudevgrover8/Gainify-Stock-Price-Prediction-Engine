"""
Temporary safe reference wrapper for legacy yearly.py.

IMPORTANT:
- We are not importing yearly.py directly here.
- yearly.py contains the original working orchestrator logic.
- Later, we will split its functions carefully into preprocessing, features,
  bucketing, models, training, checkpoints, prediction, and reporting modules.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_YEARLY_PATH = PROJECT_ROOT / "legacy" / "yearly.py"


def get_legacy_yearly_path():
    """
    Return the path to the original yearly.py file copied into legacy/.
    """
    return LEGACY_YEARLY_PATH


def legacy_yearly_exists():
    """
    Check whether legacy/yearly.py exists.
    """
    return LEGACY_YEARLY_PATH.exists()